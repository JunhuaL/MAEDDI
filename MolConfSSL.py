from typing import Any, List, Union
import numpy as np 
import warnings
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
warnings.filterwarnings('ignore')
import torch
import torch as t 
import torch.nn.functional as F
from torch import nn 
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from utils import *
from DeepGCN import *
from nt_xent import NTXentLoss

class CLRModel(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 enc_num_hidden: int, 
                 num_layers: int,
                 feat_dim: int,
                 in_edge_channel: int=11,
                 mid_edge_channel: int=128,
                 n_bins: int=6,
                 mask_rate: float = 0.5,
                 drop_edge_rate: float = 0,
                 ):
        super(CLRModel,self).__init__()

        self._mask_rate = mask_rate
        self._drop_edge_rate = drop_edge_rate

        self.mol_encoder = DeeperGCN(in_dim, enc_num_hidden, num_layers,1,
                                 dropout_ratio=0.1,embedding_layer=None,
                                 graph_conv=SAGEConvV2,
                                 in_edge_channel=None,
                                 mid_edge_channel=mid_edge_channel,aggr='softmax')
        
        self.conf_encoder = DeeperGCN(in_edge_channel,mid_edge_channel,num_layers,1,
                                      dropout_ratio=0.1,embedding_layer=None,
                                      graph_conv=SAGEConvV2,
                                      in_edge_channel=n_bins,
                                      mid_edge_channel=mid_edge_channel, aggr='softmax')

        self.feat_lin = Linear(enc_num_hidden+mid_edge_channel, feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(feat_dim,feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim,feat_dim//2)
        )
    
    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.num_nodes
        perm = t.randperm(num_nodes)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        out_x = x.clone()
        out_x.x[mask_nodes] = 0.0

        return out_x, (mask_nodes, keep_nodes)

    def encoding_edge_noise(self, x, mask_rate = 0.3):
        num_edges = x.num_edges
        perm = t.randperm(num_edges)
        num_mask_edges = int(mask_rate * num_edges)

        mask_edges = perm[: num_mask_edges]
        keep_edges = perm[num_mask_edges:]
        
        out_x = x.clone()
        out_x.edge_attr[mask_edges] = 0.0

        return out_x, (mask_edges, keep_edges)

    def forward(self, x):
        mol_g,conf_g = x
        mol_batches = mol_g.batch
        conf_batches = conf_g.batch

        mol_i, (mol_i_mask_nodes, mol_i_keep_nodes) = self.encoding_mask_noise(mol_g.clone(),self._mask_rate)
        mol_j, (mol_j_mask_nodes, mol_j_keep_nodes) = self.encoding_mask_noise(mol_g.clone(),self._mask_rate)

        conf_i, (conf_i_mask_nodes, conf_i_keep_nodes) = self.encoding_mask_noise(conf_g.clone(),self._mask_rate)
        conf_j, (conf_j_mask_nodes, conf_j_keep_nodes) = self.encoding_mask_noise(conf_g.clone(),self._mask_rate)
        conf_i, (conf_i_mask_edges, conf_i_keep_edges) = self.encoding_edge_noise(conf_i,self._mask_rate)
        conf_j, (conf_j_mask_edges, conf_j_keep_edges) = self.encoding_edge_noise(conf_j,self._mask_rate)

        mol_ris,mol_ri_edge_attr = self.mol_encoder(mol_i.x, mol_i.edge_index, mol_i.edge_attr, mol_i.batch)
        mol_rjs,mol_rj_edge_attr = self.mol_encoder(mol_j.x, mol_j.edge_index, mol_j.edge_attr, mol_j.batch)

        conf_ris,conf_ri_edge_attr = self.conf_encoder(conf_i.x, conf_i.edge_index, conf_i.edge_attr, conf_i.batch)
        conf_rjs,conf_rj_edge_attr = self.conf_encoder(conf_j.x, conf_j.edge_index, conf_j.edge_attr, conf_j.batch)

        mol_ris = global_mean_pool(mol_ris,mol_batches)
        mol_rjs = global_mean_pool(mol_rjs,mol_batches)

        conf_ris = global_mean_pool(conf_ris,conf_batches)
        conf_rjs = global_mean_pool(conf_rjs,conf_batches)

        ris = t.cat([mol_ris,conf_ris],dim=-1)
        rjs = t.cat([mol_rjs,conf_rjs],dim=-1)

        ris = self.feat_lin(ris)
        rjs = self.feat_lin(rjs)

        ri_out = self.out_lin(ris)
        rj_out = self.out_lin(rjs)

        return ri_out,rj_out
    
class PreModel_Container(LightningModule):
    def __init__(self,
                in_dim: int,
                num_hidden: int,
                num_layers: int,
                batch_size: int=128,
                in_edge_channel: int=11,
                mid_edge_channel: int=128,
                n_bins: int=6,
                mask_rate: float = 0.6,
                drop_edge_rate: float = 0.6,
                lr: float = 0.001,
                verbose: bool = True,
                my_logging: bool = False,
                scheduler_ReduceLROnPlateau_tracking: str = 'ntxent'
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.verbose = verbose
        self.my_logging = my_logging
        self.scheduler_ReduceLROnPlateau_tracking = scheduler_ReduceLROnPlateau_tracking
        self.lr = lr

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.in_dim = in_dim + 2
        self.enc_mid_channel = num_hidden
        self.in_edge_dim = in_edge_channel + 2
        self.enc_mid_edge_channel = mid_edge_channel
        self.n_bins = n_bins
        self.mask_rate = mask_rate
        self.drop_edge_rate = drop_edge_rate
        
        self.loss_func = NTXentLoss(0,self.batch_size,0.1,True)

        self.model = CLRModel(self.in_dim,self.enc_mid_channel,self.num_layers,
                              self.enc_mid_channel,self.in_edge_dim,self.enc_mid_edge_channel,
                              self.n_bins,self.mask_rate,self.drop_edge_rate)
        
        if self.verbose: print(self.model)
        self.epoch_metrics = Struct(train=[],valid=[],test=[])
        self.metric_dict = {}

    def forward(self, batch):
        return self.model(batch)
        
    def training_step(self,batch,batch_idx):
        zis, zjs = self(batch)
        loss = self.loss_func(zis,zjs)
        
        self.log('trn_loss',loss, prog_bar=False, on_step=False, on_epoch=True)
        lr = iter(self.my_optimizers.param_groups).__next__()['lr']
        self.log('lr', np.round(lr,6), prog_bar=True, on_step=True, on_epoch=False)

        return_dict = {'loss':loss}
        return return_dict
    
    def validation_step(self,batch,batch_idx):
        zis, zjs = self(batch)
        loss = self.loss_func(zis,zjs)

        self.log('val_loss',loss,prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        
        return_dict = {'loss':t2np(loss)}
        return return_dict
    
    def test_step(self,batch,batch_idx):
        zis, zjs = self(batch)
        loss = self.loss_func(zis,zjs)

        self.log('tst_loss',loss,prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        
        return_dict = {'loss':t2np(loss)}
        return return_dict
    
    def training_epoch_end(self, outputs):
        losses = np.asarray([t2np(x['loss']) for x in outputs])
        metric_dict = dict()
        metric_dict['prefix'] = 'epoch_trn'
        metric_dict['epoch'] = self.current_epoch
        metric_dict['ntxent'] = losses.mean()

        if self.my_logging:
            self.logger.log_metrics(keep_scalar_func(metric_dict,prefix='epoch_trn'))
        
        try: self.epoch_metrics.train.pop(-1)
        except: pass
        self.epoch_metrics.train.append(metric_dict)
    
    def validation_epoch_end(self, outputs):
        losses = np.asarray([x['loss'] for x in outputs])
        metric_dict = dict()
        metric_dict['prefix'] = 'epoch_val'
        metric_dict['epoch'] = self.current_epoch
        metric_dict['ntxent'] = losses.mean()

        self.log('val_epoch_NTXent', metric_dict['ntxent'], prog_bar=False, on_step=False, on_epoch=True)

        try: self.epoch_metrics.valid.pop(-1)
        except: pass
        self.epoch_metrics.valid.append(metric_dict)
        if self.my_logging:
            self.logger.log_metrics(keep_scalar_func(metric_dict,prefix='epoch_val'))
        
        if (len(self.epoch_metrics.train)>0) :
            self.print_metrics_on_epoch_end(self.epoch_metrics.train[-1])
        self.print_metrics_on_epoch_end(self.epoch_metrics.valid[-1])

        self.my_schedulers.step(metric_dict[self.scheduler_ReduceLROnPlateau_tracking])

    def test_epoch_end(self, outputs):
        losses = np.asarray([x['loss'] for x in outputs])
        metric_dict = dict()
        metric_dict['prefix'] = 'epoch_tst'
        metric_dict['epoch'] = self.current_epoch
        metric_dict['ntxent'] = losses.mean()

        if self.my_logging:
            self.logger.log_metrics(keep_scalar_func(metric_dict,prefix='epoch_tst'))
        
        try: self.epoch_metrics.test.pop(-1)
        except: pass
        self.epoch_metrics.test.append(metric_dict)
        
        self.print_metrics_on_epoch_end(self.epoch_metrics.test[-1])


    def print_metrics_on_epoch_end(self,metric_dict):
        try:
            lr = iter(self.my_optimizers.param_groups).__next__()['lr']
        except:
            lr = 0
        
        print('\n%s:Ep%04d|| Loss: %.05f\n'%(metric_dict['prefix'],metric_dict['epoch'],metric_dict['ntxent']))

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def configure_optimizers(self):
        self.my_optimizers = t.optim.Adam(self.parameters(), lr=self.lr)
        
        mode = 'min'

        self.my_schedulers = t.optim.lr_scheduler.ReduceLROnPlateau(self.my_optimizers,
                                                                    mode= mode,
                                                                    factor= 0.1, patience=8, verbose=True,
                                                                    threshold=0.0001, threshold_mode='abs'
                                                                    )
        return self.my_optimizers