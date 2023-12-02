import numpy as np 
import warnings
warnings.filterwarnings('ignore')
import torch
import torch as t 
import torch.nn.functional as F
from torch import nn 
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from utils import *
from DeepGCN import *
from itertools import chain

class My_MSE_Loss(nn.Module):
    def __init__(self):
        super(My_MSE_Loss,self).__init__()
        self.node_loss = F.mse_loss
        self.edge_loss = F.mse_loss

    def forward(self,output,target):
        y_out_node,y_out_edge = output
        y_node,y_edge = target
        node_loss = self.node_loss(y_out_node,y_node)
        edge_loss = self.edge_loss(y_out_edge,y_edge)
        return node_loss + edge_loss

def mask_edge(graph, mask_prob):
    E = graph.num_edges

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx

def dropout_edge(edge_index, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True):
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0)) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask

class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            in_edge_channel: int=11,
            mid_edge_channel: int=128,
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            concat_hidden: bool = False,
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        # build encoder
        self.encoder = DeeperGCN(in_dim, enc_num_hidden, num_layers,1,
                                 dropout_ratio=0.1,embedding_layer=None,
                                 graph_conv=SAGEConvV2,
                                 in_edge_channel=in_edge_channel,
                                 mid_edge_channel=mid_edge_channel,aggr='softmax')

        # build decoder for attribute prediction
        self.decoder = DeeperGCN(enc_num_hidden, in_dim, 1, 1,
                                 dropout_ratio=0.1, embedding_layer=None,
                                 graph_conv=SAGEConvV2,
                                 in_edge_channel=mid_edge_channel,
                                 mid_edge_channel=in_edge_channel,aggr='softmax',decode=True)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.enc_edge_token = nn.Parameter(torch.zeros(1,in_edge_channel))
        if concat_hidden:
            self.encoder2decoder_nodes = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
            self.encoder2decoder_edges = nn.Linear(mid_edge_channel * num_layers, mid_edge_channel, bias= False)
        else:
            self.encoder2decoder_nodes = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
            self.encoder2decoder_edges = nn.Linear(mid_edge_channel, mid_edge_channel, bias=False)
        
        # self.final_edge_decoder = nn.Linear(in_dim,in_edge_channel,bias=False)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size
    
    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.num_nodes
        perm = torch.randperm(num_nodes)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes)[:num_noise_nodes]

            out_x = x.clone()
            out_x.x[token_nodes] = 0.0
            out_x.x[noise_nodes] = x.x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x.x[mask_nodes] = 0.0

        out_x.x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def encoding_edge_noise(self, x, mask_rate = 0.3):
        num_edges = x.num_edges
        perm = torch.randperm(num_edges)
        num_mask_edges = int(mask_rate * num_edges)

        mask_edges = perm[: num_mask_edges]
        keep_edges = perm[num_mask_edges:]

        if self._replace_rate > 0:
            num_noise_edges = int(self._replace_rate * num_mask_edges)
            perm_mask = torch.randperm(num_mask_edges)
            token_edges = mask_edges[perm_mask[: int(self._mask_token_rate * num_mask_edges)]]
            noise_edges = mask_edges[perm_mask[-int(self._replace_rate * num_mask_edges):]]
            noise_to_be_chosen = torch.randperm(num_edges)[:num_noise_edges]

            out_x = x.clone()
            out_x.edge_attr[token_edges] = 0.0
            out_x.edge_attr[noise_edges] = x.edge_attr[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_edges = mask_edges
            out_x.edge_attr[mask_edges] = 0.0

        out_x.edge_attr[token_edges] += self.enc_edge_token

        return out_x, (mask_edges, keep_edges)

    def forward(self, x):
        # ---- attribute reconstruction ----
        node_recon,edge_recon = self.mask_attr_prediction(x)
        # edge_recon = self.final_edge_decoder(edge_recon)
        return node_recon,edge_recon
    
    def mask_attr_prediction(self, x):
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self._mask_rate)

        use_x, (mask_edges, keep_edges) = self.encoding_edge_noise(use_x, self._mask_rate)

        use_edge_index = x.edge_index
        use_edge_attr = x.edge_attr

        node_rep,edge_attr = self.encoder(use_x.x, use_edge_index, use_edge_attr, use_x.batch)

        # ---- attribute reconstruction ----
        x_rep = self.encoder2decoder_nodes(node_rep)
        edge_rep = self.encoder2decoder_edges(edge_attr)


        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            x_rep[mask_nodes] = 0
            edge_rep[mask_edges] = 0


        if self._decoder_type in ("mlp", "linear") :
            x_recon = self.decoder(x_rep)
        else:
            x_recon,edge_recon = self.decoder(x_rep, use_edge_index,edge_rep)
        x_init = x.x[mask_nodes]
        x_rec = x_recon[mask_nodes]

        edge_init = x.edge_attr[mask_edges]
        edge_rec = edge_recon[mask_edges]
        return x_recon,edge_recon

    def embed(self, x):
        rep = self.encoder(x.x, x.edge_index, x.edge_attr, x.batch)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
    

class PreModel_Container(LightningModule):
    def __init__(self,
                in_dim: int,
                num_hidden: int,
                num_layers: int,
                nhead: int,
                nhead_out: int,
                in_edge_channel: int=11,
                mid_edge_channel: int=128,
                mask_rate: float = 0.3,
                encoder_type: str = "gat",
                decoder_type: str = "gat",
                loss_fn: str = "mse",
                drop_edge_rate: float = 0.0,
                replace_rate: float = 0.1,
                alpha_l: float = 2,
                lr: float = 0.001, 
                concat_hidden: bool = False,
                verbose: bool = True,
                my_logging: bool = False,
                scheduler_ReduceLROnPlateau_tracking: str = 'mse',

                ):
        super().__init__()
        self.save_hyperparameters()

        self.verbose = verbose
        self.my_logging = my_logging
        self.scheduler_ReduceLROnPlateau_tracking = scheduler_ReduceLROnPlateau_tracking
        self.lr = lr

        self.num_layers = num_layers
        self.in_dim = in_dim + 2
        self.enc_mid_channel = num_hidden
        self.nhead = nhead
        self.nhead_out = nhead_out
        self.in_edge_dim = in_edge_channel
        self.enc_mid_edge_channel = mid_edge_channel
        self.mask_rate = mask_rate
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        
        self.drop_edge_rate = drop_edge_rate
        self.replace_rate = replace_rate
        self.alpha_l = alpha_l
        self.concat_hidden = concat_hidden 

        if loss_fn == 'mse':
            self.loss_func = My_MSE_Loss()

        self.model = PreModel(self.in_dim,self.enc_mid_channel,self.num_layers,self.nhead,
                              self.nhead_out,self.in_edge_dim,self.enc_mid_edge_channel,self.mask_rate,
                              self.encoder_type,self.decoder_type,self.drop_edge_rate,self.replace_rate
                              ,self.concat_hidden)
        
        if self.verbose: print(self.model,)
        self.epoch_metrics = Struct(train=[],valid=[],test=[])  
        self.metric_dict = {}

    def forward(self,batch):
        return self.model(batch)
    
    def training_step(self,batch,batch_idx):
        y = batch
        y_out = self(batch)
        y_node = y.x
        y_edge = y.edge_attr
        y_out_node, y_out_edge = y_out

        loss = self.loss_func(y_out,(y_node,y_edge))

        self.log('train_loss', loss, prog_bar=False, on_step=False,
                 on_epoch=True)
        lr = iter(self.my_optimizers.param_groups).__next__()['lr']
        self.log('lr', np.round(lr,6), prog_bar=True, on_step=True,
                 on_epoch=False)
        
        return_dict = {'loss':loss,'y_node':t2np(y_node),'y_edge':t2np(y_edge),
                       'y_out_node':t2np(y_out_node),'y_out_edge':t2np(y_out_edge)}
        return return_dict
    
    def validation_step(self, batch, batch_idx):
        y = batch
        y_out = self(batch)
        y_node = y.x
        y_edge = y.edge_attr
        y_out_node, y_out_edge = y_out

        loss = self.loss_func(y_out,(y_node,y_edge))

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return_dict = {'y_node':t2np(y_node),'y_edge':t2np(y_edge),
                       'y_out_node':t2np(y_out_node),'y_out_edge':t2np(y_out_edge)}
        
        return return_dict
    
    def test_step(self, batch, batch_idx):
        y = batch
        y_out = self(batch)
        y_node = y.x
        y_edge = y.edge_attr
        y_out_node, y_out_edge = y_out

        loss = self.loss_func(y_out,(y_node,y_edge))

        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return_dict = {'y_node':t2np(y_node),'y_edge':t2np(y_edge),
                       'y_out_node':t2np(y_out_node),'y_out_edge':t2np(y_out_edge)}
        
        return return_dict

    def training_epoch_end(self, outputs):
        y_out_nodes = np.concatenate([x['y_out_node'] for x in outputs])
        y_out_edges = np.concatenate([x['y_out_edge'] for x in outputs])
        y_nodes = np.concatenate([x['y_node'] for x in outputs])
        y_edges = np.concatenate([x['y_edge'] for x in outputs])
        y_out = (y_out_nodes,y_out_edges)
        y = (y_nodes,y_edges)
        metric_dict = self.cal_metrics_on_epoch_end(y,y_out,'trn')

        if self.my_logging:
            self.logger.log_metrics(keep_scalar_func(metric_dict,prefix='epoch_trn'))
        
        try: self.epoch_metrics.train.pop(-1)
        except: pass
        self.epoch_metrics.train.append(metric_dict)

    def validation_epoch_end(self, outputs):
        y_out_nodes = np.concatenate([x['y_out_node'] for x in outputs])
        y_out_edges = np.concatenate([x['y_out_edge'] for x in outputs])
        y_nodes = np.concatenate([x['y_node'] for x in outputs])
        y_edges = np.concatenate([x['y_edge'] for x in outputs])
        y_out = (y_out_nodes,y_out_edges)
        y = (y_nodes,y_edges)
        metric_dict = self.cal_metrics_on_epoch_end(y,y_out,'val')
        self.log('val_epoch_MSE', metric_dict['mse'], prog_bar=False, on_step=False, on_epoch=True)

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
        y_out_nodes = np.concatenate([x['y_out_node'] for x in outputs])
        y_out_edges = np.concatenate([x['y_out_edge'] for x in outputs])
        y_nodes = np.concatenate([x['y_node'] for x in outputs])
        y_edges = np.concatenate([x['y_edge'] for x in outputs])
        y_out = (y_out_nodes,y_out_edges)
        y = (y_nodes,y_edges)
        metric_dict = self.cal_metrics_on_epoch_end(y,y_out,'tst')

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
        
        print('\n%s:Ep%04d|| Loss: %.05f\n'%(metric_dict['prefix'],metric_dict['epoch'],metric_dict['mse']))

    def cal_metrics_on_epoch_end(self,y_true,y_pred,prefix,current_epoch=None):
        y_node,y_edge = y_true
        y_out_node,y_out_edge = y_pred

        metric_dict = dict()
        metric_dict['prefix'] = prefix
        metric_dict['epoch'] = self.current_epoch if current_epoch is None else current_epoch
        node_loss = ((y_node - y_out_node)**2).mean()
        edge_loss = ((y_edge - y_out_edge)**2).mean()
        metric_dict['mse'] = node_loss+edge_loss
        return metric_dict

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def configure_optimizers(self):
        self.my_optimizers =  torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.scheduler_ReduceLROnPlateau_tracking in ['mse',]:
            mode = 'min'
        elif self.scheduler_ReduceLROnPlateau_tracking in ['F1','auPRC']:
            mode = 'max'
        else: raise 
        self.my_schedulers = t.optim.lr_scheduler.ReduceLROnPlateau(self.my_optimizers,
                                mode= mode,#'min',
                                factor=0.1, patience=8, verbose=True, 
                                threshold=0.0001, threshold_mode='abs', )
        return self.my_optimizers 
