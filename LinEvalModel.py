import numpy as np 
import pandas as pd 
import json
from typing import Optional, List, NamedTuple
import torch
import torch as t 
from torch import Tensor
import torch.nn.functional as F
from torch import nn 
# from torch.nn import ModuleList, BatchNorm1d
from torch.autograd import Variable
from typing import Union, Tuple,Optional
from torch_geometric.typing import OptPairTensor, Adj, Size,OptTensor
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter, scatter_softmax
from torch.nn import Parameter
from torch_geometric.nn import global_mean_pool
# import  torch_geometric.nn as gnn
from dataset import (smile_dict,trans_seqs,seq_dict)
from pytorch_lightning import LightningModule
from utils import * 
from metrics import  (evaluate_binary,evaluate_multiclass,evaluate_multilabel,evaluate_regression)
from DeepGCN import *

from torch.nn.init import uniform_,zeros_,xavier_normal_
@torch.no_grad()
def init_linear(m):
    if type(m) == nn.Linear:
        xavier_normal_(m.weight,gain=1)
        if m.bias is not None :
            zeros_(m.bias)
        else:
            pass 

class MyCrossEntropyLoss(t.nn.Module):
    def __init__(self,weight = None, size_average=None, ignore_index = -100, reduce=None, reduction = 'mean'):
        super(MyCrossEntropyLoss, self).__init__()
        self.loss = t.nn.NLLLoss(weight,size_average,ignore_index,reduce,reduction)
    
    def forward(self,output,target):
        '''
        output : [N,C], predicted probability of sample belonging to each classes,i.e., values after activation functions (softmax)
                target: [N,],
        
        '''
        output = t.log(output+ 1e-10)
        return self.loss(output,target)

# Modified DeepDrug Model
class DeepDrug(nn.Module):
    def __init__(self,in_dim,enc_num_hidden,num_out_dim=1,out_activation_func = 'softmax',
                 dropout_ratio=0.1,num_layers = 22,
                 entry1_seq_len=200,in_edge_channel=11,mid_edge_channel=128, linEval=False,use_seq=True,use_conf=True, g_conv=SAGEConvV2):
        super(DeepDrug,self).__init__()
        self.use_conf = use_conf
        self.use_seq = use_seq
        self.out_activation_func = out_activation_func
        self.dropout_ratio = dropout_ratio

        if use_conf:
            self.gconv1 = DeeperGCN(in_dim,enc_num_hidden,num_layers,1,
                                    dropout_ratio=0.1,embedding_layer=None,
                                    graph_conv=g_conv,
                                    in_edge_channel=None,
                                    mid_edge_channel=mid_edge_channel,aggr='softmax')
            
            self.gconv1_conf = DeeperGCN(in_edge_channel,mid_edge_channel,num_layers,1,
                                      dropout_ratio=0.1,embedding_layer=None,
                                      graph_conv=g_conv,
                                      in_edge_channel=6,
                                      mid_edge_channel=mid_edge_channel, aggr='softmax')
            dim_gconv1_out = enc_num_hidden + mid_edge_channel

            self.gconv2_conf = self.gconv1_conf
        else:
            self.gconv1 = DeeperGCN(in_dim, enc_num_hidden, num_layers,1,
                                    dropout_ratio=0.1,embedding_layer=None,
                                    graph_conv=g_conv,
                                    in_edge_channel= in_edge_channel,
                                    mid_edge_channel=mid_edge_channel,aggr='softmax')
            dim_gconv1_out = enc_num_hidden
        
        if use_seq:
            self.gconv1_seq = CNN(len(smile_dict),enc_num_hidden,seq_len=entry1_seq_len,decoder=False,pretraining=False)
            dim_gconv1_seq_out = enc_num_hidden
            dim_gconv1_out += dim_gconv1_seq_out
            self.gconv2_seq = self.gconv1_seq

        self.gconv2 = self.gconv1
        dim_gconv2_out = dim_gconv1_out
        

        if linEval:
            for param in self.gconv1.parameters():
                param.requires_grad = False
            for param in self.gconv2.parameters():
                param.requires_grad = False

        channels = [dim_gconv1_out + dim_gconv2_out,] + [128, 32]
        latent_dim = channels[-1]
        nn_list = []

        for idx, num in enumerate(channels[:-1]):
            nn_list.append(nn.Linear(channels[idx],channels[idx+1]))
            nn_list.append(nn.BatchNorm1d(channels[idx+1]))
            if self.dropout_ratio > 0:
                nn_list.append(nn.Dropout(self.dropout_ratio))
            nn_list.append(nn.ReLU())
        self.global_fc_nn = nn.Sequential(*nn_list)
        self.fc2 = nn.Linear(latent_dim,num_out_dim)

        self.reset_parameters()
    
    def reset_parameters(self,):
        self.apply(init_linear)

    def forward(self,entry1_data,entry2_data,get_latent_variable=False):
        if self.use_seq:
            entry1_data,entry1_seq_data = entry1_data
            entry2_data,entry2_seq_data = entry2_data

        if self.use_conf:
            entry1_data,entry1_conf = entry1_data
            entry2_data,entry2_conf = entry2_data

        entry1_x,entry1_edge_index,entry1_edge_attr,entry1_batch = entry1_data.x,entry1_data.edge_index,entry1_data.edge_attr,entry1_data.batch
        entry1_out_node, entry1_out_edge = self.gconv1(entry1_x,entry1_edge_index,entry1_edge_attr,entry1_batch)
        entry1_mean = global_mean_pool(entry1_out_node,entry1_batch)
        
        if self.use_seq:
            entry1_seq_mean = self.gconv1_seq(entry1_seq_data)
            entry1_mean = t.cat([entry1_mean,entry1_seq_mean],dim=-1)

        if self.use_conf:
            entry1_conf_x,entry1_conf_edge_index,entry1_conf_edge_attr,entry1_conf_batch = entry1_conf.x,entry1_conf.edge_index,entry1_conf.edge_attr,entry1_conf.batch
            entry1_conf_out_node, _ = self.gconv1_conf(entry1_conf_x,entry1_conf_edge_index,entry1_conf_edge_attr,entry1_conf_batch)
            entry1_conf_mean = global_mean_pool(entry1_conf_out_node,entry1_conf_batch)
            entry1_mean = t.cat([entry1_mean,entry1_conf_mean],dim=-1)

        entry2_x,entry2_edge_index,entry2_edge_attr,entry2_batch = entry2_data.x,entry2_data.edge_index,entry2_data.edge_attr,entry2_data.batch
        entry2_out_node, entry2_out_edge = self.gconv2(entry2_x,entry2_edge_index,entry2_edge_attr,entry2_batch)
        entry2_mean = global_mean_pool(entry2_out_node,entry2_batch)
        
        if self.use_seq:
            entry2_seq_mean = self.gconv2_seq(entry2_seq_data)
            entry2_mean = t.cat([entry2_mean,entry2_seq_mean],dim=-1)

        if self.use_conf:
            entry2_conf_x,entry2_conf_edge_index,entry2_conf_edge_attr,entry2_conf_batch = entry2_conf.x,entry2_conf.edge_index,entry2_conf.edge_attr,entry2_conf.batch
            entry2_conf_out_node, _ = self.gconv2_conf(entry2_conf_x,entry2_conf_edge_index,entry2_conf_edge_attr,entry2_conf_batch)
            entry2_conf_mean = global_mean_pool(entry2_conf_out_node,entry2_conf_batch)
            entry2_mean = t.cat([entry2_mean,entry2_conf_mean],dim=-1)

        cat_features = t.cat([entry1_mean,entry2_mean],dim=-1)
        x = self.global_fc_nn(cat_features)
        if get_latent_variable:
            return x
        else:
            x = self.fc2(x)
            if self.out_activation_func == 'softmax':
                return F.softmax(x,dim=-1)
            elif self.out_activation_func == 'sigmoid':
                return t.sigmoid(x)
            elif self.out_activation_func is None:
                return entry1_mean, entry2_mean
    

class DeepDrug_Container(LightningModule):
    def __init__(self,num_out_dim=1, task_type = 'multi_classification',
                 lr = 0.001, category = None, verbose=True, my_logging=False, 
                 scheduler_ReduceLROnPlateau_tracking='mse',
                 model_type = 'deepdrug', linEval=False, n_layers=22,use_seq=True, use_conf=False, g_conv=SAGEConvV2):
        super().__init__()

        self.save_hyperparameters()
        assert task_type in ['regression','binary_classification','binary',
                            'multi_classification','multilabel_classification','multiclass','multilabel',
                             ]

        self.verbose = verbose
        self.my_logging = my_logging
        self.scheduler_ReduceLROnPlateau_tracking = scheduler_ReduceLROnPlateau_tracking
        self.lr = lr 
        self.task_type = task_type
        self.num_classes = num_out_dim
        self.model_type = model_type
        self.n_layers = n_layers

        self.category = category
        if self.category == 'DDI':
            self.entry2_type= 'drug' 
            self.entry2_num_graph_layer= self.n_layers
            self.entry2_seq_len= 200
            self.entry2_in_channel=  119 + 2 
            self.entry2_in_edge_channel= 12 + 2 if use_conf else 11
            self.siamese_feature_module=True  

        if self.task_type in ['multi_classification','multiclass']:
            out_activation_func = 'softmax'
            self.loss_func =MyCrossEntropyLoss()
        elif self.task_type in ['binary_classification','binary','multilabel_classification','multilabel']:
            out_activation_func = 'sigmoid'
            self.loss_func = F.binary_cross_entropy
        else:
            out_activation_func = None
        
        assert self.model_type in ['deepdrug','geometric']
        if self.model_type == 'deepdrug':
            self.model = DeepDrug(num_out_dim=num_out_dim,out_activation_func = out_activation_func,
                                in_dim=self.entry2_in_channel,enc_num_hidden=128,num_layers=self.entry2_num_graph_layer,
                                entry1_seq_len=self.entry2_seq_len,in_edge_channel=self.entry2_in_edge_channel,
                                mid_edge_channel=128, linEval = linEval,dropout_ratio=0.2,use_seq=use_seq,use_conf=use_conf,
                                g_conv = g_conv
                                )
        else:
            raise

        if self.verbose: print(self.model,)
        self.epoch_metrics = Struct(train=[],valid=[],test=[])  
        self.metric_dict = {}
    
    def forward(self,batch) :
        (entry1_data,entry2_data),y  = batch
        return self.model(entry1_data,entry2_data)
    
    def training_step(self, batch, batch_idx):
        (entry1_data,entry2_data),y  = batch
        y_out = self(batch)
        if self.task_type in ['multi_classification','multiclass',]:
            loss = self.loss_func(y_out, y.reshape(-1))
        elif self.task_type in ['binary','binary_classification']:
            loss = self.loss_func(y_out, y.float())
        elif self.task_type in ['multilabel_classification','multilabel',]:
            loss = self.loss_func(y_out, y.float())
        else:
            loss = self.loss_func(y_out, y)

        self.log('train_loss', loss, prog_bar=False, on_step=False,
                 on_epoch=True)
        lr = iter(self.my_optimizers.param_groups).__next__()['lr']
        self.log('lr', np.round(lr,6), prog_bar=True, on_step=True,
                 on_epoch=False)


        return_dict = {'loss':loss,'y_out':t2np(y_out),'y':t2np(y)}

        return return_dict 

    def validation_step(self, batch, batch_idx):
        (entry1_data,entry2_data),y  = batch
        y_out = self(batch)
        if self.task_type in ['multi_classification','multiclass',]:
            loss = self.loss_func(y_out, y.reshape(-1))
        elif self.task_type in ['binary','binary_classification']:
            loss = self.loss_func(y_out, y.float())
        elif self.task_type in ['multilabel_classification','multilabel',]:
            loss = self.loss_func(y_out, y.float())
        else:
            loss = self.loss_func(y_out, y)

        self.log('val_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True, sync_dist=True)

        # return loss 
        return_dict = {'y_out':t2np(y_out),'y':t2np(y)}
        return return_dict  

    def test_step(self, batch, batch_idx):
        (entry1_data,entry2_data),y  = batch
        y_out = self(batch)
        if self.task_type in ['multi_classification','multiclass',]:
            loss = self.loss_func(y_out, y.reshape(-1))
        elif self.task_type in ['binary','binary_classification']:
            loss = self.loss_func(y_out, y.float())
        elif self.task_type in ['multilabel_classification','multilabel',]:
            loss = self.loss_func(y_out, y.float())
        else:
            loss = self.loss_func(y_out, y)
        self.log('test_loss', loss, prog_bar=False, on_step=False,
                 on_epoch=True, sync_dist=True)
        return_dict = {'y_out':t2np(y_out),'y':t2np(y)} 
        return return_dict 

    def training_epoch_end(self,outputs):
        y_out  = np.concatenate([x['y_out'] for x in outputs])
        y  = np.concatenate([x['y'] for x in outputs])
            
        metric_dict = self.cal_metrics_on_epoch_end(y,y_out,'trn',)
        if self.my_logging:
            self.logger.log_metrics(keep_scalar_func(metric_dict,prefix='epoch_trn'))
        try: self.epoch_metrics.train.pop(-1)
        except: pass
        self.epoch_metrics.train.append(metric_dict)

    def validation_epoch_end(self,outputs):
        y_out  = np.concatenate([x['y_out'] for x in outputs])
        y  = np.concatenate([x['y'] for x in outputs])
        metric_dict = self.cal_metrics_on_epoch_end(y,y_out,'val')
        if self.task_type in ['binary','multiclass','multilabel']:
            self.log('val_epoch_F1', metric_dict['F1'], prog_bar=False, on_step=False,on_epoch=True)
            self.log('val_epoch_auPRC', metric_dict['auPRC'], prog_bar=False, on_step=False,on_epoch=True)
        elif self.task_type in ['regression',]:
            self.log('val_epoch_MSE', metric_dict['mse'], prog_bar=False, on_step=False,on_epoch=True)

        try: self.epoch_metrics.valid.pop(-1)
        except: pass
        self.epoch_metrics.valid.append(metric_dict)
        if self.my_logging:
            self.logger.log_metrics(keep_scalar_func(metric_dict,prefix='epoch_val'))

        if (len(self.epoch_metrics.train)>0) :
            self.print_metrics_on_epoch_end(self.epoch_metrics.train[-1])
        self.print_metrics_on_epoch_end(self.epoch_metrics.valid[-1])
        self.my_schedulers.step(metric_dict[self.scheduler_ReduceLROnPlateau_tracking])  #'mse','F1'

    def test_epoch_end(self,outputs):
        y_out  = np.concatenate([x['y_out'] for x in outputs])
        y  = np.concatenate([x['y'] for x in outputs])
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

        if 'F1' in metric_dict.keys():
            print('\n%s:Ep%04d|| F1: %.03f,auROC %.03f,auPRC: %.03f'%(metric_dict['prefix'],metric_dict['epoch'],metric_dict['F1'],
                                        metric_dict['auROC'],
                                        metric_dict['auPRC'],))
        elif 'mse' in metric_dict.keys():
            print('\n%s:Ep%04d|| MSE: %.03f,lr: %.07f,r2: %.03f,pear-r: %.03f,con_index: %.03f,expVar: %.03f,cidx: %.03f,rm2: %.03f'%(
                                        metric_dict['prefix'],metric_dict['epoch'],metric_dict['mse'],lr,
                                        metric_dict['r2'],
                                        metric_dict['pearsonr'],
                                        metric_dict['concordance_index'],
                                        metric_dict['explained_variance'],
                                        metric_dict['cindex'],
                                        metric_dict['rm2'],
                                        ))

            if metric_dict['prefix'] == 'tst' :
                print('\n%s:Ep%04d|| %.03f,%.03f,%.03f,%.03f,%.03f,%.03f,%.03f'%(
                                        metric_dict['prefix'],metric_dict['epoch'],metric_dict['mse'],
                                        metric_dict['r2'],
                                        metric_dict['pearsonr'],
                                        metric_dict['concordance_index'],
                                        metric_dict['explained_variance'],
                                        metric_dict['cindex'],
                                        metric_dict['rm2'],
                                        ))


    def cal_metrics_on_epoch_end(self,y_true,y_pred,prefix,current_epoch=None ):

        if self.task_type in ['multi_classification','multiclass',]:
            metric_dict =evaluate_multiclass(y_true,y_pred,to_categorical=True,num_classes=self.num_classes)     
        elif self.task_type in ['binary','binary_classification']:
            metric_dict =evaluate_binary(y_true,y_pred)
        elif self.task_type in ['multilabel_classification','multilabel',]:
            metric_dict = evaluate_multilabel(y_true,y_pred)   
        metric_dict['prefix'] = prefix
        metric_dict['epoch'] = self.current_epoch if current_epoch is None else current_epoch
        return metric_dict


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()


    def configure_optimizers(self):
        self.my_optimizers =  torch.optim.Adam(filter(lambda p : p.requires_grad, self.parameters()), lr=self.lr, weight_decay= 1e-6)

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
