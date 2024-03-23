import numpy as np 
import pandas as pd 
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
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
from torch.utils.checkpoint import checkpoint

def structDict(**argvs):
    return argvs

def calc_cnn_output(seq_len,kernels):
    for k in kernels:
        seq_len -= k
        seq_len += 1
    return seq_len

class CNN(nn.Sequential):
    def __init__(self,in_channel,mid_channel,seq_len,dropout_ratio=0.1,decoder=False,pretraining=False):
        super(CNN, self).__init__()
        self.seq_len= seq_len
        self.decoder = decoder
        self.pretraining = pretraining
        in_channel = in_channel

        encoding = 'drug'
        if decoder:
            config = structDict( 
                            cls_hidden_dims = [512,1024,1024], 
                            cnn_drug_filters = [96,64,len(smile_dict)],
                            cnn_target_filters = [96,64,32],
                            cnn_drug_kernels = [8,6,4],
                            cnn_target_kernels = [12,8,4]
                            )
        else:
            config = structDict( 
                            cls_hidden_dims = [1024,1024,512], 
                            cnn_drug_filters = [32,64,96],
                            cnn_target_filters = [32,64,96],
                            cnn_drug_kernels = [4,6,8],
                            cnn_target_kernels = [4,8,12]
                            )
        if encoding == 'drug':
            in_ch = [in_channel] + config['cnn_drug_filters']
            kernels = config['cnn_drug_kernels']
            layer_size = len(config['cnn_drug_filters'])
            self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
                                                    out_channels = in_ch[i+1], 
                                                    kernel_size = kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            
            if not pretraining:
                n_size_d = self._get_conv_output(( in_channel,seq_len,))
                self.fc1 = nn.Linear(n_size_d, mid_channel)

        if encoding == 'protein':
            in_ch = [in_channel] + config['cnn_target_filters']
            kernels = config['cnn_target_kernels']
            layer_size = len(config['cnn_target_filters'])
            self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
                                                    out_channels = in_ch[i+1], 
                                                    kernel_size = kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()

            if not pretraining:
                n_size_p = self._get_conv_output(( in_channel,seq_len,))
                self.fc1 = nn.Linear(n_size_p, mid_channel)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        if not self.pretraining:
            x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v.double())
        if not self.pretraining:
            v = v.view(v.size(0), -1)
            v = self.fc1(v.float())
            return v
        else:
            return v.float()

class DeepGCNLayerV2(torch.nn.Module):
    def __init__(self, conv=None, norm=None, act=None, block='res+', 
                 dropout=0., ckpt_grad=False,edge_norm = None,):
        super(DeepGCNLayerV2, self).__init__()

        self.conv = conv
        self.norm = norm
        self.act = act
        self.edge_norm  = edge_norm
        self.block = block.lower()
        assert self.block in ['res+', 'res', 'dense', 'plain']
        self.dropout = dropout
        self.ckpt_grad = ckpt_grad


    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, *args, **kwargs):
        """"""
        args = list(args)
        x = args.pop(0)
        org_edge_attr = args[1] #org_edge_attr:[edge_index,edge_attr,....]
        if org_edge_attr is None: org_edge_attr = 0
        if self.block == 'res+':
            if self.norm is not None:
                h = self.norm(x)
            if self.act is not None:
                h = self.act(h)
            if self.edge_norm is not None:
                args[1] = self.edge_norm(args[1])
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.conv is not None and self.ckpt_grad and h.requires_grad:
                h,edge_attr = checkpoint(self.conv, h, *args, **kwargs)
            else:
                h,edge_attr = self.conv(h, *args, **kwargs)

            return x + h,org_edge_attr +edge_attr 

        else:
            if self.conv is not None and self.ckpt_grad and x.requires_grad:
                h,edge_attr = checkpoint(self.conv, x, *args, **kwargs)
            else:
                h,edge_attr = self.conv(x, *args, **kwargs)
            if self.norm is not None:
                h = self.norm(h)
            if self.act is not None:
                h = self.act(h)
            if self.edge_norm is not None:
                edge_attr = self.edge_norm(edge_attr)

            if self.block == 'res':
                return x + h, org_edge_attr+edge_attr
            elif self.block == 'dense':
                return torch.cat([x, h], dim=-1),org_edge_attr+edge_attr
            elif self.block == 'plain':
                return h,edge_attr

    def __repr__(self):
        return '{}(block={})'.format(self.__class__.__name__, self.block)

class SAGEConvV2(MessagePassing):
    # concat edge_index and node features
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True,
                 in_edge_channels: Union[int, Tuple[int, int],None] = None ,
                 aggr: str = 'mean', t: float = 1.0, learn_t: bool = False,
                 p: float = 1.0, learn_p: bool = False,
                  **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', None)
        super(SAGEConvV2, self).__init__(**kwargs)

        self.aggr = aggr
        self.eps = 1e-7
        assert aggr in ['softmax', 'softmax_sg', 'power','mean','add','sum']
        if self.aggr in ['softmax', 'softmax_sg', 'power',]:
            self.initial_t = t
            if learn_t and aggr == 'softmax':
                self.t = Parameter(torch.Tensor([t]), requires_grad=True)
            else:
                self.t = t
            self.initial_p = p
            if learn_p:
                self.p = Parameter(torch.Tensor([p]), requires_grad=True)
            else:
                self.p = p

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_edge_channels = in_edge_channels 
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        
        if in_edge_channels is not None:
            self.lin_l = nn.Sequential(
                                        nn.Linear(in_channels[0]*2 + in_edge_channels, in_channels[0]*2 , bias=bias),
                                        nn.ReLU(),
                                        nn.Linear(in_channels[0]*2 , out_channels, bias=bias),
                                        )   
        else:
            self.lin_l = nn.Sequential(
                                        nn.Linear(in_channels[0]*2 , in_channels[0]*2 , bias=bias),
                                        nn.ReLU(),
                                        nn.Linear(in_channels[0]*2 , out_channels, bias=bias),
                                        )

    def reset_parameters(self,):
        self.lin_l.apply(init_linear)

        if self.root_weight:
            self.lin_r.reset_parameters()

        if self.aggr in ['softmax', 'softmax_sg', 'power',]:
            if self.t and isinstance(self.t, Tensor):
                self.t.data.fill_(self.initial_t)
            if self.p and isinstance(self.p, Tensor):
                self.p.data.fill_(self.initial_p)


    def message(self, x_i: Tensor,x_j: Tensor,edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            x = t.cat([x_i,x_j],dim=-1)
        else:
            x = t.cat([x_i,x_j,edge_attr],dim=-1)
        x  = self.lin_l(x)
        return x
    
    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        if self.aggr == 'softmax':
            out = scatter_softmax(inputs * self.t, index, dim=self.node_dim)
            return scatter(inputs * out, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='sum'),inputs

        elif self.aggr == 'softmax_sg':
            out = scatter_softmax(inputs * self.t, index, dim=self.node_dim).detach()
            return scatter(inputs * out, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='sum'),inputs

        elif self.aggr == 'power':
            min_value, max_value = 1e-7, 1e1
            torch.clamp_(inputs, min_value, max_value)
            out = scatter(torch.pow(inputs, self.p), index, dim=self.node_dim,
                          dim_size=dim_size, reduce='mean')
            torch.clamp_(out, min_value, max_value)
            return torch.pow(out, 1 / self.p),inputs

        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                               reduce=self.aggr),inputs
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, 
                edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor)
        out,edge_attr  = self.propagate(edge_index, x=x, edge_attr=edge_attr,size=size)
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out,edge_attr
    
class RGINConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 in_edge_channels: Union[int, Tuple[int, int],None] = None,
                 bias:bool = True,
                 eps: float = 1e-7, learn_eps: bool = False,
                  **kwargs):
        super(RGINConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_edge_channels = in_edge_channels 
        self.normalize = normalize

        if learn_eps:
            self.eps = Parameter(torch.Tensor([eps]),requires_grad=True)
        else:
            self.eps = eps

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        
        self.nn = nn.Sequential(
                                 nn.Linear(in_channels[0], in_channels[0], bias=bias),
                                 nn.ReLU(),
                                 nn.Linear(in_channels[0], out_channels, bias=bias)
                                )
        
        if in_edge_channels is not None:
            self.lin_l = nn.Sequential(
                                        nn.Linear(in_channels[0] + in_edge_channels, in_channels[0] , bias=bias),
                                        nn.ReLU(),
                                        nn.Linear(in_channels[0] , out_channels, bias=bias),
                                        )   
        else:
            self.lin_l = nn.Sequential(
                                        nn.Linear(in_channels[0] , in_channels[0] , bias=bias),
                                        nn.ReLU(),
                                        nn.Linear(in_channels[0] , out_channels, bias=bias),
                                        )

    def reset_parameters(self,):
        self.lin_l.apply(init_linear)
        self.nn.apply(init_linear)
    
    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            x = x_j
        else:
            x = t.cat([x_j,edge_attr],dim=-1)
        x  = self.lin_l(x)
        return x
    
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, 
                edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:

        if isinstance(x,Tensor):
            x = (x, x)
        
        out = self.propagate(edge_index=edge_index, x=x, edge_attr = edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1+self.eps) * x_r
        
        return self.nn(out),edge_attr

class DeeperGCN(nn.Module):
    def __init__(self, in_channel,mid_channel, num_layers,num_blocks = 1, dropout_ratio=0.1,embedding_layer=None,graph_conv=SAGEConvV2,
                in_edge_channel=None,mid_edge_channel=None,node_encoding=True,aggr='softmax',decode=False):
        super(DeeperGCN, self).__init__()
        self.update_attr = False
        self.decode = decode
        if graph_conv in [SAGEConvV2,]:
            if aggr == 'softmax':
                graph_para_dict = {'in_edge_channels':mid_edge_channel,
                                    'aggr': 'softmax', "t":  1.0, 'learn_t':  True,}
            elif aggr in ['mean','add','sum']:
                graph_para_dict = {'in_edge_channels':mid_edge_channel,'aggr':aggr}
            use_edge_encoder = True 
            self.update_attr = True 
        elif graph_conv in [RGINConv,]:
            graph_para_dict = {'in_edge_channels': mid_edge_channel,
                               'eps': 1.0, 'learn_eps': True,}
            use_edge_encoder = True
            self.update_attr = True
        else:
            raise 

        if in_edge_channel is None:
            graph_para_dict = {'in_edge_channels':None,'aggr': 'softmax', "t":  1.0, 'learn_t':  True}
        
        self.embedding_layer = embedding_layer

        in_channel = in_channel if self.embedding_layer is None else self.embedding_layer.embedding_dim
        self.dropout_ratio = dropout_ratio
        
        if node_encoding:
            self.node_encoder = nn.Sequential(
                                    nn.Linear(in_channel,mid_channel),
                                    nn.LayerNorm(mid_channel),
                                    )
        else:
            self.node_encoder = None 

        self.gcn_blocks = nn.ModuleList()

        for block in range(num_blocks):
            layers = nn.ModuleList()
            for i in range(1, num_layers + 1):
                conv = graph_conv(mid_channel, mid_channel,**graph_para_dict )
                norm = nn.LayerNorm(mid_channel, elementwise_affine=True)
                edge_norm = nn.LayerNorm(mid_edge_channel, elementwise_affine=True) if in_edge_channel is not None else None
                act = nn.ReLU(inplace=True)

                layer = DeepGCNLayerV2(conv, norm, act, block='res+', dropout=dropout_ratio,
                                     # ckpt_grad=  False,
                                     ckpt_grad=  i % 3,edge_norm =edge_norm,
                                     )
                layers.append(layer)
            self.gcn_blocks.append(layers)

        self.edge_encoder = None 
        self.use_attr = True if in_edge_channel is not None else False 
        if self.use_attr and use_edge_encoder:
            self.edge_encoder = nn.Sequential(
                                    nn.Linear(in_edge_channel,mid_edge_channel),
                                    nn.LayerNorm(mid_edge_channel),
                                    )
        if self.decode:
            self.edge_decoder = nn.Linear(mid_channel,mid_edge_channel)

    def forward(self, x, edge_index,edge_attr=None,batch=None):
        if self.embedding_layer is not None:
            x = self.embedding_layer(x)
        if self.node_encoder is not None:
            x = self.node_encoder(x)
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        for gcn_block in self.gcn_blocks:
            if self.use_attr:  
                x = gcn_block[0].conv(x, edge_index,edge_attr)
                if self.update_attr: x,edge_attr = x
                for layer in gcn_block[1:]:
                    x = layer(x, edge_index,edge_attr)  
                    if self.update_attr: x,edge_attr = x     
            else:
                x = gcn_block[0].conv(x, edge_index)
                if self.update_attr: x,__edge_attr = x  #edge_attr is None
                for layer in gcn_block[1:]:
                    x = layer(x, edge_index, None) 
                    if self.update_attr: x,__edge_attr = x
            x = gcn_block[0].act(gcn_block[0].norm(x))
            # if self.update_attr: x,edge_attr = x

        if self.decode:
            edge_attr = self.edge_decoder(edge_attr)
        return x,edge_attr
