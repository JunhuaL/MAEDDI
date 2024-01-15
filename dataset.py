import numpy as np 
import pandas as pd
import multiprocessing as mp
from copy import deepcopy
from torch_geometric.data import (InMemoryDataset, Data,DataLoader)
from torch_geometric.utils import (dense_to_sparse,to_undirected,add_self_loops,remove_self_loops,degree)
import os 
from copy import deepcopy
from collections import defaultdict,Counter,OrderedDict
from functools import reduce
import pickle as pkl
from itertools import product
import numpy as np
import evaluation
import pandas as pd
from typing import Optional, List, NamedTuple
import matplotlib.pyplot as plt
from time import sleep 
import torch
import torch as t 
from torch import Tensor
import torch.nn.functional as F
from torch import nn 
from torch.nn import ModuleList, BatchNorm1d
from torch.autograd import Variable
from torch.utils.data.distributed import DistributedSampler 
from molGraphConvFeaturizer import MolGraphConvFeaturizer as user_MolGraphConvFeaturizer


from pytorch_lightning import (LightningDataModule,)

def read_df_or_parquet(file,save_parquet=False,**args):
    if os.path.exists(file.replace('.csv','.parquet')):
        print('parquet format file was used which is found in :',file.replace('.csv','.parquet'))
        return  pd.read_parquet(file.replace('.csv','.parquet'))
    elif os.path.exists(file):
        tmp_df = pd.read_csv(file,**args)
        print('print save parquet file to load quickly next time at',file.replace('.csv','.parquet'))
        if save_parquet: tmp_df.to_parquet(file.replace('.csv','.parquet'))
        return  tmp_df
    else: 
        return None 



from torch_sparse import coalesce
def graph_to_undirected(data):

    # num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index,edge_attr,num_nodes =data.edge_index,data.edge_attr,data.num_nodes

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    if edge_attr is not None: edge_attr = torch.cat([edge_attr,edge_attr],dim=0)

    edge_index = torch.stack([row, col], dim=0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes,op="mean")

    data.edge_index = edge_index
    data.edge_attr = edge_attr
    return data

from torch_geometric.utils import degree as geo_degree
def graph_add_degree(data,degree='both'):
    # print('data:',data,type(data))
    assert degree in ['indegree','outdegree','both']

    deg1 = geo_degree(data.edge_index[0],data.num_nodes).reshape(-1,1)
    deg2 = geo_degree(data.edge_index[1],data.num_nodes).reshape(-1,1)
    data.x = t.cat([data.x,deg1,deg2],dim=1)
    return data 


def add_self_loops(edge_index, edge_weight: Optional[torch.Tensor] = None,
                   fill_value: float = 1., num_nodes: Optional[int] = None):
    # N = maybe_num_nodes(edge_index, num_nodes)
    if num_nodes is None: num_nodes = np.max(edge_index)+1 
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.size(0) == edge_index.size(1)
        size = np.array(edge_weight.size()).tolist()
        size[0] = num_nodes
        loop_weight = edge_weight.new_full(size, fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight



class EntryDataset(InMemoryDataset):
    def __init__(self, root_folder,transform=None, pre_transform=None,
                 pre_filter=None,filename = 'data',inmemory=False):
        if not inmemory :
            os.makedirs(os.path.join(root_folder,'processed'),exist_ok=True)
        super(EntryDataset, self).__init__(root_folder, transform, pre_transform, pre_filter)
        self.inmemory = inmemory 
        self.filename = filename
        if os.path.exists(self.processed_paths[0]):
            print('loading processed data...')
            tmp = t.load(self.processed_paths[0])
            if len(tmp) == 3:
                self.data, self.slices,self.entryIDs  = tmp
            elif len(tmp) == 4:
                self.data, self.slices,self.entryIDs,self.unpIDs  = tmp
        else:
            print('file not exists in DeepDrug....')
        
    @property
    def processed_file_names(self,):
        return '{}.pt'.format(self.filename)
        
    def add_self_loops(self,):
        print('add self loop for each grpah....')
        def add_self_loops_func(data):
            data.edge_index,data.edge_attr = add_self_loops(data.edge_index,edge_weight=data.edge_attr,fill_value=0,num_nodes=data.num_nodes)
            return data
        data_list = [add_self_loops_func(data) for data in self]
        
        data,slices = self.collate(data_list)

        self.data = data 
        self.slices = slices
        self.__data_list__ = data_list
        self._data_list = data_list 

    def to_undirected(self,):
        print('convert to undirected graph for each sample...')
        data_list = [graph_to_undirected(data) for data in self]
        data,slices = self.collate(data_list)

        self.data = data 
        self.slices = slices
        self.__data_list__ = data_list
        self._data_list = data_list 

    def add_node_degree(self,):
        print('add degrees as node features for each sample...')
        data_list = [graph_add_degree(data) for data in self]
        data,slices = self.collate(data_list)

        self.data = data 
        self.slices = slices
        self.__data_list__ = data_list
        self._data_list = data_list 

    def drug_process(self,
                    drug_df,flag_add_self_loops=False,
                    default_dim_features=119,default_dim_nodes=50,use_edges=True):
        import deepchem as dc 
        from rdkit import Chem 
        from tqdm import tqdm
        
        assert np.all(np.in1d(['drugID','SMILES'] , drug_df.columns.values))
        self.entryIDs = drug_df.drugID.values
        
        mols_list= list(map(Chem.MolFromSmiles, drug_df.SMILES))  # some SMILES maybe are failed to parse
        featurizer = user_MolGraphConvFeaturizer(use_edges=True,use_chirality=True,use_partial_charge=True)
        deepchem_list = featurizer.featurize(mols_list)
        print("featurize complete")
        data_list = []      
        for convMol in tqdm(deepchem_list):
            #print(convMol)
            if isinstance(convMol,np.ndarray) :
                feat_mat=np.zeros((default_dim_nodes,default_dim_features))
                num_nodes = feat_mat.shape[0]
                edges = np.array([[],[]])
                edges_attr = np.array([])
                
                
            else:
                feat_mat = convMol.node_features#.atom_features
                num_nodes = feat_mat.shape[0]
                edges_attr = convMol.edge_features
                edges = convMol.edge_index
                
                
            if flag_add_self_loops:
                edges = add_self_loops(edges,num_nodes=num_nodes)[0]
                
            
            data_list.append(Data(x=t.from_numpy(feat_mat).float(),
                                edge_index=t.from_numpy(edges).long(),
                                edge_attr=t.from_numpy(edges_attr).float()))
        
        print("reformat complete")
        data,slices = self.collate(data_list)
        t.save((data,slices,self.entryIDs), self.processed_paths[0]) 
        print("written to disk")
        self.data, self.slices,self.entryIDs = t.load(self.processed_paths[0])

    def edge_process(self, 
                    drug_df,flag_add_self_loops=False,
                    default_dim_features=11,default_dim_nodes=40,n_conformers=1):
        from molGraphConvFeaturizer import BondGraphFeaturizer
        from rdkit import Chem
        from tqdm import tqdm
        
        assert np.all(np.in1d(['drugID','SMILES'] , drug_df.columns.values))
        self.entryIDs = drug_df.drugID.values
        mols_list= list(map(Chem.MolFromSmiles, drug_df.SMILES))

        featurizer = BondGraphFeaturizer(use_dihedrals=False,use_edges=True,n_confs=n_conformers)
        bondgraph_list = featurizer.featurize(mols_list)

        data_list = []
        for convBonds in tqdm(bondgraph_list):
            if isinstance(convBonds,np.ndarray) :
                feat_mat = np.zeros((default_dim_nodes,default_dim_features+n_conformers))
                num_nodes = feat_mat.shape[0]
                edges = np.array([[],[]])
                edges_attr = np.array([])
            else:
                feat_mat = convBonds.node_features
                num_nodes = feat_mat.shape[0]
                edges = convBonds.edge_index
                edges_attr = convBonds.edge_features

            data_list.append(Data(x=t.from_numpy(feat_mat).float(),
                                edge_index=t.from_numpy(edges).long(),
                                edge_attr=t.from_numpy(edges_attr).float()))
        
        data,slices = self.collate(data_list)
        t.save((data,slices,self.entryIDs), self.processed_paths[0])
        self.data, self.slices, self.entryIDs = t.load(self.processed_paths[0])

    def protein_process(self,protein_df,pdb_graph_dict,flag_add_self_loops=False, default_dim_features=1,
                        default_dim_nodes=540,key= 'fingerprint',edge_thresh=None,int_featrue=True):
        #import deepchem as dc 
        #from rdkit import Chem 
        from tqdm import tqdm
        
        
        assert np.all(np.in1d(['targetID','sequences'] , protein_df.columns.values))
         
        self.entryIDs = protein_df.targetID.values
        self.entrySeqs = protein_df.sequences.values
        if 'unpID' in protein_df.columns.values:
            self.unpIDs = protein_df.unpID.values
        else:
            self.unpIDs = self.entryIDs
        
        
        data_list = []      
        for unp_id in tqdm(self.unpIDs) :
            pdb_graph = pdb_graph_dict.get(unp_id,None)
            if pdb_graph is None:
                print(unp_id)
                feat_mat=np.zeros((default_dim_nodes,default_dim_features))
                num_nodes = feat_mat.shape[0]
                edges = t.from_numpy(np.array([[],[]])).long()
                #edge_weights = t.from_numpy(np.array([])).float()
                edge_weights = t.from_numpy(np.array([[],[]])).permute(1,0).float()
                
                
                
            else:
                
                adj_feature_flag = 'adj_features' in pdb_graph.keys()
                adj_angle_flag = 'adj_angle_mat' in pdb_graph.keys()
                feat_mat = pdb_graph[key]
                edge_mat = pdb_graph['adj']
                if edge_thresh is not None: 
                    if True : 
                        #  2 features, adj_features + angle
                        assert adj_feature_flag == True 
                        assert adj_angle_flag ==True
                        edge_flag = edge_mat<=edge_thresh
                        
                        
                        adj_features = pdb_graph['adj_features']
                        adj_angle_mat = pdb_graph['adj_angle_mat']
                        # adj_features = np.transpose(adj_features, (1, 2, 0 ))

                        
                        num_nodes = feat_mat.shape[0]
                        dis_mat = adj_features.copy()
                        # sigma = (edge_thresh**2)/5
                        # dis_mat = np.exp(- (dis_mat**2)/sigma)
                        # dis_mat = np.log1p(dis_mat)
                        dis_mat = np.concatenate([dis_mat[...,0][...,np.newaxis],adj_angle_mat[...,np.newaxis]],axis=-1)
    
                        # dis_mat[~edge_flag] = 0 
                        
                        # edges,edge_weights = dense_to_sparse(t.from_numpy(dis_mat).float())
                        index = np.array(np.nonzero(edge_flag,)) # [2,#num_edges]
                        values = np.array([dis_mat[x,y]   for x,y in np.transpose(index) ])
                        edges,edge_weights = t.from_numpy(index).long(),t.from_numpy(values).float()
                        edges,edge_weights = remove_self_loops(edges,edge_weights)
                    
            
                else:
                    edges,edge_weights = dense_to_sparse(t.from_numpy(edge_mat).float())
                
                num_nodes = feat_mat.shape[0]
                
                #edges = to_undirected(edges,num_nodes)
                
            if flag_add_self_loops:
                raise 
                edges = add_self_loops(edges,num_nodes=num_nodes)[0]
                
            if int_featrue:
                x = t.from_numpy(feat_mat).long()
            else:
                x = t.from_numpy(feat_mat).float()
            
            if edge_weights is not None :
                if len(edge_weights.shape) == 1:
                    edge_weights = edge_weights.reshape(-1,1)
                assert edge_weights.size(0) == edges.size(1)
            #print(feat_mat.shape,pdb_graph['adj'].shape,edges.shape,edge_weights.shape)
            data_list.append(Data(x=x,edge_index=edges,edge_attr=edge_weights))
        
        
        data,slices = self.collate(data_list)
        if not self.inmemory: 
            t.save((data,slices,self.entryIDs,self.unpIDs), self.processed_paths[0])  
            self.data, self.slices,self.entryIDs,self.unpIDs = t.load(self.processed_paths[0])
        else:
            self.data, self.slices  = data,slices
    

from torch.utils import data
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ?" #length=26
seq_dict = {v:i for i,v in enumerate(seq_voc)}
# seq_dict_len = len(seq_dict)
# max_seq_len = 1000

smile_dict = {'s': 0,'c': 1,'h': 2,'o': 3,'R': 4,'t': 5,'8': 6,'g': 7,
              'd': 8,'K': 9,'2': 10,'0': 11,'[': 12,'V': 13,'F': 14,'@': 15,
              '.': 16,'%': 17,'D': 18,'5': 19,'#': 20,'C': 21,'l': 22,'H': 23,
              ')': 24,'1': 25,'3': 26,'e': 27,'P': 28,'?': 29,'N': 30,'Z': 31,
              '\\': 32,'i': 33,'B': 34,'r': 35,'E': 36,'9': 37,'k': 38,'f': 39,
              '-': 40,'X': 41,'U': 42,'m': 43,'u': 44,'/': 45,'a': 46,'T': 47,
              '=': 48,'y': 49,'n': 50,']': 51,'L': 52,'b': 53,'7': 54,'(': 55,
              'S': 56,'Y': 57,'+': 58,'A': 59,'M': 60,'4': 61,'I': 62,'W': 63,
              'O': 64,'6': 65,'G': 66}

def trans_seqs(x,seq_dict,max_seq_len=200,upper=True):
    if upper:
        x= x.upper()
    temp = list(x)
    temp = [i if i in seq_dict else '?' for i in temp]
    if len(temp) < max_seq_len:
        temp = temp + ['?'] * (max_seq_len-len(temp))
    else:
        temp = temp [:max_seq_len]
    temp = [seq_dict[i] for i in temp]
    return temp
from sklearn.preprocessing import OneHotEncoder
# enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
# enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))
class SeqDataset(data.Dataset):
    def __init__(self,data_file,max_len=200,data_type='protein',onehot=False ):
        if data_type == 'protein':
            word_dict = seq_dict
            upper = True 
        elif data_type == 'drug':
            word_dict = smile_dict
            upper = False
        self.onehot = onehot

        # df = pd.read_csv(data_file,index_col=0,header=0)
        df = read_df_or_parquet(data_file,save_parquet=False,index_col=0)
        df.iloc[:,0] = df.iloc[:,0].astype(str)
        df['entry'] = df.iloc[:,0].apply(trans_seqs,max_seq_len=max_len,seq_dict=word_dict,upper=upper)
        print(df.head(4))
        self.df = df 
        self.entry_dict = df['entry'].to_dict()
        self.entryIDs= self.df.index.values 
        self.num_samples = self.df.shape[0]

        if onehot:
            self.encoder = OneHotEncoder(sparse=False).fit(np.arange(len(word_dict)).reshape(-1, 1))
            # self.entry_dict = {k:self.encoder.transform(np.reshape(v,(-1,1))).transpose() for k,v in self.entry_dict.items()} #[seq_len,num_embed,]-> [num_embed,seq_len] 
            self.org_entry_dict  = deepcopy(self.entry_dict )
            self.entry_dict  = {}


    def __getitem__(self, idx):
        entryID = self.entryIDs[idx]
        if self.onehot and (entryID not in self.entry_dict) :
            self.entry_dict[entryID] = self.encoder.transform(np.reshape(self.org_entry_dict[entryID],(-1,1))).transpose()
        data = self.entry_dict[entryID]

        if self.onehot:
            return t.Tensor(data).float()
        else:
            return t.Tensor(data).long()
        
    def __len__(self):
        return self.num_samples

class MultiEmbedDataset_v1(data.Dataset):
    def __init__(self,*args):
        self.datasets = args #list 
        self.num_samples = len(self.datasets[0])
        self.entryIDs = self.datasets[0].entryIDs
        self.datset2_entryIDs = self.datasets[1].entryIDs.tolist()
        if not np.all(self.entryIDs ==  self.datset2_entryIDs ):
            print('the order of entryIDs are not the same in dataset 1 & 2 .')
            self.datasets2_idx = {idx: self.datset2_entryIDs.index(entry)  for idx,entry in  enumerate(self.entryIDs)}
        else:
            self.datasets2_idx  = {idx:idx  for idx,entry  in enumerate(self.entryIDs)}
        for x,y in self.datasets2_idx.items():
            if self.datasets[0].entryIDs[x] != self.datasets[1].entryIDs[y]:
                print('id1 != id2 !',self.datasets[0].entryIDs[x], self.datasets[1].entryIDs[y])
                raise 
        print('checking entryIDs finished for MultiEmbedDataset_v1.')

    def __getitem__(self, idx):
        if len(self.datasets) > 2:
            return [self.datasets[0][idx], self.datasets[1][idx], self.datasets[2][self.datasets2_idx[idx]]]
        else:
            return [self.datasets[0][idx], self.datasets[1][self.datasets2_idx[idx]]]

    def __len__(self):
        return self.num_samples

class PairedDataset_v1(t.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,entry1,entry2,entry_pairs,pair_labels,
                ):
        self.entry1  = entry1
        self.entry2 = entry2
        self.entry_pairs = entry_pairs
        self.pair_labels = pair_labels
        self.entry1_ids = self.entry1.entryIDs.tolist()
        self.entry2_ids = self.entry2.entryIDs.tolist()
        self.num_samples = self.pair_labels.shape[0]

    def __getitem__(self, idx):
        tmp1,tmp2 = self.entry_pairs[idx]
        return ((self.entry1[self.entry1_ids.index(tmp1)],
                 self.entry2[self.entry2_ids.index(tmp2)]),
                self.pair_labels[idx])
        
    def __len__(self):
        return self.num_samples



class DeepDrug_Dataset(LightningDataModule):
    def __init__(self,entry1_data_folder,entry2_data_folder,entry_pairs_file,
        pair_labels_file,cv_file=None,cv_fold=0,batch_size=256,task_type='binary', n_confs=0,
        y_transfrom_func=None,category=None,entry1_seq_file=None, entry2_seq_file = None,
        split_strat='random'):  
        
        super().__init__()

        assert task_type in ['regression','binary_classification','binary',
                            'multi_classification','multi_label_classification',
                            'multilabel_classification','multiclass','multilabel',
                            ]
        self.has_setup_fit = False
        self.task_type = task_type
        self.entry1_data_folder = entry1_data_folder
        self.entry2_data_folder = entry2_data_folder
        self.entry_pairs_file = entry_pairs_file
        self.pair_labels_file = pair_labels_file
        self.cv_file = cv_file 
        self.cv_fold = cv_fold 
        self.batch_size = batch_size 
        self.y_transfrom_func = y_transfrom_func
        self.split_strat = split_strat
        self.bond_data_file = f'bond_data_{n_confs}' if n_confs > 0 else None

        self.entry1_seq_len=200

        self.category = category
        if self.category == 'DDI':
            self.entry2_seq_len = 200 
            self.entry2_type = 'drug'
        elif self.category == 'DTA':
            self.entry2_seq_len = 1000 
            self.entry2_type = 'protein'
        else:raise 


        self.entry2_multi_embed = True 
        self.entry2_seq_file = entry2_seq_file
        self.entry1_multi_embed = True 
        self.entry1_seq_file = entry1_seq_file

        
        assert self.entry2_type in ['protein','drug']



    def prepare_data(self):
        pass

    def my_prepare_data(self):
        print('preparing dataset...')
        if os.path.isfile(self.entry1_data_folder ):


            self.entry1_dataset = SeqDataset(self.entry1_data_folder,data_type='drug',max_len=self.entry1_seq_len,onehot=True)
        else:
            self.entry1_dataset = EntryDataset(root_folder=self.entry1_data_folder,filename='drug_data' if self.bond_data_file else 'data')
            self.entry1_dataset.add_node_degree()

            if self.entry1_multi_embed  == True:
                # first embeding: graph 
                # second embedding: sequence
                print('using drug sequences file:',self.entry1_data_folder )
                if self.bond_data_file:
                    self.entry1_bond_dataset = EntryDataset(root_folder=self.entry1_data_folder,filename=self.bond_data_file)
                    self.entry1_seq_dataset = SeqDataset(self.entry1_seq_file,data_type='drug',max_len=self.entry1_seq_len,onehot=True)
                    self.entry1_dataset = MultiEmbedDataset_v1(self.entry1_dataset,self.entry1_bond_dataset,self.entry1_seq_dataset)
                else:
                    self.entry1_seq_dataset = SeqDataset(self.entry1_seq_file,data_type='drug',max_len=self.entry1_seq_len,onehot=True)
                    self.entry1_dataset = MultiEmbedDataset_v1(self.entry1_dataset,self.entry1_seq_dataset)
            elif self.entry1_multi_embed  == False: pass
            else: raise 

        if self.entry2_data_folder is not  None:
            # print('entry2_data_folder is not None, use it.')
            if os.path.isfile(self.entry2_data_folder ):
                self.entry2_dataset = SeqDataset(self.entry2_data_folder,data_type=self.entry2_type, max_len=self.entry2_seq_len,onehot=True)
            else:
                self.entry2_dataset = EntryDataset(root_folder=self.entry2_data_folder,filename='drug_data' if self.bond_data_file else 'data')
                self.entry2_dataset.add_node_degree()
                    
            if self.entry2_multi_embed  == True:
                print('using target sequences file:',self.entry2_data_folder )
                if self.bond_data_file:
                    self.entry2_bond_dataset = EntryDataset(root_folder=self.entry2_data_folder,filename=self.bond_data_file)
                    self.entry2_seq_dataset = SeqDataset(self.entry2_seq_file,data_type=self.entry2_type,max_len=self.entry2_seq_len,onehot=True)
                    self.entry2_dataset = MultiEmbedDataset_v1(self.entry2_dataset,self.entry2_bond_dataset,self.entry2_seq_dataset)
                else:
                    self.entry2_seq_dataset = SeqDataset(self.entry2_seq_file,data_type=self.entry2_type,max_len=self.entry2_seq_len,onehot=True)
                    self.entry2_dataset = MultiEmbedDataset_v1(self.entry2_dataset,self.entry2_seq_dataset)
            elif self.entry2_multi_embed  == False: pass
            else: raise 

        else:
            self.entry2_dataset = self.entry1_dataset

        self.pair_labels = pd.read_csv(self.pair_labels_file,header=0,index_col=None).values#.astype(self.y_type)
        self.entry_pairs = pd.read_csv(self.entry_pairs_file,header=0,index_col = None).values#.astype(str)
        self.data_split = evaluation.Split_Strats(self.entry_pairs_file,self.entry1_seq_file,clustering='cluster' in self.split_strat)
        # print('entry_pairs:',self.entry_pairs[:5])

        # y_transform
        if self.y_transfrom_func is not None:
            self.pair_labels = self.y_transfrom_func(self.pair_labels)


        if self.task_type in ['binary_classification','binary','multiclass','multilabel',
                            'multi_classification','multi_label_classification','multilabel_classification',]:
            self.pair_labels = t.from_numpy(self.pair_labels).long()
        elif self.task_type in ['regression',]:
            self.pair_labels = t.from_numpy(self.pair_labels).float()


        if self.cv_file is not None :
            with open(self.cv_file,'rb') as f:
                cv_dict = pkl.load(f)        
            self.train_indexs  = cv_dict[self.cv_fold]['train']
            self.valid_indexs = cv_dict[self.cv_fold]['valid']
            self.test_indexs = cv_dict[self.cv_fold]['test']
        else:
            print('can not find cv_file,  '+self.split_strat+' ...')
            if self.split_strat == 'one_unknown':
                train,valid,test = self.data_split.one_known_split()
                self.train_indexs = train.index
                self.valid_indexs = valid.index
                self.test_indexs = test.index
            elif self.split_strat == 'both_unknown':
                train,valid,test = self.data_split.both_unknown_split()
                self.train_indexs = train.index
                self.valid_indexs = valid.index
                self.test_indexs = test.index
            elif self.split_strat == 'sample_from_all_clusters':
                train,valid,test = self.data_split.all_cluster_split('neither')
                self.train_indexs = train.index
                self.valid_indexs = valid.index
                self.test_indexs = test.index
            elif self.split_strat == 'whole_cluster_sampling':
                train,valid,test = self.data_split.chosen_cluster_split('neither')
                self.train_indexs = train.index
                self.valid_indexs = valid.index
                self.test_indexs = test.index
            elif self.split_strat == 'sample_from_all_clusters+one_unknown':
                train,valid,test = self.data_split.all_cluster_split('one_unknown')
                self.train_indexs = train.index
                self.valid_indexs = valid.index
                self.test_indexs = test.index
            elif self.split_strat == 'sample_from_all_clusters+both_unknown':
                train,valid,test = self.data_split.all_cluster_split('both_unknown')
                self.train_indexs = train.index
                self.valid_indexs = valid.index
                self.test_indexs = test.index
            elif self.split_strat == 'whole_cluster_sampling+one_unknown':
                train,valid,test = self.data_split.chosen_cluster_split('one_unknown')
                self.train_indexs = train.index
                self.valid_indexs = valid.index
                self.test_indexs = test.index
            elif self.split_strat == 'whole_cluster_sampling+both_unknown':
                train,valid,test = self.data_split.chosen_cluster_split('both_unknown')
                self.train_indexs = train.index
                self.valid_indexs = valid.index
                self.test_indexs = test.index
            elif self.split_strat == 'random':
                num_samples = self.pair_labels.shape[0]
                tmp_indexs = np.random.permutation(num_samples)
                self.train_indexs  = tmp_indexs[:int(num_samples*0.7)]
                self.valid_indexs = tmp_indexs[int(num_samples*0.7):int(num_samples*0.8)]
                self.test_indexs = tmp_indexs[int(num_samples*0.8):]
            else:
                raise ValueError(self.split_strat +"is not an available split strategy.")

    def setup(self, stage = None):
        if not self.has_setup_fit:
            self.my_prepare_data()
            self.has_setup_fit = True
        pass

    def train_dataloader(self):
        print('in train dataloader...')
        train_pairs,train_labels = self.entry_pairs[self.train_indexs],self.pair_labels[self.train_indexs]
    
        train_split = PairedDataset_v1(self.entry1_dataset,self.entry2_dataset,
            train_pairs,train_labels)

        dataloader = DataLoader(train_split,num_workers=0,shuffle=True,batch_size=self.batch_size,pin_memory=True) 
        return dataloader 
    def val_dataloader(self):
        print('in val dataloader...')
        val_split = PairedDataset_v1(self.entry1_dataset,self.entry2_dataset,
            self.entry_pairs[self.valid_indexs],self.pair_labels[self.valid_indexs])
        return DataLoader(val_split,num_workers=0,shuffle=False,batch_size=self.batch_size ) 

    def test_dataloader(self):
        print('in test dataloader...')
        test_split = PairedDataset_v1(self.entry1_dataset,self.entry2_dataset,
            self.entry_pairs[self.test_indexs],self.pair_labels[self.test_indexs])
        return DataLoader(test_split,num_workers=0,shuffle=False,batch_size=self.batch_size,) 

class MolDataset(t.utils.data.Dataset):
    def __init__(self,dataset,entry_ids):
        self.dataset = dataset
        self.dataset_entry_ids = dataset.entryIDs.tolist()
        self.entry_ids = entry_ids
        self.num_samples = len(entry_ids)
    
    def __getitem__(self,idx):
        return self.dataset[self.dataset_entry_ids.index(self.entry_ids[idx])]
    
    def __len__(self):
        return self.num_samples
    
class Pretraining_Dataset(LightningDataModule):
    def __init__(self,data_folder,batch_size=128,task_type='pretrain',split_strat='random'):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.task_type = task_type
        self.split_strat = split_strat
    
    def prepare_data(self):
        pass

    def my_prepare_data(self):
        self.dataset = EntryDataset(self.data_folder)    
        self.dataset.add_node_degree()
        
        self.data_split = evaluation.Split_Strats(None,self.data_folder+'cleaned_data.csv',clustering='cluster' in self.split_strat)
        #generate train test split indices
        if self.split_strat == 'random':
            num_samples = len(self.dataset)
            tmp_indxs = np.random.permutation(num_samples)
            self.train_indxs = tmp_indxs[:int(num_samples*0.8)]
            self.valid_indxs = tmp_indxs[int(num_samples*0.8):]
            # self.test_indxs = tmp_indxs[int(num_samples*0.8):]
        elif self.split_strat == 'sample_from_all_clusters':
            train,evals = self.data_split.all_cluster_split_pretrain()
            self.train_indxs = train.index
            self.valid_indxs = evals.index
        elif self.split_strat == 'whole_cluster_sampling':
            train,evals = self.data_split.chosen_cluster_split_pretrain()
            self.train_indxs = train.index
            self.valid_indxs = evals.index
        else:
            raise

    
    def setup(self, stage=None):
        self.my_prepare_data()
        pass
    
    def train_dataloader(self):
        print('in train dataloader...')
        train_ids = self.dataset.entryIDs[self.train_indxs]
        train_split = MolDataset(self.dataset,train_ids)
        dataloader = DataLoader(train_split,num_workers=0,shuffle=True,batch_size=64,pin_memory=True)
        return dataloader
    
    def val_dataloader(self):
        print('in val dataloader...')
        valid_ids = self.dataset.entryIDs[self.valid_indxs]
        valid_split = MolDataset(self.dataset,valid_ids)
        dataloader = DataLoader(valid_split,num_workers=0,shuffle=False,batch_size=64)
        return dataloader
    
    def test_dataloader(self):
        print('in test dataloader...')
        test_ids = self.dataset.entryIDs[self.test_indxs]
        test_split = MolDataset(self.dataset,test_ids)
        dataloader = DataLoader(test_split,num_workers=0,shuffle=False,batch_size=64)
        return dataloader
    
class SMILES_DataModule(LightningDataModule):
    def __init__(self,datafile,batch_size,data_type,split_strat,max_len=200):
        super().__init__()
        self.datafile = datafile
        self.batch_size = batch_size
        self.data_type = data_type
        self.max_len = max_len
        self.split_strat = split_strat
    
    def prepare_data(self):
        pass
    
    def my_prepare_data(self):
        print(self.data_type)
        self.dataset = SeqDataset(self.datafile, self.max_len, self.data_type, onehot=True)
        self.data_split = evaluation.Split_Strats(None,self.datafile,clustering='cluster' in self.split_strat)
        if self.split_strat == 'random':
            num_samples = len(self.dataset)
            tmp_indxs = np.random.permutation(num_samples)
            self.train_indxs = tmp_indxs[:int(num_samples*0.8)]
            self.valid_indxs = tmp_indxs[int(num_samples*0.8):]
        elif self.split_strat == 'sample_from_all_clusters':
            train,evals = self.data_split.all_cluster_split_pretrain()
            self.train_indxs = train.index
            self.valid_indxs = evals.index
        elif self.split_strat == 'whole_cluster_sampling':
            train,evals = self.data_split.chosen_cluster_split_pretrain()
            self.train_indxs = train.index
            self.valid_indxs = evals.index
        else:
            raise
    
    def setup(self, stage=None):
        self.my_prepare_data()
        pass
    
    def train_dataloader(self):
        print('in train dataloader...')
        train_ids = self.dataset.entryIDs[self.train_indxs]
        train_split = SeqDataset(self.datafile, self.max_len, self.data_type, onehot=True)
        train_split.entryIDs = train_ids
        train_split.num_samples = len(train_split.entryIDs)
        dataloader = DataLoader(train_split,num_workers=0,shuffle=True,batch_size=64,pin_memory=True)
        return dataloader
    
    def val_dataloader(self):
        print('in val dataloader...')
        valid_ids = self.dataset.entryIDs[self.valid_indxs]
        valid_split = SeqDataset(self.datafile, self.max_len, self.data_type, onehot=True)
        valid_split.entryIDs = valid_ids
        valid_split.num_samples = len(valid_split.entryIDs)
        dataloader = DataLoader(valid_split,num_workers=0,shuffle=False,batch_size=64)
        return dataloader
    
    def test_dataloader(self):
        print('in test dataloader...')
        test_ids = self.dataset.entryIDs[self.test_indxs]
        test_split = SeqDataset(self.datafile, self.max_len, self.data_type, onehot=True)
        dataloader = DataLoader(test_split,num_workers=0,shuffle=False,batch_size=64)
        return dataloader