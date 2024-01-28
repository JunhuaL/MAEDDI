import os
import re
import numpy as np 
import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch as t
from torch_geometric.data import (InMemoryDataset, Data,DataLoader)
from torch_geometric.utils import (dense_to_sparse,to_undirected,add_self_loops,remove_self_loops,degree)
from torch_geometric.data.separate import separate
from torch_geometric.utils import degree as geo_degree

from pytorch_lightning import (LightningDataModule,)

def graph_add_degree(data,degree='both'):
    # print('data:',data,type(data))
    assert degree in ['indegree','outdegree','both']

    deg1 = geo_degree(data.edge_index[0],data.num_nodes).reshape(-1,1)
    deg2 = geo_degree(data.edge_index[1],data.num_nodes).reshape(-1,1)
    data.x = t.cat([data.x,deg1,deg2],dim=1)
    return data 

class Large_MolDataset(t.utils.data.Dataset):
    def __init__(self, data_dir,split_entry_ids=None):
        filenames = os.listdir(data_dir)
        self.data_paths = [data_dir+name for name in filenames]
        
        self.all_entry_ids = []
        self.partition_bases = [0]
        for i in range(len(self.data_paths)):
            _,_,entry_ids = t.load(self.data_paths[i])
            self.all_entry_ids.append(entry_ids)
            self.partition_bases.append(len(entry_ids) + self.partition_bases[i])
        self.all_entry_ids = np.concatenate(self.all_entry_ids)
        self.split_entry_ids = split_entry_ids if split_entry_ids is not None else self.all_entry_ids
        self.num_samples = len(self.split_entry_ids)
    
    def __getitem__(self, idx):
        idx = np.where(self.all_entry_ids == self.split_entry_ids[idx])[0][0]
        for i in range(len(self.partition_bases)-1):
            if idx >= self.partition_bases[i] and idx < self.partition_bases[i+1]:
                partition_b = self.partition_bases[i]
                partition_i = i
                break

        working_file = self.data_paths[partition_i]
        offset = idx - partition_b
        data,slices,entry_id = t.load(working_file)
        x = separate(
            cls=data.__class__,
            batch=data,
            idx=offset,
            slice_dict=slices,
            decrement=False,
        )

        return graph_add_degree(x)
    
    def __len__(self):
        return self.num_samples


class Large_PretrainingDataset(LightningDataModule):
    def __init__(self,data_folder,batch_size=128):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def my_prepare_data(self):
        self.dataset = []
        files = [self.data_folder + file for file in os.listdir(self.data_folder)]
        for file in files:
            _,_,entry_ids = t.load(file)
            self.dataset.append(entry_ids)
        self.dataset = np.concatenate(self.dataset)

        num_samples = len(self.dataset)
        tmp_indxs = np.random.permutation(num_samples)
        self.train_indxs = tmp_indxs[:int(num_samples*0.9)]
        self.valid_indxs = tmp_indxs[:int(num_samples*0.9):]
    
    def setup(self, stage=None):
        self.my_prepare_data()
        pass

    def train_dataloader(self):
        train_ids = self.dataset[self.train_indxs]
        train_split = Large_MolDataset(self.data_folder,train_ids)
        dataloader = DataLoader(train_split,num_workers=2,shuffle=True,batch_size=self.batch_size,pin_memory=True,drop_last=True)
        return dataloader

    def val_dataloader(self):
        valid_ids = self.dataset[self.valid_indxs]
        valid_split = Large_MolDataset(self.data_folder,valid_ids)
        dataloader = DataLoader(valid_split,num_workers=2,shuffle=False,batch_size=self.batch_size,pin_memory=True,drop_last=True)
        return dataloader  
