import os
import numpy as np 
import pandas as pd
import torch as t
from torch_geometric.data import (InMemoryDataset, Data,DataLoader)
from torch_geometric.data.separate import separate
from torch_geometric.data.collate import collate
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

        self.data = [] 
        self.entry_ids = []
        for file in self.data_paths:
            temp_data,slices,entry_ids = t.load(file)
            for i in range(len(entry_ids)):
                x = separate(cls=temp_data.__class__,
                                batch=temp_data,
                                idx=i,
                                slice_dict=slices,
                                decrement=False
                        )
                self.data.append(graph_add_degree(x))
            self.entry_ids.append(entry_ids)
        
        self.entry_ids = np.concatenate(self.entry_ids)
        self.data,self.slices,_ = collate(cls=self.data[0].__class__,
                                          data_list=self.data,
                                          increment=False,
                                          add_batch=False)

        if split_entry_ids is not None:
            index = np.argsort(self.entry_ids)
            ypos = np.searchsorted(self.entry_ids[index],split_entry_ids)
            split_indices = index[ypos]
            
            temp_data = []
            self.entry_ids = split_entry_ids
            for i in split_indices:
                x = separate(cls=self.data.__class__,
                            batch=self.data,
                            idx=i,
                            slice_dict=self.slices,
                            decrement=False
                        )
                temp_data.append(x)
            self.data, self.slices, _ = collate(cls=temp_data[0].__class__,
                                                data_list=temp_data,
                                                increment=False,
                                                add_batch=False
                                            )

        self.num_samples = len(self.entry_ids)

    def __getitem__(self, idx):
        data = separate(cls=self.data.__class__,
                        batch=self.data,
                        idx=idx,
                        slice_dict=self.slices,
                        decrement=False)
        return data
    
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
        self.valid_indxs = tmp_indxs[int(num_samples*0.9):]
    
    def setup(self, stage=None):
        self.my_prepare_data()
        pass

    def train_dataloader(self):
        train_ids = self.dataset[self.train_indxs]
        train_split = Large_MolDataset(self.data_folder,train_ids)
        dataloader = DataLoader(train_split,num_workers=0,shuffle=True,batch_size=self.batch_size,pin_memory=True,drop_last=True)
        return dataloader

    def val_dataloader(self):
        valid_ids = self.dataset[self.valid_indxs]
        valid_split = Large_MolDataset(self.data_folder,valid_ids)
        dataloader = DataLoader(valid_split,num_workers=0,shuffle=False,batch_size=self.batch_size,pin_memory=True,drop_last=True)
        return dataloader  
