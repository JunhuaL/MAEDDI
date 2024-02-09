import os
import numpy as np 
import pandas as pd
import torch as t
from torch_geometric.data import (InMemoryDataset, Data,DataLoader)
from torch_geometric.data.separate import separate
from torch_geometric.data.collate import collate
from torch_geometric.utils import degree as geo_degree
from pytorch_lightning import (LightningDataModule,)
from dataset import SeqDataset

def graph_add_degree(data,degree='both'):
    # print('data:',data,type(data))
    assert degree in ['indegree','outdegree','both']

    deg1 = geo_degree(data.edge_index[0],data.num_nodes).reshape(-1,1)
    deg2 = geo_degree(data.edge_index[1],data.num_nodes).reshape(-1,1)
    data.x = t.cat([data.x,deg1,deg2],dim=1)
    return data 

class Large_MolDataset(t.utils.data.Dataset):
    def __init__(self, data_dir):
        filenames = os.listdir(data_dir)
        self.data_paths = [data_dir+name for name in filenames]

        self.data = [] 
        self.entryIDs = []
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
            self.entryIDs.append(entry_ids)
        
        self.entryIDs = np.concatenate(self.entryIDs)
        self.data,self.slices,_ = collate(cls=self.data[0].__class__,
                                          data_list=self.data,
                                          increment=False,
                                          add_batch=False)

        self.num_samples = len(self.entryIDs)

    def __getitem__(self, idx):
        data = separate(cls=self.data.__class__,
                        batch=self.data,
                        idx=idx,
                        slice_dict=self.slices,
                        decrement=False)
        return data
    
    def __len__(self):
        return self.num_samples

class Large_MolWrapper(t.utils.data.Dataset):
    def __init__(self,dataset,entryIDs):
        self.dataset = dataset
        self.entryIDs = entryIDs
        self.num_samples = len(entryIDs)
        self.dataset_entryIDs = dataset.entryIDs
        self.entryIDs_idxs = np.searchsorted(self.dataset_entryIDs,self.entryIDs)
        
    def __getitem__(self, idx):
        return self.dataset[self.entryIDs_idxs[idx]]
    
    def __len__(self):
        return self.num_samples

class Large_MultiEmbedDataset(t.utils.data.Dataset):
    def __init__(self,datasets):
        self.datasets = datasets
        self.num_samples = len(self.datasets[0])
        self.entryIDs = self.datasets[0].entryIDs
        print('checking entryIDs finished for Mol_Wrapper.')
    
    def __getitem__(self,idx):
        return [self.datasets[0][idx], self.datasets[1][idx]]
    
    def __len__(self):
        return self.num_samples

class Large_PretrainingDataset(LightningDataModule):
    def __init__(self,data_folder,use_seq=False,batch_size=128):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.use_seq = use_seq

    def prepare_data(self):
        pass

    def my_prepare_data(self):
        self.dataset = Large_MolDataset(self.data_folder)
        if self.use_seq:
            csv_dir = '/'.join(self.data_folder.split('/')[:-2]) + '/drug.csv' 
            self.seq_data = SeqDataset(csv_dir,data_type='drug',onehot=True)
            self.dataset = Large_MultiEmbedDataset([self.dataset,self.seq_data])
        
        num_samples = len(self.dataset)
        tmp_indxs = np.random.permutation(num_samples)
        self.train_indxs = tmp_indxs[:int(num_samples*0.9)]
        self.valid_indxs = tmp_indxs[int(num_samples*0.9):]
    
    def setup(self, stage=None):
        self.my_prepare_data()
        pass

    def train_dataloader(self):
        print("In train loader..")
        train_ids = self.dataset.entryIDs[self.train_indxs]
        train_split = Large_MolWrapper(self.dataset,train_ids)
        dataloader = DataLoader(train_split,num_workers=0,shuffle=False,batch_size=self.batch_size,pin_memory=True,drop_last=True)
        return dataloader

    def val_dataloader(self):
        print("In valid loader..")
        valid_ids = self.dataset.entryIDs[self.valid_indxs]
        valid_split = Large_MolWrapper(self.dataset,valid_ids)
        dataloader = DataLoader(valid_split,num_workers=0,shuffle=False,batch_size=self.batch_size,pin_memory=True,drop_last=True)
        return dataloader  
