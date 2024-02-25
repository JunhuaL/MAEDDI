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
from molGraphConvFeaturizer import ConfGraphFeaturizer
from molGraphConvFeaturizer import MolGraphConvFeaturizer

def graph_add_degree(data,degree='both'):
    # print('data:',data,type(data))
    assert degree in ['indegree','outdegree','both']

    deg1 = geo_degree(data.edge_index[0],data.num_nodes).reshape(-1,1)
    deg2 = geo_degree(data.edge_index[1],data.num_nodes).reshape(-1,1)
    data.x = t.cat([data.x,deg1,deg2],dim=1)
    return data 

def discretize(values,bins,upper):
    step_size = upper//bins
    ohe_feats = np.digitize(values,range(step_size,upper,step_size))
    ohe = np.eye(bins)[ohe_feats]
    return ohe

class Large_MolDataset(t.utils.data.Dataset):
    def __init__(self, data_dir, filename_base = 'data'):
        filenames = os.listdir(os.path.join(data_dir,'processed'))
        self.data_paths = [os.path.join(data_dir,'processed',name) for name in filenames if filename_base in name]
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

        index = np.argsort(self.dataset_entryIDs)
        sorted_entryIDs = self.dataset_entryIDs[index]

        self.entryIDs_idxs = np.searchsorted(sorted_entryIDs,self.entryIDs)

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
    def __init__(self,data_folder,use_seq=False,use_conf=False,batch_size=128):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.use_seq = use_seq
        self.use_conf = use_conf

    def prepare_data(self):
        pass

    def my_prepare_data(self):
        self.dataset = Large_MolDataset(self.data_folder)
        if self.use_seq:
            csv_dir = '/'.join(self.data_folder.split('/')[:-2]) + '/drug.csv' 
            seq_data = SeqDataset(csv_dir,data_type='drug',onehot=True)
            self.dataset = Large_MultiEmbedDataset([self.dataset,seq_data])
        if self.use_conf:
            conf_data = Large_ConfDataset(self.data_folder,'conf')
            self.dataset = Large_MultiEmbedDataset([self.dataset,conf_data])

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


class Large_ConfDataset(InMemoryDataset):
    def __init__(self,root_folder,filename='data'):
        os.makedirs(os.path.join(root_folder,'processed'),exist_ok=True)
        super(Large_ConfDataset, self).__init__(root_folder)

        self.root_folder = root_folder
        self.filename = filename
        zeroth_partition = os.path.join(root_folder,'processed',f'{filename}_0.pt')
        if os.path.exists(zeroth_partition):
            self.load()
        else:
            print("Unprocessed data")
        
    @property
    def processed_file_names(self,):
        return '{}.pt'.format(self.filename)

    def add_node_degree(self,):
        print('add degrees as node features for each sample...')
        data_list = [graph_add_degree(data) for data in self]
        data,slices = self.collate(data_list)

        self.data = data 
        self.slices = slices
        self.__data_list__ = data_list
        self._data_list = data_list 

    def load(self):
        partition_dir = os.path.join(self.root_folder,'processed')
        partition_names = [file for file in os.listdir(partition_dir) if self.filename in file]
        self.data = []
        self.entryIDs = []
        for file in partition_names:
            temp_data,temp_slices,temp_ids = t.load(os.path.join(partition_dir,file))
            for i in range(len(temp_ids)):
                x = separate(cls=temp_data.__class__,
                             batch=temp_data,
                             idx=i,
                             slice_dict=temp_slices,
                             decrement=False
                        )
                self.data.append(graph_add_degree(x))
            self.entryIDs.append(temp_ids)
        
        self.entryIDs = np.concatenate(self.entryIDs)
        self.data,self.slices,_ = collate(cls=self.data[0].__class__,
                                          data_list=self.data,
                                          increment=False,
                                          add_batch=False)

        self.num_samples = len(self.entryIDs)
        
    def conf_process(self,sdfile,partition_size,n_bins=6,default_dim_features=11,default_dim_nodes=40,data_type='conf'):
        from rdkit import Chem
        from tqdm import tqdm
        import gc

        sdf = Chem.SDMolSupplier(sdfile)
        num_mols = len(sdf)
        full_partitions = num_mols // partition_size
        remainder_partition = 0 if (num_mols % partition_size) == 0 else 1
        total_partitions = full_partitions + remainder_partition
        working_base = 0

        if data_type == 'conf':
            featurizer = ConfGraphFeaturizer(n_bins)
        else:
            featurizer = MolGraphConvFeaturizer(False,True,True)
            
        print("Initialization complete")
        for p in range(total_partitions):
            print(f"Processing partition {p}")
            mols = []
            drug_ids = []

            if (working_base + partition_size) > num_mols:
                for i in range(num_mols - working_base):
                    mol = next(sdf)
                    mols.append(mol)
                    drug_ids.append(mol.GetProp('SOURCE_ID'))
            else:
                for i in range(partition_size):
                    mol = next(sdf)
                    mols.append(mol)
                    drug_ids.append(mol.GetProp('SOURCE_ID'))
            

            drug_ids = np.asarray(drug_ids)
            chemslist = featurizer.featurize(mols)
            print("Featurizing Conformers.")
            data_list = []

            for convMol in tqdm(chemslist):
                if isinstance(convMol,np.ndarray) :
                    feat_mat=np.zeros((default_dim_nodes,default_dim_features+1))
                    edges = np.array([[],[]])
                    edges_attr = np.array([])
                
                else:
                    feat_mat = convMol.node_features
                    edges_attr = convMol.edge_features
                    edges = convMol.edge_index

                data_list.append(Data(x=t.from_numpy(feat_mat).float(),
                                    edge_index=t.from_numpy(edges).long(),
                                    edge_attr=t.from_numpy(edges_attr).float() if edges_attr is not None else None))
                
            data,slices = self.collate(data_list)
            partition_path = self.processed_paths[0].split('.')
            partition_path[-2] = partition_path[-2] + f'_{p}'
            partition_path = '.'.join(partition_path)
            t.save((data,slices,drug_ids),partition_path)
            del data
            del slices
            del mols
            del drug_ids
            del chemslist
            del data_list
            gc.collect()

            working_base = working_base + partition_size if (working_base + partition_size) < num_mols else working_base + (num_mols - working_base)
