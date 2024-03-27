from large_dataset import Large_ConfDataset

datamod = Large_ConfDataset('./dataset/Namiki/drug/')

datamod.conf_process('./dataset/Namiki/drug/mols.sdf',10000,default_dim_features=119,default_dim_nodes=50,data_type='mol')