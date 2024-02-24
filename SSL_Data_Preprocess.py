from large_dataset import Large_ConfDataset
import sys

root_dir = sys.argv[1]
filename = sys.argv[2]

data_mod = Large_ConfDataset(root_dir,filename)

sdf_file = sys.argv[3]
data_type = sys.argv[4]

if data_type == 'mol':
    default_dim = 119
    default_nodes = 50
else:
    default_dim = 11
    default_nodes = 40

data_mod.conf_process(sdf_file,10000,default_dim_features=default_dim,default_dim_nodes=default_nodes,data_type=data_type)  