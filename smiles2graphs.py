import pandas as pd
from dataset import EntryDataset
import sys

from_i,to_i = int(sys.argv[1]), int(sys.argv[2])

save_folder = './dataset/Chemberta/drug/'

partition_i = from_i//100000

drug_df = pd.read_csv(save_folder+'drug_10m.csv',skiprows=from_i,nrows=100000,header=None)
drug_df.columns = ['drugID','SMILES']
dataset = EntryDataset(save_folder,filename=f'data_p{partition_i}')
dataset.drug_process(drug_df)
