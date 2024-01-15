import pandas as pd
from dataset import EntryDataset
import sys

from_i,to_i = sys.argv[1], sys.argv[2]

save_folder = './dataset/Chemberta/drug/'

drug_df = pd.read_csv(save_folder+'drug.csv',skiprows=from_i,nrows=to_i,index_col=False)
dataset = EntryDataset(save_folder)
dataset.drug_process(drug_df)