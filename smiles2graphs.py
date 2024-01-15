import pandas as pd
from dataset import EntryDataset

save_folder = './dataset/Chemberta/drug/'

drug_df = pd.read_csv(save_folder+'drug.csv')
dataset = EntryDataset(save_folder)
dataset.drug_process(drug_df)