import pandas as pd
from dataset import EntryDataset
import sys

if __name__ == '__main__':
    save_folder = './dataset/Chemberta/drug/'
    dataset = EntryDataset(save_folder)

    drug_df_path = './dataset/Chemberta/drug/drug.csv'
    dataset.large_drug_process(drug_df_path)
