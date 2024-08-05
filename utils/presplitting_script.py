import os
import argparse

from evaluation import Split_Strats

def main(args):
    dataset = args.dataset
    data_dir = f'./dataset/{dataset}/'

    if not os.path.isdir(os.path.join(data_dir,'splits')):
        os.makedirs(os.path.join(data_dir,'splits'),exist_ok=True)

    entry_pairs_pth = os.path.join(data_dir,'binary_1vs1','entry_pairs.csv')
    drug_pth = os.path.join(data_dir,'drug','drug.csv')
    split_strat = args.split_strat
    no_splits = args.no_splits

    save_path = os.path.join(data_dir,'splits',split_strat)
    os.makedirs(save_path, exist_ok=True)

    for i in range(no_splits):
        part_folder = os.path.join(save_path,str(i))
        os.makedirs(part_folder,exist_ok=True)
        splitter = Split_Strats(entry_pairs_pth,drug_pth,clustering=True)
        split_func = getattr(splitter,split_strat)
        train,valid,test = split_func('neither')
        train.to_csv(os.path.join(part_folder,'train.csv'))
        valid.to_csv(os.path.join(part_folder,'valid.csv'))
        test.to_csv(os.path.join(part_folder,'test.csv'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation splits')
    
    parser.add_argument('--dataset', default='DrugBank', type=str)
    parser.add_argument('--split_strat', default='all_cluster_split', choices=['all_cluster_split','chosen_cluster_split'], type=str)
    parser.add_argument('--no_splits', default=3, type=int)

    args = parser.parse_args()
    main(args)