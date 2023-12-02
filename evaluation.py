import numpy as np
import pandas as pd
import hdbscan
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem

class Split_Strats:
    def __init__(self,entry_data_file,drug_data_file,clustering=False):
        if entry_data_file:
            self.entry_data = pd.read_csv(entry_data_file)
        else:
            self.entry_data = None
        self.drug_data = pd.read_csv(drug_data_file)
        if clustering:
            mols = np.array(list(map(Chem.MolFromSmiles,self.drug_data['SMILES'])))
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol,2) for mol in mols if mol!=None]
            self.distMat = self.TanimotoDistMat(fps)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5,gen_min_span_tree=True,metric='precomputed')
            clusterer.fit(self.distMat)
            self.cluster_labels = clusterer.labels_
            null_idxs = np.where(mols==None)[0]
            self.drug_data = self.drug_data.drop(null_idxs)

    def TanimotoDistMat(self,fps):
        n = len(fps)
        similarities = np.zeros((n,n))
        for i in range(n):
            similarity = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i+1])
            similarities[i,:i+1] = similarity
            similarities[:i+1,i] = similarity
        return 1-similarities
        
    def one_known_split(self):
        train_data = self.entry_data
        one_out_data = []
        data_ratio = len(train_data)/len(self.entry_data)
        selected_drugs = [] 
        while data_ratio>0.7:
            drug = np.random.choice(self.drug_data.drugID)
            drug_entries = train_data[(train_data['entry1']==drug)|(train_data['entry2']==drug)]
            train_data = train_data[(train_data['entry1']!=drug)&(train_data['entry2']!=drug)]
            one_out_data.append(drug_entries)
            selected_drugs.append(drug)
            data_ratio = len(train_data)/len(self.entry_data)

        one_out_data = pd.concat(one_out_data)
        val_data = one_out_data.sample(frac=1/3)
        test_data = one_out_data.drop(val_data.index)
        return train_data, val_data, test_data

    def both_unknown_split(self):
        train_data = self.entry_data
        both_unknown_data = []
        data_ratio = len(train_data)/len(self.entry_data)
        selected_drugs = []
        while data_ratio>0.7:
            drug = np.random.choice(self.drug_data.drugID)
            drug_entries = train_data[(train_data['entry1']==drug)|(train_data['entry2']==drug)]
            train_data = train_data[(train_data['entry1']!=drug)&(train_data['entry2']!=drug)]
            both_unknown_data.append(drug_entries)
            selected_drugs.append(drug)
            data_ratio = len(train_data)/len(self.entry_data)

        both_unknown_data = pd.concat(both_unknown_data)
        both_unknown_data = both_unknown_data[(both_unknown_data['entry1'].isin(selected_drugs))
                                            & (both_unknown_data['entry2'].isin(selected_drugs))]
        val_data = both_unknown_data.sample(frac=1/3)
        test_data = both_unknown_data.drop(val_data.index)
        return train_data, val_data, test_data

    def all_cluster_split(self,method='one_unknown',entry_data=None):
        drugs = self.drug_data
        drugs['c_label'] = self.cluster_labels
        selected_drugs = []
        for cluster in np.unique(self.cluster_labels):
            unknown_drugs = drugs[drugs['c_label']==cluster].sample(frac=0.2)
            selected_drugs.append(unknown_drugs)
        selected_drugs = pd.concat(selected_drugs)
        drug_labels = selected_drugs['drugID']

        train_data = self.entry_data if entry_data is None else entry_data
        train_data = train_data[(~train_data['entry1'].isin(drug_labels))&(~train_data['entry2'].isin(drug_labels))]
        eval_data = self.entry_data.drop(train_data.index) if entry_data is None else entry_data.drop(train_data.index)

        if method == 'one_unknown':
            both_unknown_data = eval_data[(eval_data['entry1'].isin(drug_labels))&(eval_data['entry2'].isin(drug_labels))]
            eval_data = eval_data.drop(both_unknown_data.index)
        elif method == 'both_unknown':
            eval_data = eval_data[(eval_data['entry1'].isin(drug_labels))&(eval_data['entry2'].isin(drug_labels))]

        test_drugs = []
        for cluster in np.unique(self.cluster_labels):
            unknown_drugs = selected_drugs[selected_drugs['c_label']==cluster].sample(frac=0.6)
            test_drugs.append(unknown_drugs)
        test_drugs = pd.concat(test_drugs)
        drug_labels = test_drugs['drugID']

        val_data = eval_data
        val_data = val_data[(~val_data['entry1'].isin(drug_labels))&(~val_data['entry2'].isin(drug_labels))]
        test_data = eval_data.drop(val_data.index) 

        print(len(train_data)/(len(train_data)+len(val_data)+len(test_data)))
        print(len(val_data)/(len(train_data)+len(val_data)+len(test_data)))
        print(len(test_data)/(len(train_data)+len(val_data)+len(test_data)))
        return train_data, val_data, test_data

    def chosen_cluster_split(self,method='one_unknown',entry_data=None):
        drugs = self.drug_data
        drugs['c_label'] = self.cluster_labels
        train_data = self.entry_data if entry_data is None else entry_data
        data_ratio = len(train_data)/len(self.entry_data) if entry_data is None else len(train_data)/len(entry_data)
        selected_drugs = []
        eval_data = []
        unique_cls = np.unique(self.cluster_labels[self.cluster_labels>-1])
        eval_cls = []
        while data_ratio>0.7:
            clust_id = np.random.choice(unique_cls)
            eval_cls.append(clust_id)
            drug_ids = drugs[drugs['c_label'] == clust_id]['drugID'].values
            drug_entries = train_data[(train_data['entry1'].isin(drug_ids))|(train_data['entry2'].isin(drug_ids))]
            train_data = train_data[(~train_data['entry1'].isin(drug_ids))&(~train_data['entry2'].isin(drug_ids))]
            eval_data.append(drug_entries)
            selected_drugs+=list(drug_ids)
            data_ratio = len(train_data)/len(self.entry_data) if entry_data is None else len(train_data)/len(entry_data)

        eval_data = pd.concat(eval_data)
        if method == 'one_unknown':
            both_unknown_data = eval_data[(eval_data['entry1'].isin(selected_drugs))&(eval_data['entry2'].isin(selected_drugs))]
            eval_data = eval_data.drop(both_unknown_data.index)
        elif method == 'both_unknown':
            eval_data = eval_data[(eval_data['entry1'].isin(selected_drugs))&(eval_data['entry2'].isin(selected_drugs))]
        
        val_data = eval_data
        data_ratio = len(val_data)/len(eval_data)
        selected_drugs = []
        test_data = []
        while data_ratio > 0.4:
            clust_id = np.random.choice(eval_cls)
            drug_ids = drugs[drugs['c_label'] == clust_id]['drugID'].values
            drug_entries = val_data[(val_data['entry1'].isin(drug_ids))|(val_data['entry2'].isin(drug_ids))]
            val_data = val_data[(~val_data['entry1'].isin(drug_ids))&(~val_data['entry2'].isin(drug_ids))]
            test_data.append(drug_entries)
            selected_drugs += list(drug_ids)
            data_ratio = len(val_data)/len(eval_data)
        test_data = pd.concat(test_data)

        print(len(train_data)/(len(train_data)+len(val_data)+len(test_data)))
        print(len(val_data)/(len(train_data)+len(val_data)+len(test_data)))
        print(len(test_data)/(len(train_data)+len(val_data)+len(test_data)))
        return train_data, val_data, test_data