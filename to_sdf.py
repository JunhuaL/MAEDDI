import pandas as pd
from conformers import ConformerGenerator
import sys
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit import Chem
import logging

logger = logging.getLogger(__name__)

data_dir = sys.argv[1]
missing_confs = pd.read_csv(data_dir)
writer = Chem.SDWriter('./dataset/DrugBank/drug/missing_confs.sdf')

entryIDs = missing_confs.drugID
mols_list = list(map(Chem.MolFromSmiles,missing_confs.SMILES))

generator = ConformerGenerator(1,-1,'uff',50)

for i,mol in enumerate(mols_list):
    try:
        conf_mol = generator.generate_conformers(mol)
        conf_mol.SetProp('SOURCE_ID',entryIDs[i])
        writer.write(conf_mol, confId=0)
    except Exception as e:
        logger.warning(
            "Failed to featurize datapoint %d. Appending empty array" % i)
        logger.warning("Exception message: {}".format(e))
