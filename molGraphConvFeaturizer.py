
'''
modify from 
https://github.com/deepchem/deepchem/blob/master/deepchem/feat/molecule_featurizers/mol_graph_conv_featurizer.py#L1-L233 
'''

from typing import List, Tuple
import numpy as np

from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit import Chem
from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.graph_data import GraphData
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.molecule_feature_utils import get_atom_type_one_hot
from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info
from deepchem.utils.molecule_feature_utils import get_atom_hydrogen_bonding_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_chirality_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge
from deepchem.utils.molecule_feature_utils import get_atom_partial_charge
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_type_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_in_same_ring_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_conjugated_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_stereo_one_hot
from conformers import ConformerGenerator
# from deepchem.utils.molecule_feature_utils import get_atom_formal_charge_one_hot
# from deepchem.utils.molecule_feature_utils import get_atom_implicit_valence_one_hot
# from deepchem.utils.molecule_feature_utils import get_atom_explicit_valence_one_hot
# from deepchem.utils.rdkit_utils import compute_all_pairs_shortest_path
# from deepchem.utils.rdkit_utils import compute_pairwise_ring_info

DEFAULT_ATOM_TYPE_SET = [
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]
DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3"]
DEFAULT_TOTAL_NUM_Hs_SET = [0, 1, 2, 3, 4]
DEFAULT_FORMAL_CHARGE_SET = [-2, -1, 0, 1, 2]
DEFAULT_TOTAL_DEGREE_SET = [0, 1, 2, 3, 4, 5]
DEFAULT_RING_SIZE_SET = [3, 4, 5, 6, 7, 8]
DEFAULT_BOND_TYPE_SET = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
DEFAULT_BOND_STEREO_SET = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
DEFAULT_GRAPH_DISTANCE_SET = [1, 2, 3, 4, 5, 6, 7]
DEFAULT_ATOM_IMPLICIT_VALENCE_SET = [0, 1, 2, 3, 4, 5, 6]
DEFAULT_ATOM_EXPLICIT_VALENCE_SET = [1, 2, 3, 4, 5, 6]

USER_ATOM_TYPE_SET  = ['Fe','Sb','N','Ca','Mo','Lu','Mg','I','La',
                       'Ra','Ac','Ce','Bi','Nd','Be','Sn','Se','Xe',
                       'Ti','Ne','S','C','Hg','Cl','Y','Tl','Zn','Zr',
                       'Ag','Cd','Na','O','Si','W','Ga','Tc','Cs','Mn',
                       'He','Pb','Al','Ru','Au','Pt','Ta','Re','Sm','Kr',
                       'Gd','As','Te','Cr','Nb','H','In','Pd','Co','F',
                       'B','Cu','Li','Sr','Ni','Br','P','Rb','V','Ho',
                       'Bk','Ba','K']
USER_TOTAL_DEGREE_SET = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
USER_HYBRIDIZATION_SET = ["SP", "SP2", "SP3", 'SP3D','SP3D2']
def get_atom_implicit_valence_one_hot(
    atom: RDKitAtom,
    allowable_set: List[int] = DEFAULT_ATOM_IMPLICIT_VALENCE_SET,
    include_unknown_set: bool = True) -> List[float]:
  """Get an one-hot feature of implicit valence of an atom.
  Parameters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
  allowable_set: List[int]
    Atom implicit valence to consider. The default set is `[0, 1, ..., 6]`
  include_unknown_set: bool, default True
    If true, the index of all types not in `allowable_set` is `len(allowable_set)`.
  Returns
  -------
  List[float]
    A one-hot vector of implicit valence an atom has.
    If `include_unknown_set` is False, the length is `len(allowable_set)`.
    If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
  """
  return one_hot_encode(atom.GetImplicitValence(), allowable_set,
                        include_unknown_set)
def get_atom_explicit_valence_one_hot(
    atom: RDKitAtom,
    allowable_set: List[int] = DEFAULT_ATOM_EXPLICIT_VALENCE_SET,
    include_unknown_set: bool = True) -> List[float]:
  """Get an one-hot feature of explicit valence of an atom.
  Parameters
  ---------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
  allowable_set: List[int]
    Atom explicit valence to consider. The default set is `[1, ..., 6]`
  include_unknown_set: bool, default True
    If true, the index of all types not in `allowable_set` is `len(allowable_set)`.
  Returns
  -------
  List[float]
    A one-hot vector of explicit valence an atom has.
    If `include_unknown_set` is False, the length is `len(allowable_set)`.
    If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
  """
  return one_hot_encode(atom.GetExplicitValence(), allowable_set,
                        include_unknown_set)


def _construct_atom_feature(
    atom: RDKitAtom, h_bond_infos: List[Tuple[int, str]], use_chirality: bool,
    use_partial_charge: bool) -> np.ndarray:
  """Construct an atom feature from a RDKit atom object.
  Parameters
  ----------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
  h_bond_infos: List[Tuple[int, str]]
    A list of tuple `(atom_index, hydrogen_bonding_type)`.
    Basically, it is expected that this value is the return value of
    `construct_hydrogen_bonding_info`. The `hydrogen_bonding_type`
    value is "Acceptor" or "Donor".
  use_chirality: bool
    Whether to use chirality information or not.
  use_partial_charge: bool
    Whether to use partial charge data or not.
  Returns
  -------
  np.ndarray
    A one-hot vector of the atom feature.
    44+1+5+2+1+12+6+8+7+1+1+2+1 = 91 features
  """
  atom_type = get_atom_type_one_hot(atom,USER_ATOM_TYPE_SET,include_unknown_set = True)
  formal_charge = get_atom_formal_charge(atom)
  hybridization = get_atom_hybridization_one_hot(atom,USER_HYBRIDIZATION_SET,include_unknown_set = False)
  acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
  aromatic = get_atom_is_in_aromatic_one_hot(atom)
  degree = get_atom_total_degree_one_hot(atom,USER_TOTAL_DEGREE_SET,include_unknown_set = True)
  total_num_Hs = get_atom_total_num_Hs_one_hot(atom,DEFAULT_TOTAL_NUM_Hs_SET,include_unknown_set = True)
  atom_feat = np.concatenate([
      atom_type, formal_charge, hybridization, acceptor_donor, aromatic, degree,
      total_num_Hs
  ])

  ### user additional features ####
  if True:
    imp_valence = get_atom_implicit_valence_one_hot(atom,DEFAULT_ATOM_IMPLICIT_VALENCE_SET,include_unknown_set=True)
    exp_valence = get_atom_explicit_valence_one_hot(atom,DEFAULT_ATOM_EXPLICIT_VALENCE_SET,include_unknown_set=True)
    atom_feat = np.concatenate([atom_feat,imp_valence,exp_valence,[atom.HasProp('_ChiralityPossible'), atom.GetNumRadicalElectrons()],])
  ###########    END    ############

  if use_chirality:
    # chirality = get_atom_chirality_one_hot(atom) 
    chirality = get_atom_chirality_one_hot(atom) 
    atom_feat = np.concatenate([atom_feat, np.array(chirality)])

  if use_partial_charge:
    partial_charge = get_atom_partial_charge(atom)
    atom_feat = np.concatenate([atom_feat, np.array(partial_charge)])
  return atom_feat


def _construct_bond_feature(bond: RDKitBond) -> np.ndarray:
  """Construct a bond feature from a RDKit bond object.
  Parameters
  ---------
  bond: rdkit.Chem.rdchem.Bond
    RDKit bond object
  Returns
  -------
  np.ndarray
    A one-hot vector of the bond feature.
  """
  bond_type = get_bond_type_one_hot(bond)
  same_ring = get_bond_is_in_same_ring_one_hot(bond)
  conjugated = get_bond_is_conjugated_one_hot(bond)
  stereo = get_bond_stereo_one_hot(bond)
  return np.concatenate([bond_type, same_ring, conjugated, stereo])

def edge_adjacency_matrix(mol: RDKitMol) -> np.ndarray:
  bonds = mol.GetBonds()
  n = len(bonds)
  edge_mat = np.zeros((n,n))
  for i in range(n):
    for j in range(i+1,n):
      edge_i = [bonds[i].GetBeginAtomIdx(),bonds[i].GetEndAtomIdx()]
      edge_j = [bonds[j].GetBeginAtomIdx(),bonds[j].GetEndAtomIdx()]
      if any([atom in edge_j for atom in edge_i]):
        edge_mat[i][j] = 1
        edge_mat[j][i] = 1
  return edge_mat

class MolGraphConvFeaturizer(MolecularFeaturizer):
  """This class is a featurizer of general graph convolution networks for molecules.
  The default node(atom) and edge(bond) representations are based on
  `WeaveNet paper <https://arxiv.org/abs/1603.00856>`_. If you want to use your own representations,
  you could use this class as a guide to define your original Featurizer. In many cases, it's enough
  to modify return values of `construct_atom_feature` or `construct_bond_feature`.
  The default node representation are constructed by concatenating the following values,
  and the feature length is 30.
  - Atom type: A one-hot vector of this atom, "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other atoms".
  - Formal charge: Integer electronic charge.
  - Hybridization: A one-hot vector of "sp", "sp2", "sp3".
  - Hydrogen bonding: A one-hot vector of whether this atom is a hydrogen bond donor or acceptor.
  - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
  - Degree: A one-hot vector of the degree (0-5) of this atom.
  - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
  - Chirality: A one-hot vector of the chirality, "R" or "S". (Optional)
  - Partial charge: Calculated partial charge. (Optional)
  The default edge representation are constructed by concatenating the following values,
  and the feature length is 11.
  - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
  - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
  - Conjugated: A one-hot vector of whether this bond is conjugated or not.
  - Stereo: A one-hot vector of the stereo configuration of a bond.
  If you want to know more details about features, please check the paper [1]_ and
  utilities in deepchem.utils.molecule_feature_utils.py.
  Examples
  --------
  >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
  >>> featurizer = MolGraphConvFeaturizer(use_edges=True)
  >>> out = featurizer.featurize(smiles)
  >>> type(out[0])
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> out[0].num_node_features
  30
  >>> out[0].num_edge_features
  11
  References
  ----------
  .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
     Journal of computer-aided molecular design 30.8 (2016):595-608.
  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self,
               use_edges: bool = False,
               use_chirality: bool = False,
               use_partial_charge: bool = False):
    """
    Parameters
    ----------
    use_edges: bool, default False
      Whether to use edge features or not.
    use_chirality: bool, default False
      Whether to use chirality information or not.
      If True, featurization becomes slow.
    use_partial_charge: bool, default False
      Whether to use partial charge data or not.
      If True, this featurizer computes gasteiger charges.
      Therefore, there is a possibility to fail to featurize for some molecules
      and featurization becomes slow.
    """
    self.use_edges = use_edges
    self.use_partial_charge = use_partial_charge
    self.use_chirality = use_chirality

  def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
    """Calculate molecule graph features from RDKit mol object.
    Parameters
    ----------
    datapoint: rdkit.Chem.rdchem.Mol
      RDKit mol object.
    Returns
    -------
    graph: GraphData
      A molecule graph with some features.
    """
    assert datapoint.GetNumAtoms(
    ) > 1, "More than one atom should be present in the molecule for this featurizer to work."
    if 'mol' in kwargs:
      datapoint = kwargs.get("mol")
      raise DeprecationWarning(
          'Mol is being phased out as a parameter, please pass "datapoint" instead.'
      )

    if self.use_partial_charge:
      try:
        datapoint.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
      except:
        # If partial charges were not computed
        try:
          from rdkit.Chem import AllChem
          AllChem.ComputeGasteigerCharges(datapoint)
        except ModuleNotFoundError:
          raise ImportError("This class requires RDKit to be installed.")

    # construct atom (node) feature
    Chem.rdPartialCharges.ComputeGasteigerCharges(datapoint)
    h_bond_infos = construct_hydrogen_bonding_info(datapoint)
    atom_features = np.asarray(
        [
            _construct_atom_feature(atom, h_bond_infos, self.use_chirality,
                                    self.use_partial_charge)
            for atom in datapoint.GetAtoms()
        ],
        dtype=float,
    )

    # construct edge (bond) index
    src, dest = [], []
    for bond in datapoint.GetBonds():
      # add edge list considering a directed graph
      start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
      src += [start, end]
      dest += [end, start]

    # construct edge (bond) feature
    bond_features = None  # deafult None
    if self.use_edges:
      features = []
      for bond in datapoint.GetBonds():
        features += 2 * [_construct_bond_feature(bond)]
      bond_features = np.asarray(features, dtype=float)
    else:
      bond_features = None

    return GraphData(
        node_features=atom_features,
        edge_index=np.asarray([src, dest], dtype=int),
        edge_features=bond_features)

class BondGraphFeaturizer(MolecularFeaturizer):
  """ This class is a featurizer for the bonds of a molecule. The nodes are bonds and the edges
  between the nodes describe the geometric relationship between the bonds, e.g angles, adjacency.
  """
  def __init__(self, use_dihedrals, use_edges, n_confs):
    self.n_confs = n_confs
    self.use_dihedrals = use_dihedrals
    self.use_edges = use_edges
    self.conf_gen = ConformerGenerator(n_confs,-1,'mmff94s',50)
  
  def _featurize(self, datapoint: RDKitMol, **kwargs):
    assert datapoint.GetNumAtoms(
    ) > 1, "More than one atom should be present in the molecule for this featurizer to work."
    if 'mol' in kwargs:
      datapoint = kwargs.get("mol")
      raise DeprecationWarning(
          'Mol is being phased out as a parameter, please pass "datapoint" instead.'
      )
    
    # Generating Conformers
    bonds = datapoint.GetBonds()
    conf_mol = self.conf_gen.generate_conformers(datapoint)
    confs = conf_mol.GetConformers()

    # construct node (bond) features
    bond_features = []
    for bond in bonds:
      features = _construct_bond_feature(bond)
      bond_lens = np.zeros(self.n_confs)
      for i in range(len(confs)):
        bond_lens[i] = rdMolTransforms.GetBondLength(confs[i],bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())
      features = np.concatenate([features,bond_lens],axis=None)
      bond_features.append(features)

    bond_features = np.asarray(bond_features,dtype=float)

    if self.use_dihedrals:
      pass
    else:
      src, dest = [], []
      edge_features = None
      angle_features = []
      for i_idx in range(len(bonds)):
        for j_idx in range(i_idx+1,len(bonds)):
          bond_i = [bonds[i_idx].GetBeginAtomIdx(),bonds[i_idx].GetEndAtomIdx()]
          bond_j = [bonds[j_idx].GetBeginAtomIdx(),bonds[j_idx].GetEndAtomIdx()]
          
          if any([atom in bond_j for atom in bond_i]):
            src += [i_idx,j_idx]
            dest += [j_idx,i_idx]
            if self.use_edges:
              common_atom = list(set(bond_i).intersection(set(bond_j)))
              end_atoms = list(set(bond_i).symmetric_difference(set(bond_j)))
              angles = np.zeros(self.n_confs)
              for i in range(len(confs)):
                angles[i] = rdMolTransforms.GetAngleRad(confs[i],end_atoms[0],common_atom[0],end_atoms[1])
              angle_features += 2*[angles]

      edge_features = np.asarray(angle_features, dtype=float)
    return GraphData(node_features=bond_features,
                    edge_index=np.asarray([src, dest], dtype=int),
                    edge_features=edge_features)

def discretize(values,bins,upper):
    step_size = upper//bins
    ohe_feats = np.digitize(values,range(step_size,upper,step_size))
    ohe = np.eye(bins)[ohe_feats]
    return ohe

class ConfGraphFeaturizer(MolecularFeaturizer):
  """ This class is a featurizer for the bonds of a molecule. The nodes are bonds and the edges
  between the nodes describe the geometric relationship between the bonds, e.g angles, adjacency.
  """
  def __init__(self,n_bins):
    self.n_bins = n_bins

  def _featurize(self, datapoint: RDKitMol, **kwargs):
    assert datapoint.GetNumAtoms(
    ) > 1, "More than one atom should be present in the molecule for this featurizer to work."
    if 'mol' in kwargs:
      datapoint = kwargs.get("mol")
      raise DeprecationWarning(
          'Mol is being phased out as a parameter, please pass "datapoint" instead.'
      )
    
    # Generating Conformers
    bonds = datapoint.GetBonds()
    conf = datapoint.GetConformer()

    # construct node (bond) features
    bond_features = []
    for bond in bonds:
      features = _construct_bond_feature(bond)
      bond_len = rdMolTransforms.GetBondLength(conf,bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())
      features = np.concatenate([features,bond_len],axis=None)
      bond_features.append(features)

    bond_features = np.asarray(bond_features,dtype=float)

    src, dest = [], []
    angle_features = []
    for i in range(len(bonds)):
      for j in range(i+1,len(bonds)):
        bond_i = [bonds[i].GetBeginAtomIdx(),bonds[i].GetEndAtomIdx()]
        bond_j = [bonds[j].GetBeginAtomIdx(),bonds[j].GetEndAtomIdx()]
        
        common_atom = set(bond_i).intersection(set(bond_j))
        if common_atom:
          src += [i,j]
          dest += [j,i]
          common_atom = list(common_atom)[0]
          end_atoms = list(set(bond_i).symmetric_difference(set(bond_j)))
          angle = rdMolTransforms.GetAngleRad(conf,end_atoms[0],common_atom,end_atoms[1])
          angle_features += 2*[angle]

    edge_features = discretize(angle_features,self.n_bins,180)
    #edge_features = np.asarray(angle_features, dtype=float)

    return GraphData(node_features=bond_features,
                    edge_index=np.asarray([src, dest], dtype=int),
                    edge_features=edge_features)

if __name__  == '__main__':
    mols = ['COC1=C(C=C2C(=C1)CCN=C2C3=CC(=C(C=C3)Cl)Cl)Cl',
            'COC(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)[C@@H](O)CN(Cc1ccc(-c2ccccn2)cc1)NC(=O)[C@@H](NC(=O)OC)C(C)(C)C)C(C)(C)C',
            'C[N+](C)(C)CCOC(N)=O']

    graph_featurizer = MolGraphConvFeaturizer(use_edges=False,use_chirality=True,use_partial_charge=True)
    bond_featurizer = BondGraphFeaturizer(use_edges=True,use_dihedrals=False,n_confs=4)
    graph_mols = graph_featurizer.featurize(mols)
    graph_bonds = bond_featurizer.featurize(mols)

    print('node_features',graph_mols[2].node_features.shape)
    #print('edge_features',graph_mols[0].edge_features.shape)
    print('edge_index',graph_mols[2].edge_index.shape)

    for graph in graph_bonds:
      print('node_features',graph.node_features.shape)
      print('edge_features',graph.edge_features.shape)
      print('edge_index',graph.edge_index.shape)