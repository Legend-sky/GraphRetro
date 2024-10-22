import numpy as np
from rdkit import Chem
import networkx as nx
from typing import List, Tuple, Union

from seq_graph_retro.utils.chem import get_sub_mol
from seq_graph_retro.molgraph.mol_features import BOND_TYPES, BOND_FLOATS

class RxnGraph:
    """
    RxnGraph is an abstract class for storing all elements of a reaction, like
    reactants, products and fragments. The edits associated with the reaction
    are also captured in edit labels. One can also use h_labels, which keep track
    of atoms with hydrogen changes. For reactions with multiple edits, a done
    label is also added to account for termination of edits.
    """

    def __init__(self,
                 prod_mol: Chem.Mol,
                 frag_mol: Chem.Mol = None,
                 reac_mol: Chem.Mol = None,
                 edits_to_apply: List = [],
                 rxn_class: int = None) -> None:
        """
        Parameters
        ----------
        prod_mol: Chem.Mol,
            Product molecule
        frag_mol: Chem.Mol, default None
            Fragment molecule(s)
        reac_mol: Chem.Mol, default None
            Reactant molecule(s)
        edits_to_apply: List, default [],
            Edits to apply to the product molecule, captured in edit_/h_labels
        rxn_class: int, default None,
            Reaction class for this reaction.
        """
        self.prod_mol = RxnElement(mol=prod_mol, rxn_class=rxn_class)
        if frag_mol is not None:
            self.frag_mol = MultiElement(mol=frag_mol, rxn_class=rxn_class)
        if reac_mol is not None:
            self.reac_mol = MultiElement(mol=reac_mol, rxn_class=rxn_class)
        self.edits_to_apply = edits_to_apply
        self.edit_label, self.h_label, self.done_label = self._get_labels()
        self.rxn_class = rxn_class

    def _get_labels(self) -> Tuple[np.ndarray]:
        """Returns the different labels associated with the reaction."""
        return None, None, None

    def get_attributes(self,
                       mol_attrs: List = ['prod_mol', 'frag_mol', 'reac_mol'],
                       label_attrs: List = ['edit_label', 'h_label']) -> Tuple:
        """
        Parameters
        ----------
        Returns the different attributes associated with the reaction graph.

        mol_attrs: List,
            Molecule objects to return
        label_attrs: List,
            Label attributes to return. Individual label attrs are coerced into
            a single label
        """
        mol_tuple = ()      #prod_mol,frag_mol
        label_tuple = ()    #edit_label,h_label

        for attr in mol_attrs:
            if hasattr(self, attr):
                mol_tuple += (getattr(self, attr),) #添加prod_mol和frag_mol，这是两个类RxnElement和MultiElement
            else:
                print(f"Does not have attr {attr}")

        for attr in label_attrs:
            if hasattr(self, attr):
                label_tuple += (getattr(self, attr).flatten(), )    #添加edit_label(33*5)和h_label(30*1)并进行一维化为(195*1)

        if len(label_tuple):
            label_tuple = np.concatenate(label_tuple)   #转换为数组，类型变为numpy.ndarray
            new_tuple = mol_tuple + (label_tuple,)      #一共是3个元素RxnElement类和MultiElement类，以及195*1的一维反应中心数据
            return new_tuple

        return mol_tuple

class RxnElement:
    """
    RxnElement is an abstract class for dealing with single molecule. The graph
    and corresponding molecule attributes are built for the molecule. The constructor
    accepts only mol objects, sidestepping the use of SMILES string which may always
    not be achievable, especially for a unkekulizable molecule.
    """

    def __init__(self, mol: Chem.Mol, rxn_class: int = None) -> None:
        """
        Parameters
        ----------
        mol: Chem.Mol,
            Molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        """
        self.mol = mol
        self.rxn_class = rxn_class
        self._build_mol()
        self._build_graph()

    def _build_mol(self) -> None:
        """Builds the molecule attributes."""
        self.num_atoms = self.mol.GetNumAtoms() #原子数量
        self.num_bonds = self.mol.GetNumBonds() #键数量
        self.amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() #map：索引
                            for atom in self.mol.GetAtoms()}
        self.idx_to_amap = {value: key for key, value in self.amap_to_idx.items()}  #索引：map

    def _build_graph(self) -> None: #构建分子的有向图属性
        """Builds the graph attributes."""
        self.G_undir = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol)) #无向图，30个nodes,33条edges
        self.G_dir = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol)) #有向图，30个nodes,66条edges

        for atom in self.mol.GetAtoms():    #设置图的节点（原子）
            self.G_undir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()   #atom.GetSymbol()返回该原子的字符：C，H，O，N，Cl等
            self.G_dir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()

        for bond in self.mol.GetBonds():    #设置的图的边（化学键）
            a1 = bond.GetBeginAtom().GetIdx()   #获取键的开始原子的索引
            a2 = bond.GetEndAtom().GetIdx()     #获取键的结束原子的索引
            btype = BOND_TYPES.index( bond.GetBondType() )  #btype为[0,1,2,3,4],分别表示没有键，单键，双键，三键，苯环键
            self.G_undir[a1][a2]['label'] = btype
            self.G_dir[a1][a2]['label'] = btype
            self.G_dir[a2][a1]['label'] = btype

        self.atom_scope = (0, self.num_atoms)   #(0,30)
        self.bond_scope = (0, self.num_bonds)   #(0,33)

    #CHECK IF THESE TWO ARE NEEDED
    def update_atom_scope(self, offset: int) -> Union[List, Tuple]:
        """Updates the atom indices by the offset.

        Parameters
        ----------
        offset: int,
            要应用的偏移量
        """
        # 检查self.atom_scope是否为列表
        if isinstance(self.atom_scope, list):
            # 如果是列表，对每个范围元组中的起始索引加上偏移量，并返回新列表
            return [(st + offset, le) for (st, le) in self.atom_scope]
        # 如果self.atom_scope是元组，直接对起始索引加上偏移量，并返回新元组
        st, le = self.atom_scope
        return (st + offset, le)

    def update_bond_scope(self, offset: int) -> Union[List, Tuple]:
        """Updates the atom indices by the offset.

        Parameters
        ----------
        offset: int,
            要应用的偏移量
        """
        if isinstance(self.bond_scope, list):
            return [(st + offset, le) for (st, le) in self.bond_scope]
        st, le = self.bond_scope
        return (st + offset, le)


class BondEditsRxn(RxnGraph):

    def _get_labels(self) -> Tuple[np.ndarray]:
        """Returns the different labels associated with the reaction."""
        edit_label = np.zeros((self.prod_mol.num_bonds, len(BOND_FLOATS)))  #编辑标签：BOND_FLOATS为5中键类型，shape为原子数*5
        h_label = np.zeros(self.prod_mol.num_atoms) #氢原子标签：表示该原子的操作是针对一个氢原子的
        done_label = np.zeros((1,)) #完成标签，没有操作了

        if not isinstance(self.edits_to_apply, list):
            edits_to_apply = [self.edits_to_apply]
        else:
            edits_to_apply = self.edits_to_apply

        if len(edits_to_apply) == 0:
            done_label[0] = 1.0
            return edit_label, h_label, done_label

        else:
            for edit in edits_to_apply:
                a1, a2, b1, b2 = edit.split(":")
                a1, a2 = int(a1), int(a2)
                b1, b2 = float(b1), float(b2)

                if a2 == 0:
                    a_start = self.prod_mol.amap_to_idx[a1]
                    h_label[a_start] = 1
                else:
                    #delta = b2 - b1
                    a_start, a_end = self.prod_mol.amap_to_idx[a1], self.prod_mol.amap_to_idx[a2]

                    b_idx = self.prod_mol.mol.GetBondBetweenAtoms(a_start, a_end).GetIdx()
                    edit_label[b_idx][BOND_FLOATS.index(b2)] = 1

        return edit_label, h_label, done_label


class MultiElement(RxnElement):
    """
    MultiElement is an abstract class for dealing with multiple molecules. The graph
    is built with all molecules, but different molecules and their sizes are stored.
    The constructor accepts only mol objects, sidestepping the use of SMILES string
    which may always not be achievable, especially for an invalid intermediates.
    """
    def _build_graph(self) -> None:
        """Builds the graph attributes."""
        self.G_undir = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        self.G_dir = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))

        for atom in self.mol.GetAtoms():
            self.G_undir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()
            self.G_dir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()

        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_TYPES.index( bond.GetBondType() )
            self.G_undir[a1][a2]['label'] = btype
            self.G_dir[a1][a2]['label'] = btype
            self.G_dir[a2][a1]['label'] = btype

        frag_indices = [c for c in nx.strongly_connected_components(self.G_dir)]
        self.mols = [get_sub_mol(self.mol, sub_atoms) for sub_atoms in frag_indices]

        atom_start = 0
        bond_start = 0
        self.atom_scope = []
        self.bond_scope = []

        for mol in self.mols:
            self.atom_scope.append((atom_start, mol.GetNumAtoms()))
            self.bond_scope.append((bond_start, mol.GetNumBonds()))
            atom_start += mol.GetNumAtoms()
            bond_start += mol.GetNumBonds()
