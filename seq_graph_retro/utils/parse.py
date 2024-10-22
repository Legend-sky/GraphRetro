from rdkit import Chem
from typing import List, Dict, Tuple, Set
from collections import namedtuple, deque
ReactionInfo = namedtuple("ReactionInfo", ['rxn_smi', 'core', 'core_edits', 'lg_edits', 'attach_atoms', 'rxn_class'])

from seq_graph_retro.utils.chem import apply_edits_to_mol, get_mol, get_sub_mol
from seq_graph_retro.molgraph.mol_features import BOND_FLOAT_TO_TYPE
from seq_graph_retro.molgraph import MultiElement

def get_bond_info(mol: Chem.Mol) -> Dict:
    """Get information on bonds in the molecule.

    Parameters
    ----------
    mol: Chem.Mol
        Molecule
    """
    if mol is None:
        return {}

    bond_info = {}
    for bond in mol.GetBonds():
        a_start = bond.GetBeginAtom().GetAtomMapNum()
        a_end = bond.GetEndAtom().GetAtomMapNum()

        key_pair = sorted([a_start, a_end])
        bond_info[tuple(key_pair)] = [bond.GetBondTypeAsDouble(), bond.GetIdx()]

    return bond_info

def align_kekule_pairs(r: str, p: str) -> Tuple[Chem.Mol, Chem.Mol]:
    """Aligns kekule pairs to ensure unchanged bonds have same bond order in
    previously aromatic rings.

    Parameters
    ----------
    r: str,
        SMILES string representing the reactants
    p: str,
        SMILES string representing the product
    """
    reac_mol = Chem.MolFromSmiles(r)
    max_amap = max([atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()])
    for atom in reac_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap = max_amap + 1

    prod_mol = Chem.MolFromSmiles(p)

    prod_prev = get_bond_info(prod_mol)
    Chem.Kekulize(prod_mol)
    prod_new = get_bond_info(prod_mol)

    reac_prev = get_bond_info(reac_mol)
    Chem.Kekulize(reac_mol)
    reac_new = get_bond_info(reac_mol)

    for bond in prod_new:
        if bond in reac_new and (prod_prev[bond][0] == reac_prev[bond][0]):
            reac_new[bond][0] = prod_new[bond][0]

    reac_mol = Chem.RWMol(reac_mol)
    amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mol.GetAtoms()}

    for bond in reac_new:
        idx1, idx2 = amap_idx[bond[0]], amap_idx[bond[1]]
        bo = reac_new[bond][0]
        reac_mol.RemoveBond(idx1, idx2)
        reac_mol.AddBond(idx1, idx2, BOND_FLOAT_TO_TYPE[bo])

    return reac_mol.GetMol(), prod_mol

#获取给定反应的反应和编辑中心
def get_reaction_core(r: str, p: str, kekulize: bool = False, use_h_labels: bool = False) -> Tuple[Set, List]:
    reac_mol = get_mol(r)
    prod_mol = get_mol(p)

    if reac_mol is None or prod_mol is None:
        return set(), []

    if kekulize:
        reac_mol, prod_mol = align_kekule_pairs(r, p)
    #得到产物键的信息和原子标号信息
    prod_bonds = get_bond_info(prod_mol)
    p_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in prod_mol.GetAtoms()}

    #这里给反应物中每个原子标上序号，为了更好地获取反应物中键的信息
    max_amap = max([atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()])
    for atom in reac_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap += 1
    #得到反应物键的信息和原子标号信息
    reac_bonds = get_bond_info(reac_mol)
    reac_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mol.GetAtoms()}

    rxn_core = set()
    core_edits = []

    for bond in prod_bonds: #遍历product中的所有键
        #产物中的键在反应物中，并且键的类型不相等，说明该键类型改变了，更新到core_edits和rxn_core中
        if bond in reac_bonds and prod_bonds[bond][0] != reac_bonds[bond][0]:
            a_start, a_end = bond
            prod_bo, reac_bo = prod_bonds[bond][0], reac_bonds[bond][0]

            a_start, a_end = sorted([a_start, a_end])
            edit = f"{a_start}:{a_end}:{prod_bo}:{reac_bo}"
            core_edits.append(edit)
            rxn_core.update([a_start, a_end])
        #产物中的键不在反应物中，说明该键断开了，更新到core_edits和rxn_core中
        if bond not in reac_bonds:
            a_start, a_end = bond
            reac_bo = 0.0
            prod_bo = prod_bonds[bond][0]

            start, end = sorted([a_start, a_end])
            edit = f"{a_start}:{a_end}:{prod_bo}:{reac_bo}"
            core_edits.append(edit)
            rxn_core.update([a_start, a_end])

    for bond in reac_bonds: #遍历反应物的键
        #如果反应物中有键不再产物中，说明是新加入的键
        if bond not in prod_bonds:
            amap1, amap2 = bond
            #此时说明新加的键两边的原子都在产物中，说明是合键，也加入core_edits和rxn_core中
            #这里reactions中多出的键就不需要了，因为不涉及products中的键
            if (amap1 in p_amap_idx) and (amap2 in p_amap_idx):
                a_start, a_end = sorted([amap1, amap2])
                reac_bo = reac_bonds[bond][0]
                edit = f"{a_start}:{a_end}:{0.0}:{reac_bo}"
                core_edits.append(edit)
                rxn_core.update([a_start, a_end])

    if use_h_labels:    #是否使用h_labels,这里是使用
        if len(rxn_core) == 0:
            for atom in prod_mol.GetAtoms():
                amap_num = atom.GetAtomMapNum()
                #计算产物和反应物这个atom的氢原子数量
                numHs_prod = atom.GetTotalNumHs()
                numHs_reac = reac_mol.GetAtomWithIdx(reac_amap[amap_num]).GetTotalNumHs()
                #如果这个atom的氢原子数量不一致，则加入core_edits和rxn_core中
                if numHs_prod != numHs_reac:
                    edit = f"{amap_num}:{0}:{1.0}:{0.0}"
                    core_edits.append(edit)
                    rxn_core.add(amap_num)

    return rxn_core, core_edits

def get_reaction_info(rxn_smi: str, kekulize: bool = False, use_h_labels: bool = False,
                      rxn_class: int = None) -> ReactionInfo:
    """
    Construct a ReactionInfo namedtuple for each reaction. ReactionInfo
    contains information on the reaction core, core edits, added leaving groups
    and attaching atoms.

    Parameters
    ----------
    rxn_smi: str,
        SMILES string representing the reaction
    kekulize: bool, default False
        Whether to kekulize molecules to fetch minimal set of edits
    use_h_labels: bool, default False
        Whether to use change in hydrogen counts in edits
    rxn_class: int, default None
        Reaction class for given reaction
    """
    r, p = rxn_smi.split(">>")
    reac_mol = get_mol(r)
    prod_mol = get_mol(p)

    if reac_mol is None or prod_mol is None:
        return None
    #得到rxn_core, core_edits
    rxn_core, core_edits = get_reaction_core(r, p, kekulize=kekulize, use_h_labels=use_h_labels)
    #得到产物键的信息和原子标号信息
    prod_bonds = get_bond_info(prod_mol)
    p_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in prod_mol.GetAtoms()}
    #这里给反应物中每个原子标上序号，为了更好地获取反应物的信息
    max_amap = max([atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()])
    for atom in reac_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap += 1

    reac_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mol.GetAtoms()}

    #储存反应物和产物具有相同原子编号的原子，但这些原子没有参与反应，可以是离去基团
    visited = set([atom.GetIdx() for atom in reac_mol.GetAtoms()
                   if (atom.GetAtomMapNum() in p_amap_idx) and (atom.GetAtomMapNum() not in rxn_core)])

    lg_edits = []
    #lg_edit：找到反应物中参与反应物的原子的邻居参与反应的原子键信息。加入到lg_edit中
    for atom in rxn_core:
        root = reac_mol.GetAtomWithIdx(reac_amap[atom]) #获取反应原子
        queue = deque([root])

        while len(queue) > 0:   #BFS，以反应节点出发进行BFS
            x = queue.popleft()
            neis = x.GetNeighbors() #获取原子附近的邻居原子，并按原子索引进行排序
            neis = list(sorted(neis, key=lambda x: x.GetIdx()))

            for y in neis:  #遍历邻居原子
                y_idx = y.GetIdx()
                if y_idx in visited:    #如果邻居原子在没有参与反应的列表中，则跳过
                    continue
                #frontier 是当前队列中的原子列表，用于检查新发现的邻居原子 y 是否已经与队列中的任何原子通过化学键连接。
                frontier = [x] + [a for a in list(queue)]
                y_neis = set([z.GetIdx() for z in y.GetNeighbors()])
                #循环遍历 frontier 中的每个原子 z，并检查它是否与 y 通过化学键连接
                for i,z in enumerate(frontier):
                    if z.GetIdx() in y_neis:
                        bo = reac_mol.GetBondBetweenAtoms(z.GetIdx(), y_idx).GetBondTypeAsDouble()
                        amap1, amap2 = sorted([y.GetAtomMapNum(), z.GetAtomMapNum()])
                        bond = (amap1, amap2)

                        # The two checks are needed because the reaction core is not visited during BFS
                        #如果 z 和 y 之间存在化学键，并且这个化学键的类型与产物中的化学键类型相同，则跳过
                        if bond in prod_bonds and (prod_bonds[bond][0] == bo):
                            continue
                        #如果 z 和 y 之间的化学键在产物中存在但类型不同，或者不存在，则创建一个编辑字符串 edit，表示这个化学键的变化
                        #bo表示反应物中该键的类型
                        elif (bond in prod_bonds) and (prod_bonds[bond][0] != bo) :
                            bo_old = prod_bonds[bond][0]
                            edit = f"{amap1}:{amap2}:{bo_old}:{bo}"

                        else:
                            edit = f"{amap1}:{amap2}:{0.0}:{bo}"

                        if edit not in lg_edits and edit not in core_edits:
                            lg_edits.append(edit)

                visited.add(y_idx)
                queue.append(y)

    r_new, p_new = Chem.MolToSmiles(reac_mol), Chem.MolToSmiles(prod_mol)
    rxn_smi_new = r_new + ">>" + p_new
    attach_atoms = get_attach_atoms(rxn_smi=rxn_smi, core_edits=core_edits, core=rxn_core)
    reaction_info = ReactionInfo(rxn_smi=rxn_smi_new, core=rxn_core,
                                 core_edits=core_edits, lg_edits=lg_edits,
                                 attach_atoms=attach_atoms, rxn_class=rxn_class)
    return reaction_info

def get_attach_atoms(rxn_smi: str, core_edits: List[str], core: Set[int]) -> List[List]:
    """Gather attaching atoms for every fragment-reactant pair.

    Parameters
    ----------
    rxn_smi: str,
        SMILES string representing the reaction
    core_edits: List[str],
        Edits applied to product to obtain fragments
    core: Set[int],
        Atom maps of participating atoms in the reaction core.
    """
    r, p = rxn_smi.split(">>")

    reactants = Chem.MolFromSmiles(r)
    products = Chem.MolFromSmiles(p)
    fragments = apply_edits_to_mol(products, core_edits)

    prod_amaps = {atom.GetAtomMapNum() for atom in products.GetAtoms()}
    frag_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in fragments.GetAtoms()}

    reac_mols = MultiElement(Chem.Mol(reactants)).mols
    frag_mols = MultiElement(Chem.Mol(fragments)).mols

    reac_mols, frag_mols = map_reac_and_frag(reac_mols, frag_mols)

    attach_list = []

    for mol in reac_mols:
        core_atoms = {atom for atom in mol.GetAtoms() if atom.GetAtomMapNum() in core}
        attach_atoms = []

        for atom in core_atoms:
            can_attach = any([nei.GetAtomMapNum() not in prod_amaps for nei in atom.GetNeighbors()])
            if can_attach:
                attach_atoms.append(frag_amap_idx[atom.GetAtomMapNum()])

        attach_list.append(attach_atoms)
    assert len(attach_list) == len(frag_mols)

    return attach_list

def map_reac_and_frag(reac_mols: List[Chem.Mol], frag_mols: List[Chem.Mol]) -> Tuple[List[Chem.Mol]]:
    """Aligns reactant and fragment mols by computing atom map overlaps.

    Parameters
    ----------
    reac_mols: List[Chem.Mol],
        List of reactant mols
    frag_mols: List[Chem.Mol],
        List of fragment mols
    """
    if len(reac_mols) != len(frag_mols):
        return reac_mols, frag_mols
    reac_maps = [[atom.GetAtomMapNum() for atom in mol.GetAtoms()] for mol in reac_mols]
    frag_maps = [[atom.GetAtomMapNum() for atom in mol.GetAtoms()] for mol in frag_mols]

    overlaps = {i: [] for i in range(len(frag_mols))}
    for i, fmap in enumerate(frag_maps):
        overlaps[i].extend([len(set(fmap).intersection(set(rmap))) for rmap in reac_maps])
        overlaps[i] = overlaps[i].index(max(overlaps[i]))

    new_frag = [Chem.Mol(mol) for mol in frag_mols]
    new_reac = [Chem.Mol(reac_mols[overlaps[i]]) for i in overlaps]
    return new_reac, new_frag


def extract_leaving_groups(mol_list: List[Tuple[Chem.Mol]]) -> Tuple[Dict, List[str], List[Chem.Mol]]:
    """Extracts leaving groups from a product-fragment-reactant tuple.

    Parameters
    ----------
    mol_list: List[Tuple[Chem.Mol]]
        List of product-fragment-reactant tuples
    """
    leaving_groups = ["<bos>", "<eos>", "<unk>", "<pad>"]
    lg_mols = []
    lg_labels_all = []

    for mol_tuple in mol_list:
        p_mol, reac_mols, frag_mols = mol_tuple

        reac_mols, frag_mols = map_reac_and_frag(reac_mols, frag_mols)

        #将多个分子对象合并成一个单一分子对象的过程
        r_mol = Chem.Mol()
        for mol in reac_mols:
            r_mol = Chem.CombineMols(r_mol, Chem.Mol(mol))
        #遍历分子 p_mol 中的所有原子，并为每个原子设置显式氢原子的数量为0
        for atom in p_mol.GetAtoms():
            atom.SetNumExplicitHs(0)

        p_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in p_mol.GetAtoms()}
        r_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in r_mol.GetAtoms()}

        labels = []
        for i, mol in enumerate(reac_mols): #遍历每个部分的反应物
            idxs = []
            attach_amaps = []

            for atom in mol.GetAtoms():
                amap = atom.GetAtomMapNum()
                if amap not in p_amap_idx and amap in r_amap_idx:
                    idxs.append(r_amap_idx[amap])
                    nei_amaps = [nei.GetAtomMapNum() for nei in atom.GetNeighbors()]
                    if any(prod_map in nei_amaps for prod_map in p_amap_idx):
                        attach_amaps.append(amap)

            if len(idxs):
                lg_mol = get_sub_mol(r_mol, idxs)
                for atom in lg_mol.GetAtoms():
                    if atom.GetAtomMapNum() in attach_amaps:
                        atom.SetAtomMapNum(1)
                    else:
                        atom.SetAtomMapNum(0)

                lg = Chem.MolToSmiles(lg_mol)

                if lg not in leaving_groups:
                    leaving_groups.append(lg)
                    lg_mols.append(lg_mol)
                labels.append(lg)
            else:
                labels.append("<eos>")

        lg_labels_all.append(labels)

    lg_dict = {lg: idx for idx, lg in enumerate(leaving_groups)}
    return lg_dict, lg_labels_all, lg_mols
