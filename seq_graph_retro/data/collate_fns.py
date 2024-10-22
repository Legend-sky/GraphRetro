import torch
from rdkit import Chem
import networkx as nx

from seq_graph_retro.molgraph.mol_features import get_atom_features, get_bond_features
from seq_graph_retro.molgraph.mol_features import BOND_FDIM, ATOM_FDIM, BOND_TYPES
from seq_graph_retro.utils.torch import create_pad_tensor

from typing import Any, List, Dict, Tuple

def prepare_lg_labels(lg_dict: Dict, lg_data: List) -> torch.Tensor:
    """Prepare leaving group tensors.

    Parameters
    ----------
    lg_dict: Dict
        Dictionary containing leaving groups to indices map
    lg_data: List
        List of lists containing the leaving groups
    """
    pad_idx, unk_idx = lg_dict["<pad>"], lg_dict["<unk>"]   #获取离去基团的pad和unk索引
    lg_labels = [[lg_dict.get(lg_group, unk_idx) for lg_group in labels] for labels in lg_data]  #将离去基团的列表转换为索引列表

    lengths = [len(lg) for lg in lg_labels]
    labels = torch.full(size=(len(lg_labels), max(lengths)), fill_value=pad_idx, dtype=torch.long)
    for i, lgs in enumerate(lg_labels):
        labels[i, :len(lgs)] = torch.tensor(lgs)
    #labels是一个 PyTorch 张量，包含了所有离去基团的索引。张量的形状是 (N, max_length)，其中 N 是 lg_data 中列表的数量，max_length 是所有列表中最长的长度。
    # 如果某个反应的离开基团数量少于 max_length，则使用 pad_idx 来填充剩余的位置。
    #lengths是一个列表，包含了每个反应的离去基团的实际数量。
    return labels, lengths

def pack_graph_feats(graph_batch: List[Any], directed: bool, use_rxn_class: bool = False,
                     return_graphs: bool = False) -> Tuple[torch.Tensor, List[Tuple[int]]]:
    """Prepare graph tensors.
    Parameters
    ----------
    graph_batch: List[Any],
        Batch of graph objects. Should have attributes G_dir, G_undir
    directed: bool,
        Whether to prepare tensors for directed message passing
    use_rxn_class: bool, default False,
        Whether to use reaction class as additional input
    return_graphs: bool, default False,
        Whether to return the graphs
    """
    if directed:    #有向图，这里为True
        fnode = [get_atom_features(Chem.Atom("*"), use_rxn_class=use_rxn_class, rxn_class=0)]   #获取原子特征的向量,98*1
        fmess = [[0,0] + [0] * BOND_FDIM]   #获取键特征的向量,8*1
        agraph, bgraph = [[]], [[]]  #原子图和键图
        atoms_in_bonds = [[]]   #原子键的列表

        atom_scope, bond_scope = [], [] #原子范围和键范围
        edge_dict = {}  #边的字典
        all_G = []  #所有图

        for bid, graph in enumerate(graph_batch):   #这里bid为0，graph为RxnElement对象
            mol = graph.mol  #获取RxnElement对象的mol属性
            assert mol.GetNumAtoms() == len(graph.G_dir)
            atom_offset = len(fnode)    #原子偏移，1
            bond_offset = len(atoms_in_bonds)   #键偏移,1

            bond_to_tuple = {bond.GetIdx(): tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
                             for bond in mol.GetBonds()}
            tuple_to_bond = {val: key for key, val in bond_to_tuple.items()}    #键的索引和键的元组的映射,形式为:{(0,1):0,(1,2):1,...}

            atom_scope.append(graph.update_atom_scope(atom_offset)) #[(1,30)]，第一维加上了98的偏移
            bond_scope.append(graph.update_bond_scope(bond_offset)) #[(1,33)]

            G = nx.convert_node_labels_to_integers(graph.G_dir, first_label=atom_offset)    #将有向图的节点标签转换为整数
            all_G.append(G)   #将有向图添加到all_G中
            fnode.extend( [None for v in G.nodes] ) #往后添加了G的节点数个None，也就是30个None

            for v, attr in G.nodes(data='label'):   #v是原子序号，attr是原子标签，这里是C，H，O，N，Cl等
                G.nodes[v]['batch_id'] = bid    #为0
                fnode[v] = get_atom_features(mol.GetAtomWithIdx(v-atom_offset),
                                             use_rxn_class=use_rxn_class,
                                             rxn_class=graph.rxn_class) #每个原子都有一个98*1的特征向量,最终变为98*31的特征向量
                agraph.append([])

            bond_comp = [None for _ in range(mol.GetNumBonds())]    #键的列表，1*33
            for u, v, attr in G.edges(data='label'):    #u,v是原子序号，attr是键标签
                bond_feat = get_bond_features(mol.GetBondBetweenAtoms(u-atom_offset, v-atom_offset)).tolist()

                bond = sorted([u, v])
                mess_vec = [u, v] + bond_feat
                if [v, u] not in bond_comp: #如果[v,u]不在键的列表中,键是两个原子之间的连接，它不区分方向，即键 [v, u] 和 [u, v] 表示的是同一条键
                    idx_to_add = tuple_to_bond[(u-atom_offset, v-atom_offset)]  #获取键的索引
                    bond_comp[idx_to_add] = [u, v]  #将键的两个原子添加到键的列表中

                fmess.append(mess_vec)  #每个键都有一个8*1的特征向量，最终变成8*67的特征向量
                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid
                agraph[v].append(eid)   #将键的索引添加到原子的邻接表中
                bgraph.append([])
            atoms_in_bonds.extend(bond_comp)

            for u, v in G.edges:
                eid = edge_dict[(u, v)]
                for w in G.predecessors(u):  #获取键的前驱节点
                    if w == v: continue
                    bgraph[eid].append( edge_dict[(w, u)] )  #将键的前驱节点添加到键的邻接表中

        fnode = torch.tensor(fnode, dtype=torch.float)  #每个原子都有一个98*1的特征向量,最终变为98*31的特征向量
        fmess = torch.tensor(fmess, dtype=torch.float)  #每个键都有一个8*1的特征向量，最终变成8*67的特征向量
        atoms_in_bonds = create_pad_tensor(atoms_in_bonds).long()   #33条键，每条键有两个原子，再加一条初始[0,0],所以是34*2
        agraph = create_pad_tensor(agraph)      #30个原子，每个有多少个邻接原子，不够的按最大的补0，再加一个初始，所以是31*3
        bgraph = create_pad_tensor(bgraph)      #66个键，每个键有多少个邻接键，不够的按最大的补0，再加一个初始，所以是67*3

        graph_tensors = (fnode, fmess, agraph, bgraph, atoms_in_bonds)
        scopes = (atom_scope, bond_scope)   #atom_scope:[(1,30)]，第一维加上了98的偏移；bond_scope:[(1,33)]

        if return_graphs:   #False
            return graph_tensors, scopes, nx.union_all(all_G)
        else:
            return graph_tensors, scopes

    else:
        afeat = [get_atom_features(Chem.Atom("*"), use_rxn_class=use_rxn_class, rxn_class=0)]
        bfeat = [[0] * BOND_FDIM]
        atoms_in_bonds = [[]]
        agraph, bgraph = [[]], [[]]
        atom_scope = []
        bond_scope = []
        edge_dict = {}
        all_G = []

        for bid, graph in enumerate(graph_batch):
            mol = graph.mol
            assert mol.GetNumAtoms() == len(graph.G_undir)
            atom_offset = len(afeat)
            bond_offset = len(bfeat)

            atom_scope.append(graph.update_atom_scope(atom_offset))
            bond_scope.append(graph.update_bond_scope(bond_offset))

            G = nx.convert_node_labels_to_integers(graph.G_undir, first_label=atom_offset)
            all_G.append(G)
            afeat.extend( [None for v in G.nodes] )

            for v, attr in G.nodes(data='label'):
                G.nodes[v]['batch_id'] = bid
                afeat[v] = get_atom_features(mol.GetAtomWithIdx(v-atom_offset),
                                             use_rxn_class=use_rxn_class,
                                             rxn_class=graph.rxn_class)
                agraph.append([])
                bgraph.append([])

            for u, v, attr in G.edges(data='label'):
                bond_feat = get_bond_features(mol.GetBondBetweenAtoms(u-atom_offset, v-atom_offset)).tolist()
                bfeat.append(bond_feat)
                atoms_in_bonds.append([u, v])

                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid

                agraph[v].append(u)
                agraph[u].append(v)

                bgraph[u].append(eid)
                bgraph[v].append(eid)

        afeat = torch.tensor(afeat, dtype=torch.float)
        bfeat = torch.tensor(bfeat, dtype=torch.float)
        atoms_in_bonds = create_pad_tensor(atoms_in_bonds).long()
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)

        graph_tensors = (afeat, bfeat, agraph, bgraph, atoms_in_bonds)
        scopes = (atom_scope, bond_scope)

        if return_graphs:
            return graph_tensors, scopes, nx.union_all(all_G)
        else:
            return graph_tensors, scopes

def tensorize_bond_graphs(graph_batch, directed: bool, use_rxn_class: False,    
                          return_graphs: bool = False):
    if directed:
        edge_dict = {}
        fnode = [[0] * BOND_FDIM]
        if use_rxn_class:
            fmess = [[0, 0] + [0] * (ATOM_FDIM + 10) + [0] + [0] * 2 * (BOND_FDIM - 1)]
        else:
            fmess = [[0, 0] + [0] * ATOM_FDIM + [0] + [0] * 2 * (BOND_FDIM - 1)]
        agraph, bgraph = [[]], [[]]
        scope = []

        for bid, graph in enumerate(graph_batch):
            mol = graph.mol
            assert mol.GetNumAtoms() == len(graph.G_undir)
            offset = len(fnode)
            bond_graph = nx.line_graph(graph.G_undir)
            bond_graph = nx.to_directed(bond_graph)
            fnode.extend([None for v in bond_graph.nodes])

            scope.append((offset, mol.GetNumBonds()))
            ri = mol.GetRingInfo()

            bond_rings = ri.BondRings()
            bond_to_tuple = {bond.GetIdx(): tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
                             for bond in mol.GetBonds()}
            tuple_to_bond = {val: key for key, val in bond_to_tuple.items()}

            for u in bond_graph.nodes():
                agraph.append([])
                atom_idx_a, atom_idx_b = u
                bond_idx = tuple_to_bond[u] + offset
                fnode[bond_idx] = get_bond_features(mol.GetBondBetweenAtoms(atom_idx_a, atom_idx_b)).tolist()

            for u, v in bond_graph.edges():
                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                bond_idx_u = tuple_to_bond[tuple(sorted(u))] + offset
                bond_idx_v = tuple_to_bond[tuple(sorted(v))] + offset

                common_atom_idx = set(u).intersection(set(v))
                incommon_ring = 0
                for ring in bond_rings:
                    if (bond_idx_u-offset) in ring and (bond_idx_v-offset) in ring:
                        incommon_ring = 1
                        break

                common_atom = mol.GetAtomWithIdx(list(common_atom_idx)[0])
                edge_feats = get_atom_features(common_atom,
                                               use_rxn_class=use_rxn_class,
                                               rxn_class=graph.rxn_class) + [incommon_ring]
                atom_idx_a, atom_idx_b = u
                atom_idx_c, atom_idx_d = v

                bond_u = mol.GetBondBetweenAtoms(atom_idx_a, atom_idx_b)
                bond_v = mol.GetBondBetweenAtoms(atom_idx_c, atom_idx_d)

                bt_u, bt_v = bond_u.GetBondType(), bond_v.GetBondType()
                conj_u, conj_v = bond_u.GetIsConjugated(), bond_v.GetIsConjugated()
                sorted_u, sorted_v = sorted([bt_u, bt_v])

                feats_u = [float(sorted_u == bond_type) for bond_type in BOND_TYPES[1:]]
                feats_v = [float(sorted_v == bond_type) for bond_type in BOND_TYPES[1:]]

                edge_feats.extend(feats_u)
                edge_feats.extend(feats_v)
                edge_feats.extend(sorted([conj_u, conj_v]))

                mess_vec = [bond_idx_u, bond_idx_v] + edge_feats
                fmess.append(mess_vec)
                agraph[bond_idx_v].append(eid)
                bgraph.append([])

            for u, v in bond_graph.edges():
                eid = edge_dict[(u, v)]
                for w in bond_graph.predecessors(u):
                    if w == v: continue
                    bgraph[eid].append(edge_dict[(w, u)])

        fnode = torch.tensor(fnode, dtype=torch.float)
        fmess = torch.tensor(fmess, dtype=torch.float)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)

        graph_tensors = (fnode, fmess, agraph, bgraph, None)
        return graph_tensors, scope
