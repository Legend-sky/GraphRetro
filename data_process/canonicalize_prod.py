"""
Canonicalize the product SMILES, and then use substructure matching to infer
the correspondence to the original atom-mapped order. This correspondence is then
used to renumber the reactant atoms.
"""

from rdkit import Chem
import os
import argparse
import threading
import pandas as pd

DATA_DIR = f"{os.environ['SEQ_GRAPH_RETRO']}/datasets/uspto-full/"  #使用full的数据集

def canonicalize_prod(p):
    import copy
    p = copy.deepcopy(p)
    p = canonicalize(p)
    p_mol = Chem.MolFromSmiles(p)
    for atom in p_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    p = Chem.MolToSmiles(p_mol)
    return p

def remove_amap_not_in_product(rxn_smi):
    """
    Corrects the atom map numbers of atoms only in reactants. 
    This correction helps avoid the issue of duplicate atom mapping
    after the canonicalization step.
    """
    r, p = rxn_smi.split(">>")
    pmol = Chem.MolFromSmiles(p)
    if pmol is None:
        print("Error: pmol is None", flush=True)
        return None
    pmol_amaps = set([atom.GetAtomMapNum() for atom in pmol.GetAtoms()])
    max_amap = max(pmol_amaps) #Atoms only in reactants are labelled starting with max_amap

    rmol  = Chem.MolFromSmiles(r)
    if rmol is None:
        print("Error: rmol is None", flush=True)
        return None

    r_updated = Chem.MolToSmiles(rmol)
    rxn_smi_updated = r_updated + ">>" + p
    return rxn_smi_updated

def canonicalize(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
    except:
        print('no mol', flush=True)
        return smiles
    if tmp is None:
        return smiles
    tmp = Chem.RemoveHs(tmp)
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    return Chem.MolToSmiles(tmp)

def infer_correspondence(p):
    orig_mol = Chem.MolFromSmiles(p)
    canon_mol = Chem.MolFromSmiles(canonicalize_prod(p))
    matches = list(canon_mol.GetSubstructMatches(orig_mol))[0]  #这里分子太长子结构匹配会进入死循环
    idx_amap = {atom.GetIdx(): atom.GetAtomMapNum() for atom in orig_mol.GetAtoms()}

    correspondence = {}
    for idx, match_idx in enumerate(matches):
        match_anum = canon_mol.GetAtomWithIdx(match_idx).GetAtomMapNum()
        old_anum = idx_amap[idx]
        correspondence[old_anum] = match_anum
    return correspondence

def remap_rxn_smi(rxn_smi):
    r, p = rxn_smi.split(">>")
    canon_mol = Chem.MolFromSmiles(canonicalize_prod(p))
    correspondence = infer_correspondence(p)

    rmol = Chem.MolFromSmiles(r)
    for atom in rmol.GetAtoms():
        atomnum = atom.GetAtomMapNum()
        if atomnum in correspondence:
            newatomnum = correspondence[atomnum]
            atom.SetAtomMapNum(newatomnum)

    rmol = Chem.MolFromSmiles(Chem.MolToSmiles(rmol))
    rxn_smi_new = Chem.MolToSmiles(rmol) + ">>" + Chem.MolToSmiles(canon_mol)
    return rxn_smi_new, correspondence

def timeout_remap(rxn_smi_new, uspto_id):
    # 如果remap_rxn_smi执行超过一分钟，则跳过
    timer = threading.Timer(60, lambda: print(f"remap_rxn_smi执行超过一分钟，跳过 {uspto_id}"))
    timer.start()
    try:
        rxn_smi_new, _ = remap_rxn_smi(rxn_smi_new)
    except Exception as e:
        print(f"Error in remap_rxn_smi: {e}")
    finally:
        timer.cancel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory where data is located.")
    parser.add_argument("--filename", required=True, help="File with reactions to canonicalize")
    args = parser.parse_args()

    new_file = f"canonicalized_{args.filename}"
    df = pd.read_csv(f"{args.data_dir}/{args.filename}")
    print(f"Processing file of size: {len(df)}")

    new_dict = {'id': [], 'reactants>reagents>production': []}
    for idx in range(len(df)):
        if idx % 10000 == 0:
            print(f"Processing {idx}")
        element = df.loc[idx]
        # uspto_id, class_id, rxn_smi = element['id'], element['class'], element['reactants>reagents>production']
        #USPTO-FULL数据集中没有class列
        uspto_id, rxn_smi = element['id'], element['reactants>reagents>production']
        r, p = rxn_smi.split(">>")
        if r=='' or p=='' or len(p)>800:  #如果反应物或产物为空或者长度超过1000，则跳过
            continue
        # print(rxn_smi)
        rxn_smi_new = remove_amap_not_in_product(rxn_smi)   #移除不在产物p中的反应物r中的原子map
        if rxn_smi_new == None: #如果返回的产物为空，则跳过
            continue

        # 使用线程监控remap_rxn_smi的执行时间
        threading.Thread(target=timeout_remap, args=(rxn_smi_new, uspto_id)).start()
        
        # rxn_smi_new, _ = remap_rxn_smi(rxn_smi_new)         #重新map反应物r中的原子map
        new_dict['id'].append(uspto_id)
        # new_dict['class'].append(class_id)
        new_dict['reactants>reagents>production'].append(rxn_smi_new)

    new_df = pd.DataFrame.from_dict(new_dict)
    new_df.to_csv(f"{args.data_dir}/{new_file}", index=False)

if __name__ == "__main__":
    main()
