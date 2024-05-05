import argparse

import numpy as np
from Bio.PDB import PDBParser


def main(pdb):
    parser = PDBParser()  # recruts the parsing class
    structure = parser.get_structure('Eculizumab',
                                     'Eculizumab_1fcf3_unrelaxed_rank_005_alphafold2_ptm_model_1_seed_000.pdb')

    model = structure.get_list()[0]
    chain = model.get_list()[0]
    residues = chain.get_list()
    for res1 in residues:
        dists = []
        for res2 in residues:
            if res1.id[1] == res2.id[1]:
                continue
            mn_dist = 1e9
            for a1 in filter(lambda x: x.element != 'H', res1.get_atoms()):
                for a2 in filter(lambda x: x.element != 'H', res2.get_atoms()):
                    mn_dist = min(mn_dist, np.linalg.norm(np.array(a1) - np.array(a2)))
            dists.append((mn_dist, res2))
        dists.sort()
        if res1.resname == "CYS" or res1.id[1] in [42, 47, 50]:
            print(res1)
            for d, r in dists[:5]:
                print(f"- {r}({d})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Group in pdb analyzer''')
    parser.add_argument('--pdb', type=str, required=False, help='Path to .pdb file')
    args = parser.parse_args()
    main(pdb=args.pdb)
