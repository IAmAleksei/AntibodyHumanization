import argparse

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis import rms


def main(file_1, file_2):
    pdb_1 = mda.Universe(file_1)
    pdb_2 = mda.Universe(file_2)
    rmsds = align.alignto(pdb_1, pdb_2, select='name CA', match_atoms=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''MD analysis''')
    parser.add_argument('file-1', type=str)
    parser.add_argument('file-2', type=str)
    args = parser.parse_args()
    main(args.file_1, args.file_2)
