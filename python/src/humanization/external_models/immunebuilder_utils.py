import argparse
import os.path

import numpy as np
from ImmuneBuilder import NanoBodyBuilder2


predictor = NanoBodyBuilder2(numbering_scheme='chothia')


def get_immunebuilder_embedding(seq: str):
    antibody = predictor.predict({'H': seq})
    embedding = np.average(antibody.encodings[0], axis=0)
    return embedding


def make_pdb(seq: str, pdb_path: str):
    antibody = predictor.predict({'H': seq})
    antibody.save(pdb_path)


def main(antibody, output_path, files):
    from Bio import SeqIO
    for file in files:
        for seq in SeqIO.parse(file, 'fasta'):
            mab = seq.name.split("_")[0]
            if mab != antibody:
                continue
            way = seq.name.split("_")[2]
            prepared_seq = str(seq.seq).replace('X', '')
            make_pdb(prepared_seq, os.path.join(output_path, f"{mab}_{way}.pdb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Comparator of sequences''')
    parser.add_argument('antibody', type=str, help='Name of antibody')
    parser.add_argument('output', type=str, help='Output pdb location')
    parser.add_argument('files', metavar='file', type=str, nargs='+', help='Name of files')
    args = parser.parse_args()
    main(args.antibody, args.output, args.files)
