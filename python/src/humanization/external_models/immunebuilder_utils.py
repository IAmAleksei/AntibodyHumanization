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


def main():
    while True:
        sequence = input("Enter sequence (or `exit`): ")
        if sequence.strip() == "exit":
            break
        pdb_path = input("Enter PDB path: ")
        make_pdb(sequence, pdb_path)


if __name__ == '__main__':
    main()
