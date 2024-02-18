import argparse
from collections import defaultdict

import edit_distance
from Bio import SeqIO

from humanization import config_loader
from humanization.annotations import ChothiaHeavy, annotate_single, GeneralChainType
from humanization.utils import configure_logger
from humanization.v_gene_scorer import build_v_gene_scorer

config = config_loader.Config()
logger = configure_logger(config, "Comparator")


def main(files, dataset):
    v_gene_scorer = build_v_gene_scorer(ChothiaHeavy(), dataset)
    seqs = defaultdict(list)
    for file in files:
        for seq in SeqIO.parse(file, 'fasta'):
            mab = seq.name.split("_")[0]
            way = seq.name.split("_")[2]
            seqs[mab].append((way, str(seq.seq).replace('X', '')))
    for mab, lst in seqs.items():
        print(mab)
        thera = None
        for way, seq in lst:
            if "Therap." == way:
                thera = seq
        if thera is None:
            print("No therapeutic seq")
            continue
        print(thera, "Therap.")
        for i, (way, seq) in enumerate(lst):
            if "Therap." == way:
                continue
            sm = edit_distance.SequenceMatcher(seq, thera)
            aligned_seq = annotate_single(seq, ChothiaHeavy(), GeneralChainType.HEAVY)
            if aligned_seq is None:
                continue
            score = v_gene_scorer.query(aligned_seq)[0][1]
            sm.matches()
            print(seq, sm.distance(), round(sm.ratio(), 2), round(score, 2), way)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Comparator of sequences''')
    parser.add_argument('files', metavar='file', type=str, nargs='+', help='Name of files')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    args = parser.parse_args()
    main(args.files, args.dataset)
