import argparse
from collections import defaultdict

import edit_distance
import numpy as np
from Bio import SeqIO

from humanization import config_loader, antiberta_utils, immunebuilder_utils
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
        wild = None
        for way, seq in lst:
            if "Therap." == way:
                thera = seq
            if "Wild" == way:
                wild = seq
        if thera is None or wild is None:
            print("No therapeutic or wild seq")
            continue
        seq_emb_thera = antiberta_utils.get_antiberta_embedding(" ".join(thera))
        seq_emb_wild = antiberta_utils.get_antiberta_embedding(" ".join(wild))
        struct_emb_thera = immunebuilder_utils.get_immunebuilder_embedding(thera)
        struct_emb_wild = immunebuilder_utils.get_immunebuilder_embedding(wild)
        print(wild[:30] + "...", "Wild.")
        print(thera[:30] + "...", "Therap.", edit_distance.SequenceMatcher(wild, thera).distance(),
              round(np.linalg.norm(seq_emb_thera - seq_emb_wild), 2),
              round(np.linalg.norm(struct_emb_thera - struct_emb_wild), 2),
              sep=",")
        for i, (way, seq) in enumerate(lst):
            if way in ["Therap.", "Wild"]:
                continue
            sm_thera = edit_distance.SequenceMatcher(seq, thera)
            sm_wild = edit_distance.SequenceMatcher(seq, wild)
            seq_emb_seq = antiberta_utils.get_antiberta_embedding(" ".join(seq))
            diff_seq_emb_thera = np.linalg.norm(seq_emb_thera - seq_emb_seq)
            diff_seq_emb_wild = np.linalg.norm(seq_emb_wild - seq_emb_seq)
            struct_emb_seq = immunebuilder_utils.get_immunebuilder_embedding(seq)
            diff_struct_emb_thera = np.linalg.norm(struct_emb_thera - struct_emb_seq)
            diff_struct_emb_wild = np.linalg.norm(struct_emb_wild - struct_emb_seq)
            aligned_seq = annotate_single(seq, ChothiaHeavy(), GeneralChainType.HEAVY)
            if aligned_seq is None:
                continue
            score = v_gene_scorer.query(aligned_seq)[0][1]
            print(seq[:30] + "...", way,
                  sm_thera.distance(), sm_wild.distance(), round(score, 2),
                  round(diff_seq_emb_thera, 2), round(diff_seq_emb_wild, 2),
                  round(diff_struct_emb_thera, 2), round(diff_struct_emb_wild, 2),
                  sep=",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Comparator of sequences''')
    parser.add_argument('files', metavar='file', type=str, nargs='+', help='Name of files')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    args = parser.parse_args()
    main(args.files, args.dataset)
