import argparse
import json
import random

import numpy as np
from Bio import SeqIO

from humanization.common import config_loader
from humanization.common.annotations import HeavyChainType, annotate_single, ChothiaHeavy, GeneralChainType, \
    annotate_batch
from humanization.common.utils import configure_logger, write_sequences
from humanization.humanness_calculator.model_wrapper import load_model, load_all_models
from humanization.humanness_calculator.stats import print_distribution


config = config_loader.Config()
logger = configure_logger(config, "Humanness calculator")


def main(model_dir, out_file):
    model_wrappers = load_all_models(model_dir, GeneralChainType.HEAVY)
    chain_types = [key for key in model_wrappers.keys()]
    raw_sequences = []
    has_ada = set()
    with open('therapeutics_ada.csv', 'r') as fp:
        for line in fp.readlines():
            has_ada.add(line.split(',')[0])
    for seq in SeqIO.parse('all_therapeutics.fasta', 'fasta'):
        if seq.name in has_ada:
            raw_sequences.append((seq.name, str(seq.seq)))
    for seq in SeqIO.parse('db.fasta', 'fasta'):
        raw_sequences.append((seq.name, str(seq.seq)))
    annotated_set = annotate_batch([s for _, s in raw_sequences], ChothiaHeavy(), GeneralChainType.HEAVY)[1]
    sequences = []
    for (name, oseq), seq in zip(raw_sequences, annotated_set):
        pred = max(model_wrappers[t].predict_proba([seq])[0, 1] for t in chain_types)
        sequences.append((name, pred, oseq))

    with open(out_file, 'w') as file:
        file.write(f"Name,Score,Sequence\n")
        for name, hs, seq in sequences:
            file.write(f"{name},{round(hs, 3)},{seq}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''CM models tester''')
    parser.add_argument('--models', type=str, default='../sklearn_models2', help='Path to directory with models')
    parser.add_argument('--out-file', type=str, default='humanness.csv', help='Path to output csv')
    # parser.add_argument('files', metavar='file', type=str, nargs='+', help='Name of files')
    args = parser.parse_args()

    main(args.models, args.out_file)
