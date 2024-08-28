import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from scipy.stats import pearsonr
from tabulate import tabulate

from humanization.common import config_loader
from humanization.common.annotations import ChothiaHeavy, GeneralChainType, annotate_batch
from humanization.common.utils import configure_logger
from humanization.humanness_calculator.model_wrapper import load_all_models
from humanization.humanness_calculator.stats import print_distribution

config = config_loader.Config()
logger = configure_logger(config, "Ada analyzer")


def main(model_dir):
    adas = {}
    seqs = []
    source = {}
    with open('therapeutics_ada.csv', 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            name, ada = line.split(',')
            adas[name] = float(ada)
    for seq in list(SeqIO.parse('all_therapeutics_2.fasta', 'fasta')):
        parts = seq.name.split("_")
        seqs.append((parts[0], str(seq.seq)))
        source[parts[0]] = parts[1]
    annotated_set = annotate_batch([seq for _, seq in seqs], ChothiaHeavy(), GeneralChainType.HEAVY)[1]
    logger.info(f"{len(annotated_set)} antibodies annotated")
    assert len(annotated_set) == len(seqs)
    model_wrappers = load_all_models(model_dir, GeneralChainType.HEAVY)
    model_types = [key for key in model_wrappers.keys()]
    res = [(name, seq, adas[name]) for (name, _), seq in zip(seqs, annotated_set) if name in adas]
    res.sort(key=lambda x: x[2])
    logger.info(f"{len(res)} antibodies with ADA value")
    print("Name", "", "ADA", "Max", "Positive", "", *[t.full_type() for t in model_types], sep='\t')
    matrix = [["", "Ada<10", "10<Ada<50", "50<Ada"], [">0.9", 0, 0, 0], ["Pos", 0, 0, 0], ["Neg", 0, 0, 0]]
    scores_k = defaultdict(list)
    adas_k = defaultdict(list)
    scores = []
    adas = []
    for name, seq, ada in res:
        preds = [round(model_wrappers[t].predict_proba([seq])[0, 1], 2) for t in model_types]
        is_positive = any(model_wrappers[t].predict([seq])[0] for t in model_types)
        mx = max(preds)
        col = 1 if ada < 10 else (2 if ada < 50 else 3)
        if mx > 0.9:
            matrix[1][col] += 1
        matrix[2 if is_positive else 3][col] += 1
        scores_k[source[name]].append(mx)
        adas_k[source[name]].append(ada)
        scores.append(mx)
        adas.append(ada)
        print(name, ada, mx, is_positive, "", "", *preds, sep='\t')
    print()
    print(tabulate(matrix))
    print()
    print_distribution(np.array(scores))
    print("Correlation", pearsonr(scores, adas))
    plt.figure(figsize=(7, 5), dpi=400)
    kinds = list(scores_k.keys())
    kind_translate = {'xi': "Chimeric", "u": "Human", "zu": "Humanized", "xizu": "Chimeric/Humanized", "o": "Mouse"}
    for k in kinds:
        plt.scatter(scores_k[k], adas_k[k], alpha=0.3, s=10, label=kind_translate[k])
    plt.legend(loc='upper right')
    plt.savefig("ada_scores.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''CM models ada analyzer''')
    parser.add_argument('--models', type=str, default='../sklearn_models2', help='Path to directory with models')
    args = parser.parse_args()

    main(args.models)
