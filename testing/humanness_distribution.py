import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from humanization.common import config_loader, utils
from humanization.common.annotations import GeneralChainType, annotate_batch, HumatchNumbering
from humanization.common.utils import configure_logger
from humanization.humanness_calculator.model_wrapper import load_all_models

config = config_loader.Config()
logger = configure_logger(config, "Humanness distribution")


def main(input_file, raw_model_dirs: List[str]):
    models_dirs = list(zip(raw_model_dirs[::2], raw_model_dirs[1::2]))
    if input_file.endswith('fasta'):
        seqs = [s for _, s in utils.read_sequences(input_file)]
    else:
        with open(input_file, 'r') as f:
            seqs = [s for s in f.readlines() if len(s) > 0]
    annotation = HumatchNumbering()
    annotated_set = annotate_batch(seqs, annotation, GeneralChainType.HEAVY)[1]
    logger.info(f"{len(annotated_set)} antibodies annotated")
    assert len(annotated_set) == len(seqs)
    models_wrappers = [(name, load_all_models(d, GeneralChainType.HEAVY)) for name, d in models_dirs]
    plt.figure(figsize=(7, 5), dpi=400)
    for model_name, model_wrappers in models_wrappers:
        scores = []
        for seq in annotated_set:
            preds = [round(model.predict_proba([seq])[0, 1], 2) for model in model_wrappers.values()]
            scores.append(max(preds))
        d_result = np.histogram(scores, bins=10, range=(0.0, 1.0))
        xs = [(a + b) / 2 for a, b in zip(d_result[1], d_result[1][1:])]
        plt.plot(xs, d_result[0], label=model_name)
    plt.xlim((-0.05, 1.05))
    plt.legend(loc='upper left')
    plt.savefig("humanness_distribution.jpg")


# Run: "python3 humanness_distribution.py <input-file> <model-1-name> <model-1-path> <model-2-name> <model-2-path> ..."
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''CM models ada analyzer''')
    parser.add_argument('input', type=str, help='Input data')
    parser.add_argument('models', metavar='file', type=str, nargs='+', help='Name of models')
    args = parser.parse_args()
    models = args.models
    assert len(models) % 2 == 0, "Models list should have even length"

    main(args.input, args.models)
