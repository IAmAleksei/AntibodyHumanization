import argparse

from Bio import SeqIO

from humanization.common import config_loader
from humanization.common.annotations import HeavyChainType, ChothiaHeavy, GeneralChainType, annotate_batch
from humanization.common.utils import configure_logger
from humanization.humanness_calculator.model_wrapper import load_model

config = config_loader.Config()
logger = configure_logger(config, "Ada analyzer")


def main(model_dir):
    adas = {}
    seqs = []
    with open('therapeutics_ada.csv', 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            name, ada = line.split(',')
            adas[name] = float(ada)
    for seq in SeqIO.parse('all_therapeutics.fasta', 'fasta'):
        seqs.append((seq.name, str(seq.seq)))
    annotated_set = annotate_batch([seq for _, seq in seqs], ChothiaHeavy(), GeneralChainType.HEAVY)[1]
    logger.info(f"{len(annotated_set)} antibodies generated")
    model_wrapper = load_model(model_dir, HeavyChainType.V1)
    y_pred_proba = model_wrapper.model.predict_proba(annotated_set)[:, 1]
    assert len(y_pred_proba) == len(seqs)
    logger.info(f"Got predictions")
    for i, (name, _) in enumerate(seqs):
        if name not in adas:
            continue
        print(name, round(y_pred_proba[i], 2), adas[name], sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''CM models ada analyzer''')
    parser.add_argument('models', type=str, help='Path to directory with models')
    args = parser.parse_args()

    main(args.models)
