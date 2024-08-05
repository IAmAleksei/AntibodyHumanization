import argparse

from Bio import SeqIO

from humanization.common import config_loader
from humanization.common.annotations import HeavyChainType, ChothiaHeavy, GeneralChainType, annotate_batch
from humanization.common.utils import configure_logger
from humanization.humanness_calculator.model_wrapper import load_all_models

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
    logger.info(f"{len(annotated_set)} antibodies annotated")
    assert len(annotated_set) == len(seqs)
    model_wrappers = load_all_models(model_dir, GeneralChainType.HEAVY)
    res = [(name, seq, adas[name]) for (name, _), seq in zip(seqs, annotated_set) if name in adas]
    model_types = [key for key in model_wrappers.keys()]
    res.sort(key=lambda x: x[2])
    logger.info(f"{len(res)} antibodies with ADA value")
    print("Name", "ADA", "Max", *[t.full_type() for t in model_types], sep='\t')
    for name, seq, ada in res:
        preds = [round(model_wrappers[t].predict_proba([seq])[0, 1], 2) for t in model_types]
        print(name, ada, max(preds), "", *preds, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''CM models ada analyzer''')
    parser.add_argument('models', type=str, help='Path to directory with models')
    args = parser.parse_args()

    main(args.models)
