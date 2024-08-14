import argparse
import datetime
import json
import os.path

from humanization.algorithms import direct_humanizer, reverse_humanizer, antiberta2_humanizer, \
    reverse_antiberta2_humanizer, innovative_antiberta_humanizer
from humanization.common import config_loader
from humanization.common.annotations import ChainType, GeneralChainType, load_annotation, ChainKind
from humanization.humanness_calculator.model_wrapper import load_model, load_all_models
from humanization.common.utils import configure_logger
from humanization.common.v_gene_scorer import build_v_gene_scorer, get_similar_samples


config = config_loader.Config()
logger = configure_logger(config, "Benchmark")


def rnd(x):
    return round(x, 4) if x is not None else None


def main(models_dir, dataset_dir, wild_dataset_dir, fasta_output):
    with open('thera_antibodies.json', 'r') as fp:
        samples = json.load(fp)
    with open('extra_thera_antibodies.json', 'r') as fp:
        samples += json.load(fp)

    all_models = load_all_models(os.path.abspath(models_dir), GeneralChainType.HEAVY)
    annotation = load_annotation("chothia", ChainKind.HEAVY)
    v_gene_scorer = build_v_gene_scorer(annotation, dataset_dir)
    wild_v_gene_scorer = build_v_gene_scorer(annotation, wild_dataset_dir)
    prep_seqs = [(antibody['name'], antibody['heavy']['sequ'].replace('-', '')) for antibody in samples]
    innovative_result = innovative_antiberta_humanizer.process_sequences(
        v_gene_scorer, all_models, wild_v_gene_scorer, prep_seqs, limit_delta=15.0,
        target_v_gene_score=0.85, prefer_human_sample=False, change_batch_size=1, candidates_count=10
    )
    with open(fasta_output, 'w') as f:
        lines = []
        for name, res, det in innovative_result:
            lines.extend(
                [f"> {name}_i "
                 f"C={len(det.iterations):02d} T={det.chain_type} "
                 f"P={rnd(det.iterations[-1].model_metric)} "
                 f"HVG0={rnd(det.iterations[0].v_gene_score)} HVG={rnd(det.iterations[-1].v_gene_score)} "
                 f"WVG0={rnd(det.iterations[0].wild_v_gene_score)} WVG={rnd(det.iterations[-1].wild_v_gene_score)} "
                 f"H0={rnd(det.iterations[0].humanness_score)} H={rnd(det.iterations[-1].humanness_score)} "
                 f"TH={rnd(all_models[det.chain_type].threshold)}",
                 res])
        lines.append("")
        f.writelines("\n".join(lines))


if __name__ == '__main__':
    date = datetime.datetime.today().replace(microsecond=0)
    parser = argparse.ArgumentParser(description='''Benchmark direct humanizer''')
    parser.add_argument('--models', type=str, default="../models", help='Path to directory with models')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    parser.add_argument('--wild-dataset', type=str, required=False, help='Path to dataset for wildness calculation')
    parser.add_argument('--fasta-output', type=str, default=f"h_{date}.fasta", help='Generate fasta with all sequences')
    args = parser.parse_args()
    main(models_dir=args.models, dataset_dir=args.dataset, wild_dataset_dir=args.wild_dataset,
         fasta_output=args.fasta_output)
