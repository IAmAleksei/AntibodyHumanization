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


def main(models_dir, dataset_dir, wild_dataset_dir, humanizer_type, fasta_output):
    models_dir = os.path.abspath(models_dir)
    open(fasta_output, 'w').close()
    with open('thera_antibodies.json', 'r') as fp:
        samples = json.load(fp)
    with open('extra_thera_antibodies.json', 'r') as fp:
        samples += json.load(fp)

    temp_seqs = [antibody['heavy']['sequ'].replace('-', '') for antibody in samples]
    annotation = load_annotation("chothia", ChainKind.HEAVY)
    human_samples = get_similar_samples(annotation, dataset_dir, temp_seqs, count=1, chain_type=GeneralChainType.HEAVY)
    for idx, antibody in enumerate(samples):
        antibody['heavy']['my_type'] = []
        if human_samples[idx] is not None:
            logger.info(f"{antibody['name']} human samples:")
            for iii, (s, vg, tp) in enumerate(human_samples[idx]):
                logger.info(f"{iii + 1}. Human sample: {s} V_gene: {vg} Type: {tp}")
                if vg < 0.4:
                    logger.warn("Extremely low v gene score, this sample will not be used")
                else:
                    antibody['heavy']['my_type'].append(ChainType.from_oas_type(tp).full_type())

    all_models = load_all_models(models_dir, GeneralChainType.HEAVY)
    common_v_gene_scorer = None
    for i in range(1, 8):
        tp = f'HV{i}'
        chain_type = ChainType.from_full_type(tp)
        logger.info(f"Starting processing type {tp}")
        model_wrapper = load_model(models_dir, chain_type)
        if common_v_gene_scorer is None:
            common_v_gene_scorer = build_v_gene_scorer(model_wrapper.annotation, dataset_dir)
        v_gene_scorer = build_v_gene_scorer(model_wrapper.annotation, dataset_dir, chain_type)
        wild_v_gene_scorer = build_v_gene_scorer(model_wrapper.annotation, wild_dataset_dir)
        logger.info(f"Resources loaded")
        for limit_changes in [30]:
            for model_metric in [0.99]:
                logger.info(f"Starting processing metric {model_metric}")
                logger.info(f'Processing metric={model_metric} type={tp}')
                prep_seqs = []
                for antibody in samples:
                    if tp in antibody['heavy']['my_type']:
                        prep_seqs.append((antibody['name'], antibody['heavy']['sequ'].replace('-', '')))
                if len(prep_seqs) == 0:
                    continue
                antiberta_result, innovative_result, rev_antiberta_result, direct_result, reverse_result = [], [], [], [], []
                if humanizer_type is None or humanizer_type == "antiberta":
                    antiberta_result = antiberta2_humanizer.process_sequences(
                        model_wrapper, v_gene_scorer, prep_seqs, limit_changes=limit_changes
                    )
                if humanizer_type is None or humanizer_type == "innovative":
                    innovative_result = innovative_antiberta_humanizer.process_sequences(
                        common_v_gene_scorer, all_models, wild_v_gene_scorer, prep_seqs, limit_delta=15.0,
                        target_v_gene_score=0.85, prefer_human_sample=False, limit_changes=limit_changes,
                        change_batch_size=1, candidates_count=3
                    )
                if humanizer_type is None or humanizer_type == "rev-antiberta":
                    rev_antiberta_result = reverse_antiberta2_humanizer._process_sequences(
                        model_wrapper, v_gene_scorer, prep_seqs, model_metric, limit_changes=limit_changes
                    )
                if humanizer_type is None or humanizer_type == "direct":
                    direct_result = direct_humanizer._process_sequences(
                        model_wrapper, v_gene_scorer, prep_seqs, model_metric, aligned_result=True,
                        limit_changes=limit_changes, non_decreasing_v_gene=True
                    )
                if humanizer_type is None or humanizer_type == "reverse":
                    reverse_result = reverse_humanizer._process_sequences(
                        model_wrapper, v_gene_scorer, prep_seqs, model_metric, target_v_gene_score=0.85,
                        aligned_result=True, limit_changes=limit_changes
                    )
                with open(fasta_output, 'a') as f:
                    lines = []
                    for name, res in antiberta_result:
                        lines.extend(
                            [f"> {name}_a_{limit_changes:02d}pch_{i}t",
                             res])
                    for name, res, det in innovative_result:
                        its = det.iterations
                        lines.extend(
                            [f"> {name}_i "
                             f"C={len(its):02d} T={i} "
                             f"P={rnd(its[-1].model_metric)} "
                             f"HVG0={rnd(its[0].v_gene_score)} HVG={rnd(its[-1].v_gene_score)} "
                             f"WVG0={rnd(its[0].wild_v_gene_score)} WVG={rnd(its[-1].wild_v_gene_score)} "
                             f"H0={rnd(its[0].humanness_score)} H={rnd(its[-1].humanness_score)} "
                             f"TH={rnd(model_wrapper.threshold)}",
                             res])
                    for name, res, det in rev_antiberta_result:
                        its = det.iterations
                        lines.extend(
                            [f"> {name}_b_{model_metric}_{limit_changes:02d}pch_{i}t"
                             f"{its[0].model_metric} {its[0].v_gene_score} {its[-1].model_metric} {its[-1].v_gene_score}",
                             res])
                    for name, res, det in direct_result:
                        its = det.iterations
                        lines.extend(
                            [f"> {name}_d_{model_metric}_{len(its):02d}ch_{i}t "
                             f"{its[0].model_metric} {its[0].v_gene_score} {its[-1].model_metric} {its[-1].v_gene_score}",
                             res])
                    for name, res, det in reverse_result:
                        its = det.iterations
                        lines.extend(
                            [f"> {name}_r_{model_metric}_{len(its):02d}ch_{i}t "
                             f"{its[0].model_metric} {its[0].v_gene_score} {its[-1].model_metric} {its[-1].v_gene_score}",
                             res])
                    lines.append("")
                    f.writelines("\n".join(lines))


if __name__ == '__main__':
    date = datetime.datetime.today().replace(microsecond=0)
    parser = argparse.ArgumentParser(description='''Benchmark direct humanizer''')
    parser.add_argument('--models', type=str, default="../models", help='Path to directory with models')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    parser.add_argument('--wild-dataset', type=str, required=False, help='Path to dataset for wildness calculation')
    parser.add_argument('--humanizer', type=str, default=None,
                        choices=[None, "antiberta", "rev-antiberta", "innovative", "direct", "reverse"], help='Humanizer type')
    parser.add_argument('--fasta-output', type=str, default=f"h_{date}.fasta", help='Generate fasta with all sequences')
    args = parser.parse_args()
    main(models_dir=args.models, dataset_dir=args.dataset, wild_dataset_dir=args.wild_dataset,
         humanizer_type=args.humanizer, fasta_output=args.fasta_output)
