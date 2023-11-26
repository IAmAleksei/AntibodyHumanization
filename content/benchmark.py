import argparse
import datetime
import json
import logging
import os.path

from humanization import humanizer, reverse_humanizer
from humanization.annotations import ChainType


logger = logging.getLogger(__name__)


def main(models_dir, dataset_dir, humanizer_type, fasta_output):
    models_dir = os.path.abspath(models_dir)
    open(fasta_output, 'w').close()
    with open('thera_antibodies.json', 'r') as fp:
        samples = json.load(fp)

    for model_metric in [0.5, 0.75, 0.9, 0.95, 0.99]:
        logger.info(f"Starting processing metric {model_metric}")
        for i in range(1, 8):
            tp = f'HV{i}'
            logger.info(f'Processing metric={model_metric} type={tp}')
            prep_seqs = []
            for antibody in samples:
                if antibody['heavy']['type'] == tp:
                    prep_seqs.append((antibody['name'], antibody['heavy']['sequ'].replace('-', '')))
            if len(prep_seqs) == 0:
                continue
            direct_result, reverse_result = [], []
            if humanizer_type is None or humanizer_type == "direct":
                direct_result = humanizer.process_sequences(
                    models_dir, prep_seqs, ChainType.from_full_type(tp), model_metric,
                    dataset_file=dataset_dir, annotated_data=True, aligned_result=True)
            if humanizer_type is None or humanizer_type == "reverse":
                reverse_result = reverse_humanizer.process_sequences(
                    models_dir, prep_seqs, ChainType.from_full_type(tp), model_metric,
                    target_v_gene_score=0.85, dataset_file=dataset_dir, annotated_data=True, aligned_result=True)
            with open(fasta_output, 'a') as f:
                lines = []
                for name, res, _ in direct_result:
                    lines.extend([f"> {name}_direct_{model_metric}", res])
                for name, res, _ in reverse_result:
                    lines.extend([f"> {name}_reverse_{model_metric}", res])
                lines.append("")
                f.writelines("\n".join(lines))


if __name__ == '__main__':
    date = datetime.datetime.today().replace(microsecond=0)
    parser = argparse.ArgumentParser(description='''Benchmark direct humanizer''')
    parser.add_argument('--models', type=str, default="../models", help='Path to directory with models')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    parser.add_argument('--humanizer', type=str, default=None, choices=[None, "direct", "reverse"], help='Humanizer type')
    parser.add_argument('--fasta-output', type=str, default=f"h_{date}.fasta", help='Generate fasta with all sequences')
    args = parser.parse_args()
    main(models_dir=args.models, dataset_dir=args.dataset, humanizer_type=args.humanizer,
         fasta_output=args.fasta_output)
