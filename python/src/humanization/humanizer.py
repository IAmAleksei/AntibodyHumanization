import argparse
from typing import List, Optional, NamedTuple

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from humanization import config_loader, utils
from humanization.annotations import annotate_batch
from humanization.dataset import merge_all_columns
from humanization.dataset_preparer import read_any_heavy_dataset
from humanization.models import load_model, HeavyChainType, LightChainType, ModelWrapper
from humanization.utils import configure_logger
from humanization.v_gene_scorer import VGeneScorer

config = config_loader.Config()
logger = configure_logger(config, "Humanizer")


class SequenceChange(NamedTuple):
    position: Optional[int]
    aa: Optional[str]
    value: float

    def is_defined(self):
        return self.position is not None


class Humanizer:
    def __init__(self, model_wrapper: ModelWrapper, target_model_metric: float,
                 v_gene_scorer: VGeneScorer, target_humanness: float, modify_cdr: bool):
        self.model_wrapper = model_wrapper
        self.target_model_metric = target_model_metric
        self.target_humanness = target_humanness
        self.v_gene_scorer = v_gene_scorer
        self.modify_cdr = modify_cdr

    def _annotate_sequence(self, sequence: str) -> Optional[List[str]]:
        _, annotated_seq = annotate_batch([sequence], self.model_wrapper.annotation)
        if len(annotated_seq) == 1:
            return annotated_seq[0]
        else:
            logger.error(f"Cannot annotate `{sequence}`")
            return None

    def _test_single_change(self, sequence: List[str], column_idx: int) -> SequenceChange:
        aa_backup = sequence[column_idx]
        best_change = SequenceChange(None, None, 0.0)
        if aa_backup == 'X':
            return best_change  # No inserts
        for new_aa in utils.AA_ALPHABET:  # TODO: make it batched
            sequence[column_idx] = new_aa
            new_value = self.model_wrapper.model.predict_proba(sequence)[1]
            if new_value > best_change.value:
                best_change = SequenceChange(column_idx, new_aa, new_value)
        sequence[column_idx] = aa_backup
        return best_change

    def _find_best_change(self, current_seq: List[str]):
        current_value = self.model_wrapper.model.predict_proba(current_seq)[1]
        best_change = SequenceChange(None, None, current_value)
        for idx, column_name in enumerate(self.model_wrapper.annotation.segmented_positions):
            if column_name.startswith('cdr') and not self.modify_cdr:
                continue
            candidate_change = self._test_single_change(current_seq, idx)
            if candidate_change.value > best_change.value:
                best_change = candidate_change
        return best_change

    @staticmethod
    def seq_to_str(sequence: List[str]) -> str:
        return "".join(c for c in sequence if c != 'X')

    def query(self, sequence: str) -> str:
        current_seq = self._annotate_sequence(sequence)
        if current_seq is None:
            return sequence
        for it in range(config.get(config_loader.MAX_CHANGES)):
            current_value = self.model_wrapper.model.predict_proba(current_seq)[1]
            logger.debug(f"Iteration {it + 1}. Current model metric = {current_value}")
            best_change = self._find_best_change(current_seq)
            if best_change.is_defined():
                prev_aa = current_seq[best_change.position]
                current_seq[best_change.position] = best_change.aa
                best_value = self.model_wrapper.model.predict_proba(current_seq)[1]
                logger.debug(f"Best change: model metric = {current_value} -> {best_value}")
                if self.v_gene_scorer is not None:
                    human_sample, humanness = self.v_gene_scorer.query(current_seq)
                    logger.debug(f"Nearest human sample: {human_sample}")
                    logger.debug(f"After change humanness = {humanness}")
                else:
                    humanness = 1.0
                column_name = self.model_wrapper.annotation.segmented_positions[best_change.position]
                logger.debug(f"Position {column_name}: {prev_aa} -> {best_change.aa}")
                if best_value >= self.target_model_metric and humanness >= self.target_humanness:
                    logger.info(f"Target metrics are reached ({best_value})")
                    break
            else:
                logger.info(f"No effective changes found. Stop algorithm on model metric = {current_value}")
                break
        return self.seq_to_str(current_seq)


def read_sequences(input_file):
    if not input_file:
        sequence = input("Enter sequence: ")
        result = [("CONSOLE", sequence)]
    else:
        result = [(seq.name, str(seq.seq)) for seq in SeqIO.parse(input_file, 'fasta')]
    return result


def write_sequences(output_file, sequences):
    if not output_file:
        for name, result in sequences:
            print(f'>{name}')
            print(result)
    else:
        seqs = [SeqRecord(Seq(seq), id=name, description='') for name, seq in sequences]
        SeqIO.write(seqs, output_file, 'fasta')


def run_humanizer(sequences, model_wrapper, target_model_metric,
                  human_samples=None, target_humanness=0.0, modify_cdr=False):
    v_gene_scorer = VGeneScorer(human_samples)
    humanizer = Humanizer(model_wrapper, target_model_metric, v_gene_scorer, target_humanness, modify_cdr)
    results = []
    for name, sequence in sequences:
        result = humanizer.query(sequence)
        results.append((name, result))
    return results


def main(models_dir, input_file, dataset_file, annotated_data, modify_cdr, output_file):
    chain_type_str = input("Enter chain type (heavy or light): ")
    if chain_type_str.lower() in ["h", "heavy"]:
        chain_type_class = HeavyChainType
        v_gene_type = input("V gene type (1-7): ")
    elif chain_type_str.lower() in ["l", "light"]:
        chain_type_class = LightChainType
        v_gene_type = input("V gene type (kappa or lambda): ")
    else:
        raise RuntimeError(f"Unknown chain type: {chain_type_str}")
    chain_type = chain_type_class(v_gene_type)
    target_model_metric = float(input("Enter target model metric: "))
    model_wrapper = load_model(models_dir, chain_type)
    if dataset_file is not None:
        target_humanness = float(input("Enter target humanness: "))
        X, y = read_any_heavy_dataset(dataset_file, annotated_data, model_wrapper.annotation)
        df = X[y != 'NOT_HUMAN'].reset_index(drop=True)
        human_samples = merge_all_columns(df)
    else:
        target_humanness = 0.0
        human_samples = None
    sequences = read_sequences(input_file)
    results = run_humanizer(sequences, model_wrapper, target_model_metric, human_samples, target_humanness, modify_cdr)
    write_sequences(output_file, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Humanizer''')
    parser.add_argument('models', type=str, help='Path to directory with models')
    parser.add_argument('--input', type=str, required=False, help='Path to input fasta file')
    parser.add_argument('--modify-cdr', action='store_true', help='Allowing CDR modifications')
    parser.add_argument('--skip-cdr', dest='modify_cdr', action='store_false')
    parser.set_defaults(modify_cdr=True)
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    parser.add_argument('--annotated-data', action='store_true', help='Data is annotated')
    parser.add_argument('--raw-data', dest='annotated_data', action='store_false')
    parser.set_defaults(annotated_data=True)
    parser.add_argument('--output', type=str, required=False, help='Path to output fasta file')
    args = parser.parse_args()

    main(models_dir=args.models,
         input_file=args.input,
         dataset_file=args.dataset,
         annotated_data=args.annotated_data,
         modify_cdr=args.modify_cdr,
         output_file=args.output)
