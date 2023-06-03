import argparse
from functools import total_ordering
from typing import List, Optional, NamedTuple, Tuple

import blosum
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


BLOSUM62 = blosum.BLOSUM(62)


class SequenceChange(NamedTuple):
    position: Optional[int]
    old_aa: Optional[str]
    aa: Optional[str]
    value: float

    def is_defined(self):
        return self.position is not None

    def __repr__(self):
        if self.is_defined():
            return f"Position = {self.position}, aa = {self.aa}"
        else:
            return "Undefined"


def is_change_less(left: SequenceChange, right: SequenceChange, use_aa_similarity: bool):
    left_value, right_value = left.value, right.value
    if use_aa_similarity:
        if left.is_defined():
            left_value += 0.0001 * min(0, BLOSUM62[left.old_aa][left.aa])
        if right.is_defined():
            right_value += 0.0001 * min(0, BLOSUM62[right.old_aa][right.aa])
    return left_value < right_value


class IterationDetails(NamedTuple):
    index: int
    model_metric: float
    v_gene_score: float
    change: Optional[SequenceChange]

    def __repr__(self):
        return f"Iteration {self.index}: " \
               f"model metric = {round(self.model_metric, 5)}, v gene score = {round(self.v_gene_score, 5)}, " \
               f"change = [{self.change if self.change is not None else 'Undefined'}]"


class Humanizer:
    def __init__(self, model_wrapper: ModelWrapper, v_gene_scorer: Optional[VGeneScorer], modify_cdr: bool,
                 skip_positions: List[str], deny_use_aa: List[str], deny_change_aa: List[str], use_aa_similarity: bool):
        self.model_wrapper = model_wrapper
        self.v_gene_scorer = v_gene_scorer
        self.modify_cdr = modify_cdr
        self.skip_positions = skip_positions
        self.deny_insert_aa = deny_use_aa
        self.deny_delete_aa = deny_change_aa
        self.use_aa_similarity = use_aa_similarity

    def _annotate_sequence(self, sequence: str) -> Optional[List[str]]:
        _, annotated_seq = annotate_batch([sequence], self.model_wrapper.annotation)
        if len(annotated_seq) == 1:
            return annotated_seq[0]
        else:
            logger.error(f"Cannot annotate `{sequence}`")
            return None

    def _test_single_change(self, sequence: List[str], column_idx: int) -> SequenceChange:
        aa_backup = sequence[column_idx]
        best_change = SequenceChange(None, aa_backup, None, 0.0)
        if aa_backup in self.deny_delete_aa:
            return best_change
        for new_aa in utils.AA_ALPHABET:  # TODO: make it batched
            if aa_backup == new_aa or new_aa in self.deny_insert_aa:
                continue
            sequence[column_idx] = new_aa
            new_value = self.model_wrapper.model.predict_proba(sequence)[1]
            candidate_change = SequenceChange(column_idx, aa_backup, new_aa, new_value)
            if is_change_less(best_change, candidate_change, self.use_aa_similarity):
                best_change = candidate_change
        sequence[column_idx] = aa_backup
        return best_change

    def _find_best_change(self, current_seq: List[str]):
        current_value = self.model_wrapper.model.predict_proba(current_seq)[1]
        best_change = SequenceChange(None, None, None, current_value)
        for idx, column_name in enumerate(self.model_wrapper.annotation.segmented_positions):
            if not self.modify_cdr and column_name.startswith('cdr'):
                continue
            if column_name in self.skip_positions:
                continue
            candidate_change = self._test_single_change(current_seq, idx)
            if is_change_less(best_change, candidate_change, self.use_aa_similarity):
                best_change = candidate_change
        return best_change

    @staticmethod
    def seq_to_str(sequence: List[str]) -> str:
        return "".join(c for c in sequence if c != 'X')

    def query(self, sequence: str, target_model_metric: float,
              target_v_gene_score: float = 0.0) -> Tuple[str, List[IterationDetails]]:
        current_seq = self._annotate_sequence(sequence)
        if current_seq is None:
            raise RuntimeError(f"{sequence} cannot be annotated")
        logger.debug(f"Annotated sequence: {''.join(current_seq)}")
        iterations = []
        for it in range(1, config.get(config_loader.MAX_CHANGES) + 1):
            current_value = self.model_wrapper.model.predict_proba(current_seq)[1]
            logger.info(f"Iteration {it}. Current model metric = {round(current_value, 6)}")
            if self.v_gene_scorer is not None:
                human_sample, v_gene_score = self.v_gene_scorer.query(current_seq)
                logger.debug(f"Nearest human sample: {human_sample}. V gene score = {round(v_gene_score, 6)}")
            else:
                v_gene_score = 1.0
            iterations.append(IterationDetails(it, current_value, v_gene_score, None))
            if current_value >= target_model_metric and v_gene_score >= target_v_gene_score:
                logger.info(f"Target metrics are reached ({round(current_value, 6)})")
                break
            best_change = self._find_best_change(current_seq)
            if best_change.is_defined():
                prev_aa = current_seq[best_change.position]
                current_seq[best_change.position] = best_change.aa
                column_name = self.model_wrapper.annotation.segmented_positions[best_change.position]
                logger.info(f"Best change position {column_name}: {prev_aa} -> {best_change.aa}")
                iterations[-1] = IterationDetails(it, current_value, v_gene_score, best_change)
            else:
                logger.info(f"No effective changes found. Stop algorithm on model metric = {round(current_value, 6)}")
                break
        return self.seq_to_str(current_seq), iterations


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


def build_v_gene_scorer(annotation, dataset_file, annotated_data) -> Optional[VGeneScorer]:
    human_samples = read_human_samples(dataset_file, annotated_data, annotation)
    if human_samples is not None:
        v_gene_scorer = VGeneScorer(annotation, human_samples)
        return v_gene_scorer
    else:
        return None


def run_humanizer(humanizer: Humanizer, sequences: List[Tuple[str, str]],
                  target_model_metric: float, target_v_gene_score: float) -> List[Tuple[str, str]]:
    results = []
    for name, sequence in sequences:
        try:
            result, _ = humanizer.query(sequence, target_model_metric, target_v_gene_score)
        except RuntimeError as _:
            result = ""
        results.append((name, result))
    return results


def parse_list(value: str) -> List[str]:
    return [x for x in value.split(",") if x != ""]


def read_human_samples(dataset_file=None, annotated_data=None, annotation=None) -> Optional[List[str]]:
    if dataset_file is not None:
        X, y = read_any_heavy_dataset(dataset_file, annotated_data, annotation)
        df = X[y != 'NOT_HUMAN'].reset_index(drop=True)
        human_samples = merge_all_columns(df)
        return human_samples
    else:
        return None


def main(models_dir, input_file, dataset_file, annotated_data, modify_cdr, skip_positions,
         deny_use_aa, deny_change_aa, use_aa_similarity, output_file):
    chain_type_str = input("Enter chain type (heavy or light): ")
    if chain_type_str.lower() in ["h", "heavy"]:
        chain_type_class = HeavyChainType
        v_gene_type = input("V gene type (1-7): ")
    elif chain_type_str.lower() in ["l", "light"]:
        chain_type_class = LightChainType
        v_gene_type = input("V gene type (kappa or lambda): ")
    else:
        raise RuntimeError(f"Unknown chain type: {chain_type_str}")
    target_model_metric = float(input("Enter target model metric: "))
    if dataset_file is not None:
        target_v_gene_score = float(input("Enter target V gene score: "))
    else:
        target_v_gene_score = 0.0
    chain_type = chain_type_class(v_gene_type)
    model_wrapper = load_model(models_dir, chain_type)
    v_gene_scorer = build_v_gene_scorer(model_wrapper.annotation, dataset_file, annotated_data)
    humanizer = Humanizer(
        model_wrapper, v_gene_scorer, modify_cdr,
        parse_list(skip_positions), parse_list(deny_use_aa), parse_list(deny_change_aa), use_aa_similarity
    )
    sequences = read_sequences(input_file)
    results = run_humanizer(humanizer, sequences, target_model_metric, target_v_gene_score)
    write_sequences(output_file, results)


def common_parser_options(parser):
    parser.add_argument('models', type=str, help='Path to directory with models')
    parser.add_argument('--modify-cdr', action='store_true', help='Allow CDR modifications')
    parser.add_argument('--skip-cdr', dest='modify_cdr', action='store_false', help='Deny CDR modifications')
    parser.set_defaults(modify_cdr=True)
    parser.add_argument('--skip-positions', required=False, default="", help='Positions that could not be changed')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    parser.add_argument('--annotated-data', action='store_true', help='Data is annotated')
    parser.add_argument('--raw-data', dest='annotated_data', action='store_false')
    parser.set_defaults(annotated_data=True)
    parser.add_argument('--use-aa-similarity', action='store_true', help='Use blosum table while search best change')
    parser.add_argument('--ignore-aa-similarity', dest='use_aa_similarity', action='store_false')
    parser.set_defaults(use_aa_similarity=True)
    parser.add_argument('--deny-use-aa', type=str, default=",".join(utils.TABOO_INSERT_AA), required=False,
                        help='Amino acids that could not be used')
    parser.add_argument('--deny-change-aa', type=str,  default=",".join(utils.TABOO_DELETE_AA), required=False,
                        help='Amino acids that could not be changed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Humanizer''')
    parser.add_argument('--input', type=str, required=False, help='Path to input fasta file')
    parser.add_argument('--output', type=str, required=False, help='Path to output fasta file')
    common_parser_options(parser)

    args = parser.parse_args()

    main(models_dir=args.models,
         input_file=args.input,
         dataset_file=args.dataset,
         annotated_data=args.annotated_data,
         modify_cdr=args.modify_cdr,
         skip_positions=args.skip_positions,
         deny_use_aa=args.deny_use_aa,
         deny_change_aa=args.deny_change_aa,
         use_aa_similarity=args.use_aa_similarity,
         output_file=args.output)
