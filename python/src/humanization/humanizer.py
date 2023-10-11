import argparse
from typing import List, Optional, Tuple

from humanization import config_loader, utils
from humanization.abstract_humanizer import run_humanizer, AbstractHumanizer, SequenceChange, is_change_less, \
    IterationDetails, read_humanizer_options, seq_to_str, abstract_humanizer_parser_options
from humanization.annotations import annotate_single
from humanization.models import load_model, ModelWrapper
from humanization.utils import configure_logger, read_sequences, write_sequences, parse_list
from humanization.v_gene_scorer import VGeneScorer, build_v_gene_scorer, is_v_gene_score_less

config = config_loader.Config()
logger = configure_logger(config, "Humanizer")


class Humanizer(AbstractHumanizer):
    def __init__(self, model_wrapper: ModelWrapper, v_gene_scorer: Optional[VGeneScorer], modify_cdr: bool,
                 skip_positions: List[str], deny_use_aa: List[str], deny_change_aa: List[str], use_aa_similarity: bool):
        super().__init__(model_wrapper, v_gene_scorer)
        self.modify_cdr = modify_cdr
        self.skip_positions = skip_positions
        self.deny_insert_aa = deny_use_aa
        self.deny_delete_aa = deny_change_aa
        self.use_aa_similarity = use_aa_similarity

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

    def query(self, sequence: str, target_model_metric: float, target_v_gene_score: Optional[float] = None,
              aligned_result: bool = False) -> Tuple[str, List[IterationDetails]]:
        current_seq = annotate_single(sequence, self.model_wrapper.annotation,
                                      self.model_wrapper.chain_type.general_type())
        if current_seq is None:
            raise RuntimeError(f"{sequence} cannot be annotated")
        if self.v_gene_scorer is None and target_v_gene_score is not None:
            logger.warning(f"V Gene scorer not defined, so target score ignored")
        logger.info(f"Annotated sequence: {seq_to_str(current_seq, True)}")
        iterations = []
        current_value, v_gene_score = self._calc_metrics(current_seq)
        iterations.append(IterationDetails(0, current_value, v_gene_score, None))
        for it in range(1, config.get(config_loader.MAX_CHANGES) + 1):
            current_value, v_gene_score = self._calc_metrics(current_seq)
            logger.debug(f"Iteration {it}. "
                         f"Current model metric = {round(current_value, 6)}, V Gene score = {v_gene_score}")
            best_change = self._find_best_change(current_seq)
            if best_change.is_defined():
                prev_aa = current_seq[best_change.position]
                current_seq[best_change.position] = best_change.aa
                column_name = self.model_wrapper.annotation.segmented_positions[best_change.position]
                logger.debug(f"Best change position {column_name}: {prev_aa} -> {best_change.aa}")
                best_value, v_gene_score = self._calc_metrics(current_seq)
                iterations.append(IterationDetails(it, best_value, v_gene_score, best_change))
                if target_model_metric <= current_value and is_v_gene_score_less(target_v_gene_score, v_gene_score):
                    logger.info(f"Target metrics are reached ({round(current_value, 6)})")
                    break
            else:
                logger.info(f"No effective changes found. Stop algorithm on model metric = {round(current_value, 6)}")
                break
        return seq_to_str(current_seq, aligned_result), iterations


def process_sequences(models_dir, sequences, chain_type, target_model_metric, dataset_file=None, annotated_data=None,
                      modify_cdr=False, skip_positions="", deny_use_aa=utils.TABOO_INSERT_AA,
                      deny_change_aa=utils.TABOO_DELETE_AA, use_aa_similarity=True, target_v_gene_score=None,
                      aligned_result=False):
    model_wrapper = load_model(models_dir, chain_type)
    v_gene_scorer = build_v_gene_scorer(model_wrapper.annotation, dataset_file, annotated_data, chain_type)
    humanizer = Humanizer(
        model_wrapper, v_gene_scorer, modify_cdr,
        parse_list(skip_positions), parse_list(deny_use_aa), parse_list(deny_change_aa), use_aa_similarity
    )
    results = run_humanizer(sequences, humanizer, target_model_metric, target_v_gene_score, aligned_result)
    return results


def main(models_dir, input_file, dataset_file, annotated_data, modify_cdr, skip_positions,
         deny_use_aa, deny_change_aa, use_aa_similarity, output_file):
    sequences = read_sequences(input_file)
    chain_type, target_model_metric, target_v_gene_score = read_humanizer_options(dataset_file)
    results = process_sequences(
        models_dir, sequences, chain_type, target_model_metric, dataset_file, annotated_data, modify_cdr,
        skip_positions, deny_use_aa, deny_change_aa, use_aa_similarity, target_v_gene_score
    )
    write_sequences(output_file, results)


def common_parser_options(parser):
    abstract_humanizer_parser_options(parser)
    parser.add_argument('--modify-cdr', action='store_true', help='Allow CDR modifications')
    parser.add_argument('--skip-cdr', dest='modify_cdr', action='store_false', help='Deny CDR modifications')
    parser.set_defaults(modify_cdr=True)
    parser.add_argument('--deny-use-aa', type=str, default=utils.TABOO_INSERT_AA, required=False,
                        help='Amino acids that could not be used')
    parser.add_argument('--deny-change-aa', type=str,  default=utils.TABOO_DELETE_AA, required=False,
                        help='Amino acids that could not be changed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Humanizer''')
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
