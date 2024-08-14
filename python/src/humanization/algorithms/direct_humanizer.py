import argparse
from typing import List, Optional, Tuple

from humanization.algorithms.abstract_humanizer import run_humanizer, AbstractHumanizer, SequenceChange, \
    is_change_less, IterationDetails, read_humanizer_options, seq_to_str, abstract_humanizer_parser_options, \
    InnerChange, HumanizationDetails
from humanization.common import config_loader, utils
from humanization.common.annotations import annotate_single
from humanization.common.utils import configure_logger, read_sequences, write_sequences, parse_list
from humanization.common.v_gene_scorer import VGeneScorer, build_v_gene_scorer, is_v_gene_score_less
from humanization.humanness_calculator.model_wrapper import load_model, ModelWrapper

config = config_loader.Config()
logger = configure_logger(config, "Humanizer")


class Humanizer(AbstractHumanizer):
    def __init__(self, model_wrapper: ModelWrapper, v_gene_scorer: Optional[VGeneScorer], modify_cdr: bool,
                 skip_positions: List[str], deny_use_aa: List[str], deny_change_aa: List[str], use_aa_similarity: bool,
                 non_decreasing_v_gene: bool):
        super().__init__(model_wrapper, v_gene_scorer)
        self.modify_cdr = modify_cdr
        self.skip_positions = skip_positions
        self.deny_insert_aa = deny_use_aa
        self.deny_delete_aa = deny_change_aa
        self.use_aa_similarity = use_aa_similarity
        self.non_decreasing_v_gene = non_decreasing_v_gene if v_gene_scorer is not None else False

    def _test_single_change(self, sequence: List[str], column_idx: int,
                            current_v_gene_score: float = None) -> SequenceChange:
        aa_backup = sequence[column_idx]
        best_change = SequenceChange(None, 0.0)
        if aa_backup in self.deny_delete_aa:
            return best_change
        for new_aa in utils.AA_ALPHABET:  # TODO: make it batched
            if aa_backup == new_aa or new_aa in self.deny_insert_aa:
                continue
            sequence[column_idx] = new_aa
            new_value = self.model_wrapper.model.predict_proba(sequence)[1]
            candidate_change = SequenceChange([InnerChange(column_idx, aa_backup, new_aa)], new_value)
            if is_change_less(best_change, candidate_change, self.use_aa_similarity):
                satisfied_v_gene = not self.non_decreasing_v_gene
                if not satisfied_v_gene:
                    _, v_gene_score = self._calc_metrics(sequence)
                    satisfied_v_gene = v_gene_score >= current_v_gene_score
                if satisfied_v_gene:
                    best_change = candidate_change
        sequence[column_idx] = aa_backup
        return best_change

    def _find_best_change(self, current_seq: List[str], current_v_gene_score: float = None):
        current_value = self.model_wrapper.model.predict_proba(current_seq)[1]
        best_change = SequenceChange(None, current_value)
        for idx, column_name in enumerate(self.model_wrapper.annotation.segmented_positions):
            if not self.modify_cdr and column_name.startswith('cdr'):
                continue
            if column_name in self.skip_positions:
                continue
            candidate_change = self._test_single_change(current_seq, idx, current_v_gene_score)
            if is_change_less(best_change, candidate_change, self.use_aa_similarity):
                best_change = candidate_change
        return best_change

    def query(self, sequence: str, target_model_metric: float, target_v_gene_score: Optional[float] = None,
              aligned_result: bool = False, limit_changes: int = 999) -> List[Tuple[str, HumanizationDetails]]:
        current_seq = annotate_single(sequence, self.model_wrapper.annotation,
                                      self.model_wrapper.chain_type.general_type())
        if current_seq is None:
            raise RuntimeError(f"{sequence} cannot be annotated")
        if self.v_gene_scorer is None and target_v_gene_score is not None:
            logger.warning(f"V Gene scorer not defined, so target score ignored")
        logger.debug(f"Annotated sequence: {seq_to_str(current_seq, True)}")
        iterations = []
        current_value, v_gene_score = self._calc_metrics(current_seq)
        logger.info(f"Start model metric: ({round(current_value, 6)})")
        iterations.append(IterationDetails(0, current_value, v_gene_score))
        for it in range(1, min(config.get(config_loader.MAX_CHANGES), limit_changes) + 1):
            logger.debug(f"Iteration {it}. "
                         f"Current model metric = {round(current_value, 6)}, V Gene score = {v_gene_score}")
            best_change = self._find_best_change(current_seq, v_gene_score)
            if best_change.is_defined():
                best_change.apply(current_seq)
                logger.debug(f"Best change: {best_change}")
                best_value, best_v_gene_score = self._calc_metrics(current_seq)
                iterations.append(IterationDetails(it, best_value, best_v_gene_score, best_change))
                if target_model_metric <= best_value and is_v_gene_score_less(target_v_gene_score, best_v_gene_score):
                    logger.info(f"Target metrics are reached")
                    break
                current_value, v_gene_score = best_value, best_v_gene_score
            else:
                logger.info(f"No effective changes found")
                break
        logger.info(f"Final model metric: ({round(current_value, 6)})")
        return [(seq_to_str(current_seq, aligned_result), HumanizationDetails(iterations))]


def _process_sequences(model_wrapper, v_gene_scorer, sequences, target_model_metric,
                       modify_cdr=False, skip_positions="", deny_use_aa=utils.TABOO_INSERT_AA,
                       deny_change_aa=utils.TABOO_DELETE_AA, use_aa_similarity=True, target_v_gene_score=None,
                       aligned_result=False, limit_changes=999, non_decreasing_v_gene=False):
    humanizer = Humanizer(
        model_wrapper, v_gene_scorer, modify_cdr,
        parse_list(skip_positions), parse_list(deny_use_aa), parse_list(deny_change_aa), use_aa_similarity,
        non_decreasing_v_gene
    )
    results = run_humanizer(sequences, humanizer, target_model_metric, target_v_gene_score, aligned_result,
                            limit_changes)
    return results


def process_sequences(models_dir, sequences, chain_type, target_model_metric, dataset_file=None,
                      modify_cdr=False, skip_positions="", deny_use_aa=utils.TABOO_INSERT_AA,
                      deny_change_aa=utils.TABOO_DELETE_AA, use_aa_similarity=True, target_v_gene_score=None,
                      aligned_result=False, limit_changes=999, non_decreasing_v_gene=False):
    model_wrapper = load_model(models_dir, chain_type)
    v_gene_scorer = build_v_gene_scorer(model_wrapper.annotation, dataset_file, chain_type)
    return _process_sequences(model_wrapper, v_gene_scorer, sequences, target_model_metric, modify_cdr, skip_positions,
                              deny_use_aa, deny_change_aa, use_aa_similarity, target_v_gene_score, aligned_result,
                              limit_changes, non_decreasing_v_gene)


def main(models_dir, input_file, dataset_file, modify_cdr, skip_positions,
         deny_use_aa, deny_change_aa, use_aa_similarity, non_decreasing_v_gene, output_file):
    sequences = read_sequences(input_file)
    chain_type, target_model_metric, target_v_gene_score = read_humanizer_options(dataset_file)
    results = process_sequences(
        models_dir, sequences, chain_type, target_model_metric, dataset_file, modify_cdr,
        skip_positions, deny_use_aa, deny_change_aa, use_aa_similarity, target_v_gene_score,
        non_decreasing_v_gene=non_decreasing_v_gene
    )
    write_sequences(output_file, results)


def common_parser_options(parser):
    abstract_humanizer_parser_options(parser)
    parser.add_argument('--modify-cdr', action='store_true', help='Allow CDR modifications')
    parser.add_argument('--skip-cdr', dest='modify_cdr', action='store_false', help='Deny CDR modifications')
    parser.set_defaults(modify_cdr=True)
    parser.add_argument('--non-decreasing-v-gene', action='store_true', help='Deny decrease V gene score')
    parser.set_defaults(non_decreasing_v_gene=False)
    parser.add_argument('--deny-use-aa', type=str, default=utils.TABOO_INSERT_AA, required=False,
                        help='Amino acids that could not be used')
    parser.add_argument('--deny-change-aa', type=str, default=utils.TABOO_DELETE_AA, required=False,
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
         modify_cdr=args.modify_cdr,
         skip_positions=args.skip_positions,
         deny_use_aa=args.deny_use_aa,
         deny_change_aa=args.deny_change_aa,
         use_aa_similarity=args.use_aa_similarity,
         non_decreasing_v_gene=args.non_decreasing_v_gene,
         output_file=args.output)
