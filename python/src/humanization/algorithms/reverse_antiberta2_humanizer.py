from typing import Optional, List, Tuple

from humanization.algorithms.abstract_humanizer import AbstractHumanizer, IterationDetails, seq_to_str, \
    SequenceChange, is_change_less, run_humanizer, InnerChange
from humanization.common import config_loader, utils
from humanization.common.annotations import annotate_single
from humanization.common.utils import configure_logger, parse_list
from humanization.common.v_gene_scorer import VGeneScorer, is_v_gene_score_less
from humanization.external_models.antiberta_utils import get_mask_values
from humanization.humanness_calculator.model_wrapper import ModelWrapper

config = config_loader.Config()
logger = configure_logger(config, "Reverse AntiBERTa2 humanizer")


def _get_masks(seqs: List[List[str]]) -> List[str]:
    return get_mask_values([seq_to_str(seq, False, " ") for seq in seqs])


def _get_mask(seq: List[str]) -> str:
    return _get_masks([seq])[0]


class ReverseAntibertaHumanizer(AbstractHumanizer):
    def __init__(self, model_wrapper: ModelWrapper, v_gene_scorer: Optional[VGeneScorer],
                 deny_use_aa: List[str], deny_change_aa: List[str]):
        super().__init__(model_wrapper, v_gene_scorer)
        self.deny_insert_aa = deny_use_aa
        self.deny_delete_aa = deny_change_aa
        self.use_aa_similarity = False
        self.increasing_v_gene = True

    def _test_single_change(self, sequence: List[str], column_idx: int) -> Tuple[List[str], InnerChange]:
        aa_backup = sequence[column_idx]
        if aa_backup in self.deny_delete_aa:
            return [], None
        sequence[column_idx] = "[MASK]"
        seq_copy = sequence.copy()
        candidate_change = InnerChange(column_idx, aa_backup, None)
        sequence[column_idx] = aa_backup
        return seq_copy, candidate_change

    def _find_best_change(self, current_seq: List[str], current_v_gene_score: float = None):
        current_value = self.model_wrapper.model.predict_proba(current_seq)[1]
        best_change = SequenceChange(None, current_value)
        all_candidates = []
        for idx, column_name in enumerate(self.model_wrapper.annotation.segmented_positions):
            if column_name.startswith('cdr'):
                continue
            mod_seq, candidate_change = self._test_single_change(current_seq, idx)
            if candidate_change is not None:
                all_candidates.append((mod_seq, candidate_change))
        logger.debug(f"Get masks for {len(all_candidates)} sequences")
        masks = _get_masks([mod_seq for mod_seq, _ in all_candidates])
        for idx, (candidate_sequence, change) in enumerate(all_candidates):
            new_aa = masks[idx]
            candidate_sequence[change.position] = new_aa
            new_value = self.model_wrapper.model.predict_proba(candidate_sequence)[1]
            change = change._replace(aa=new_aa)
            candidate_change = SequenceChange([change], new_value)
            if is_change_less(best_change, candidate_change, self.use_aa_similarity):
                satisfied_v_gene = not self.increasing_v_gene
                if not satisfied_v_gene:
                    _, v_gene_score = self._calc_metrics(candidate_sequence)
                    satisfied_v_gene = v_gene_score > current_v_gene_score
                if satisfied_v_gene:
                    best_change = candidate_change
        return best_change

    def query(self, sequence: str, target_model_metric: float, target_v_gene_score: float = 0.0,
              aligned_result: bool = False, limit_changes: int = 999) -> List[Tuple[str, List[IterationDetails]]]:
        current_seq = annotate_single(sequence, self.model_wrapper.annotation,
                                      self.model_wrapper.chain_type.general_type())
        if current_seq is None:
            raise RuntimeError(f"{sequence} cannot be annotated")
        logger.debug(f"Annotated sequence: {seq_to_str(current_seq, True)}")
        iterations = []
        current_value, v_gene_score = self._calc_metrics(current_seq)
        logger.info(f"Start model metric: ({round(current_value, 6)})")
        iterations.append(IterationDetails(0, current_value, v_gene_score, None))
        for it in range(1, min(config.get(config_loader.MAX_CHANGES), limit_changes) + 1):
            logger.debug(f"Iteration {it}. "
                         f"Current model metric = {round(current_value, 6)}, V Gene score = {v_gene_score}")
            best_change = self._find_best_change(current_seq, v_gene_score)
            if best_change.is_defined():
                best_change.apply(current_seq)
                logger.debug(f"Best change: {best_change}")
                best_value, best_v_gene_score = self._calc_metrics(current_seq)
                iterations.append(IterationDetails(it, best_value, best_v_gene_score, best_change))
                if target_model_metric <= current_value and is_v_gene_score_less(target_v_gene_score,
                                                                                 best_v_gene_score):
                    logger.info(f"Target metrics are reached")
                    break
                current_value, v_gene_score = best_value, best_v_gene_score
            else:
                logger.info(f"No effective changes found")
                break
        logger.info(f"Final model metric: ({round(current_value, 6)})")
        logger.info(f"Process took {len(iterations)} iterations")
        return [(seq_to_str(current_seq, aligned_result), iterations)]


def _process_sequences(model_wrapper, v_gene_scorer, sequences, target_model_metric, deny_use_aa=utils.TABOO_INSERT_AA,
                       deny_change_aa=utils.TABOO_DELETE_AA, target_v_gene_score=None,
                       aligned_result=False, limit_changes=999):
    humanizer = ReverseAntibertaHumanizer(
        model_wrapper, v_gene_scorer, parse_list(deny_use_aa), parse_list(deny_change_aa)
    )
    results = run_humanizer(sequences, humanizer, target_model_metric, target_v_gene_score, aligned_result,
                            limit_changes)
    return results
