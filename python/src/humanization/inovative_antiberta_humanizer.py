import argparse
from typing import Optional, List, Tuple

import numpy as np

from humanization import config_loader, utils
from humanization.abstract_humanizer import seq_to_str, IterationDetails, is_change_less, SequenceChange, \
    run_humanizer, BaseHumanizer, read_humanizer_options
from humanization.annotations import annotate_single, ChothiaHeavy, GeneralChainType, Annotation
from humanization.antiberta_utils import get_embeddings_delta, get_antiberta_embeddings
from humanization.utils import configure_logger, parse_list, read_sequences, write_sequences
from humanization.v_gene_scorer import VGeneScorer, is_v_gene_score_less, build_v_gene_scorer

config = config_loader.Config()
logger = configure_logger(config, "Inovative AntiBERTa2 humanizer")


def _get_embeddings(seqs: List[List[str]]) -> np.array:
    return get_antiberta_embeddings([seq_to_str(seq, False, " ") for seq in seqs])


def _get_embedding(seq: List[str]) -> np.array:
    return _get_embeddings([seq])[0, :]


class InovativeAntibertaHumanizer(BaseHumanizer):
    def __init__(self, v_gene_scorer: VGeneScorer, deny_use_aa: List[str], deny_change_aa: List[str]):
        super().__init__(v_gene_scorer)
        self.annotation = ChothiaHeavy()
        self.deny_insert_aa = deny_use_aa
        self.deny_delete_aa = deny_change_aa
        self.use_aa_similarity = False

    def get_annotation(self) -> Annotation:
        return self.annotation

    def _test_single_change(self, sequence: List[str], column_idx: int, new_aa: str) -> Tuple[List[str], SequenceChange]:
        aa_backup = sequence[column_idx]
        if aa_backup in self.deny_delete_aa or aa_backup == new_aa or new_aa in self.deny_insert_aa:
            return [], SequenceChange(None, aa_backup, None, 10e9)
        sequence[column_idx] = new_aa
        seq_copy = sequence.copy()
        candidate_change = SequenceChange(column_idx, aa_backup, new_aa, None)
        sequence[column_idx] = aa_backup
        return seq_copy, candidate_change

    def _find_best_change(self, current_seq: List[str], original_embedding: np.array, cur_human_sample: List[str]):
        best_change = SequenceChange(None, None, None, 10e9)
        all_candidates = []
        for idx, column_name in enumerate(self.get_annotation().segmented_positions):
            if column_name.startswith('cdr'):
                continue
            mod_seq, candidate_change = self._test_single_change(current_seq, idx, cur_human_sample[idx])
            if candidate_change.position is not None:
                all_candidates.append((mod_seq, candidate_change))
        logger.debug(f"Get embeddings for {len(all_candidates)} sequences")
        embeddings = _get_embeddings([mod_seq for mod_seq, _ in all_candidates])
        for idx, (_, candidate_change) in enumerate(all_candidates):
            candidate_change = candidate_change._replace(value=get_embeddings_delta(original_embedding, embeddings[idx]))
            if is_change_less(candidate_change, best_change, self.use_aa_similarity):
                best_change = candidate_change
        return best_change

    def _calc_metrics(self, original_embedding: np.array, current_seq: List[str], human_sample: Optional[str] = None,
                      prefer_human_sample: bool = False) -> Tuple[float, float]:
        current_value = get_embeddings_delta(original_embedding, _get_embedding(current_seq))
        _, v_gene_score = self._get_v_gene_score(current_seq, human_sample, prefer_human_sample)
        return current_value, v_gene_score

    def _query_one(self, original_seq, cur_human_sample, limit_delta: float, target_v_gene_score: float,
                   aligned_result: bool, prefer_human_sample: bool,
                   limit_changes: int) -> Tuple[str, List[IterationDetails]]:
        original_embedding = _get_embedding(original_seq)
        current_seq = original_seq.copy()
        logger.info(f"Used human sample: {cur_human_sample}")
        iterations = []
        current_value, v_gene_score = self._calc_metrics(original_embedding, current_seq, cur_human_sample,
                                                         prefer_human_sample)
        iterations.append(IterationDetails(0, current_value, v_gene_score, None))
        for it in range(1, min(config.get(config_loader.MAX_CHANGES), limit_changes) + 1):
            logger.debug(f"Iteration {it}. "
                         f"Current delta = {round(current_value, 6)}, V Gene score = {v_gene_score}")
            best_change = self._find_best_change(current_seq, original_embedding, cur_human_sample)
            if best_change.is_defined():
                prev_aa = current_seq[best_change.position]
                current_seq[best_change.position] = best_change.aa
                best_value, best_v_gene_score = self._calc_metrics(original_embedding, current_seq, cur_human_sample,
                                                                   prefer_human_sample)
                logger.debug(f"Trying apply metric {best_value} and v_gene_score {best_v_gene_score}")
                if best_value >= limit_delta:
                    current_seq[best_change.position] = prev_aa
                    logger.info(f"It {it}. Current metrics are best ({round(current_value, 6)})")
                    break
                column_name = self.get_annotation().segmented_positions[best_change.position]
                logger.debug(f"Best change position {column_name}: {prev_aa} -> {best_change.aa}")
                iterations.append(IterationDetails(it, best_value, best_v_gene_score, best_change))
                if best_value < limit_delta and is_v_gene_score_less(target_v_gene_score, best_v_gene_score):
                    logger.info(f"It {it}. Target metrics are reached (v_gene_score = {best_v_gene_score})")
                    break
                current_value, v_gene_score = best_value, best_v_gene_score
            else:
                logger.info(f"It {it}. No effective changes found."
                            f" Stop algorithm on model metric = {round(current_value, 6)}")
                break
        logger.info(f"Process took {len(iterations)} iterations")
        return seq_to_str(current_seq, aligned_result), iterations

    def query(self, sequence: str, limit_delta: float = 15, target_v_gene_score: float = 0.0, human_sample: str = None,
              aligned_result: bool = False, prefer_human_sample: bool = False,
              limit_changes: int = 999) -> List[Tuple[str, List[IterationDetails]]]:
        general_type = GeneralChainType.HEAVY
        current_seq = annotate_single(sequence, self.annotation, general_type)
        if current_seq is None:
            raise RuntimeError(f"{sequence} cannot be annotated")
        original_seq = [x for x in current_seq]
        logger.debug(f"Annotated sequence: {seq_to_str(current_seq, True)}")
        if not human_sample:
            logger.debug(f"Retrieve human sample from V Gene scorer")
            v_gene_samples = self.v_gene_scorer.query(current_seq)
            human_samples = [human_sample for human_sample, _, _ in v_gene_samples]
        else:
            human_sample = annotate_single(human_sample, self.annotation, general_type)
            human_samples = [human_sample]
        result = []
        for cur_human_sample in human_samples:
            result.append(self._query_one(original_seq, cur_human_sample, limit_delta, target_v_gene_score,
                          aligned_result, prefer_human_sample, limit_changes))
        return result


def process_sequences(v_gene_scorer, sequences, limit_delta=16.0, human_sample=None, deny_use_aa=utils.TABOO_INSERT_AA,
                      deny_change_aa=utils.TABOO_DELETE_AA, target_v_gene_score=None,
                      aligned_result=False, prefer_human_sample=False, limit_changes=999):
    humanizer = InovativeAntibertaHumanizer(v_gene_scorer, parse_list(deny_use_aa), parse_list(deny_change_aa))
    results = run_humanizer(sequences, humanizer, limit_delta, target_v_gene_score,
                            human_sample, aligned_result, prefer_human_sample, limit_changes)
    return results


def main(input_file, dataset_file, deny_use_aa, deny_change_aa, human_sample, limit_changes, output_file):
    sequences = read_sequences(input_file)
    _, target_delta, target_v_gene_score = read_humanizer_options(dataset_file)
    v_gene_scorer = build_v_gene_scorer(ChothiaHeavy(), dataset_file)
    assert v_gene_scorer is not None
    results = process_sequences(
        v_gene_scorer, sequences, target_delta, human_sample, deny_use_aa, deny_change_aa, target_v_gene_score,
        limit_changes=limit_changes
    )
    write_sequences(output_file, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Inovative antiberta humanizer''')
    parser.add_argument('--input', type=str, required=False, help='Path to input fasta file')
    parser.add_argument('--output', type=str, required=False, help='Path to output fasta file')
    parser.add_argument('--human-sample', type=str, required=False,
                        help='Human sample used for creation chimeric sequence')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    parser.add_argument('--deny-use-aa', type=str, default=utils.TABOO_INSERT_AA, required=False,
                        help='Amino acids that could not be used')
    parser.add_argument('--deny-change-aa', type=str, default=utils.TABOO_DELETE_AA, required=False,
                        help='Amino acids that could not be changed')
    parser.add_argument('--limit-changes', type=int, default=30, required=False, help='Limit count of changes')

    args = parser.parse_args()

    main(input_file=args.input,
         dataset_file=args.dataset,
         deny_use_aa=args.deny_use_aa,
         deny_change_aa=args.deny_change_aa,
         human_sample=args.human_sample,
         limit_changes=args.limit_changes,
         output_file=args.output)
