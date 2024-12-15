import argparse
from typing import Optional, List, Tuple, Dict

import numpy as np

from humanization.algorithms.abstract_humanizer import seq_to_str, IterationDetails, is_change_less, SequenceChange, \
    run_humanizer, BaseHumanizer, InnerChange, blosum_sum, HumanizationDetails
from humanization.common import config_loader, utils
from humanization.common.annotations import annotate_single, ChothiaHeavy, GeneralChainType, Annotation, ChainType
from humanization.common.utils import configure_logger, parse_list, read_sequences, write_sequences, generate_report
from humanization.common.v_gene_scorer import VGeneScorer, is_v_gene_score_less, build_v_gene_scorer
from humanization.external_models.antiberta_utils import get_antiberta_embeddings
from humanization.external_models.embedding_utils import diff_embeddings
from humanization.humanness_calculator.model_wrapper import load_all_models, ModelWrapper

config = config_loader.Config()
logger = configure_logger(config, "Innovative AntiBERTa2 humanizer")

EMBEDDING_SOURCE = "antiberta"


# This function calculates embeddings for a list of sequences.
# Input: 2d array NxM, where N - count of sequences, M - maximum count of aminoacids in sequence with used annotation
# Sample input: [['Q', 'E', ...], ['Q', 'V', 'Q', ...]]
# Output: 2d array NxL, where N - count of sequences, L - size of embedding
# Sample output: numpy.array([[0.01, 0.53, ...], [0.32, 0.2, ...]])
def _get_embeddings(seqs: List[List[str]]) -> np.array:
    if EMBEDDING_SOURCE == "antiberta":
        return get_antiberta_embeddings([seq_to_str(seq, False, " ") for seq in seqs])
    # ...
    raise RuntimeError("Unknown embedding source")


# This function calculates humanness scores for a list of sequences.
# Input: 2d array NxM, where N - count of sequences, M - maximum count of aminoacids in sequence with used annotation
# Sample input: [['Q', 'E', ...], ['Q', 'V', 'Q', ...]]
# Output: 1d array of size N, where N - count of sequences
# Sample output: numpy.array([0.98, 0.04])
def _get_humanness_scores(
        sequences: List[List[str]],
        chain_type: ChainType,
        models=None) -> np.ndarray:
    if models is not None:
        # If models argument passed then use it.
        return models[chain_type].predict_proba(sequences)[:, 1]
    # Implement any custom algorithm here
    return np.ones(len(sequences), dtype=np.float)


def _get_embedding(seq: List[str]) -> np.array:
    return _get_embeddings([seq])[0, :]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class InnovativeAntibertaHumanizer(BaseHumanizer):
    def __init__(self, v_gene_scorer: VGeneScorer, wild_v_gene_scorer: VGeneScorer,
                 models: Optional[Dict[ChainType, ModelWrapper]], deny_use_aa: List[str],
                 deny_change_aa: List[str], deny_change_pos: List[str]):
        super().__init__(v_gene_scorer)
        self.wild_v_gene_scorer = wild_v_gene_scorer
        self.default_annotation = ChothiaHeavy()
        self.deny_insert_aa = deny_use_aa
        self.deny_delete_aa = deny_change_aa
        self.deny_change_pos = deny_change_pos
        self.use_aa_similarity = False
        self.models = models
        # Any extra dependencies may be put here

    def get_annotation(self, chain_type: ChainType = None) -> Annotation:
        if self.models is None:
            return self.default_annotation
        if chain_type is None:
            # Take the annotation from random model, because all models should have the same one
            return next(iter(self.models.values())).annotation
        return self.models[chain_type].annotation

    def _test_single_change(self, sequence: List[str], column_idx: int, new_aa: str) -> Optional[InnerChange]:
        aa_backup = sequence[column_idx]
        if aa_backup in self.deny_delete_aa or aa_backup == new_aa or new_aa in self.deny_insert_aa:
            return None
        return InnerChange(column_idx, aa_backup, new_aa)

    def _get_v_gene_penalty(self, mod_seq: List[str], cur_vgs: float, cur_wild_vgs: float) -> float:
        if self.wild_v_gene_scorer is None:
            return 0.0
        _, wild_vgs, _ = self.wild_v_gene_scorer.query(mod_seq)[0]
        _, vgs, _ = self.v_gene_scorer.query(mod_seq)[0]
        if vgs - wild_vgs > 0.001 or vgs < 0.85 or \
                (vgs < 0.86 and wild_vgs - cur_wild_vgs < -0.001 + vgs - cur_vgs) or wild_vgs - cur_wild_vgs < -0.001:
            return max(0.0, wild_vgs + 0.02 - vgs)
        else:
            return 1e6  # Reject change

    def _get_random_forest_value(self, sequences: List[List[str]], chain_type: ChainType) -> np.ndarray:
        return _get_humanness_scores(sequences, chain_type, self.models)

    def _generate_candidates(self, current_seq: List[str], all_candidate_changes: List[InnerChange], changes_left: int,
                             res: List[Tuple[List[str], List[InnerChange]]] = None, cur_index: int = 0,
                             cur_changes: List[InnerChange] = None) -> List[Tuple[List[str], List[InnerChange]]]:
        if res is None:
            res = []
        if cur_changes is None:
            cur_changes = []
        if changes_left == 0:
            res.append((current_seq.copy(), cur_changes.copy()))
        elif changes_left > 0:
            for i in range(cur_index, len(all_candidate_changes)):
                current_change = all_candidate_changes[i]
                cur_changes.append(current_change)
                current_seq[current_change.position] = current_change.aa
                self._generate_candidates(current_seq, all_candidate_changes, changes_left - 1, res, i + 1, cur_changes)
                current_seq[current_change.position] = current_change.old_aa
                cur_changes.pop()
        return res

    def _find_best_change(self, current_seq: List[str], original_embedding: np.array, cur_human_sample: List[str],
                          chain_type: ChainType, cur_v_gene_score: float, wild_v_gene_score: float,
                          change_batch_size: int):
        best_change = SequenceChange(None, 10e5)
        all_candidate_changes = []
        for idx, column_name in enumerate(self.get_annotation().segmented_positions):
            if column_name.startswith('cdr') or column_name in self.deny_change_pos:
                continue
            candidate_change = self._test_single_change(current_seq, idx, cur_human_sample[idx])
            if candidate_change is not None:
                all_candidate_changes.append(candidate_change)
        if len(all_candidate_changes) == 0:
            return best_change, []
        unevaluated_candidates = self._generate_candidates(current_seq, all_candidate_changes, change_batch_size)
        if cur_v_gene_score > 0.81:
            for bs in range(1, change_batch_size):
                unevaluated_candidates.extend(self._generate_candidates(current_seq, all_candidate_changes, bs))
        logger.debug(f"Get embeddings for {len(unevaluated_candidates)} sequences, batch is {change_batch_size}")
        embeddings = []
        for i, chunk in enumerate(chunks(unevaluated_candidates, 500)):
            embeddings.extend(_get_embeddings([mod_seq for mod_seq, _ in chunk]))
        humanness_degree = self._get_random_forest_value([mod_seq for mod_seq, _ in unevaluated_candidates], chain_type)
        all_candidates = []
        for idx, (mod_seq, changes) in enumerate(unevaluated_candidates):
            penalties = {
                'embeds': diff_embeddings(original_embedding, embeddings[idx]) * 25,
                'v_gene': self._get_v_gene_penalty(mod_seq, cur_v_gene_score, wild_v_gene_score) * 10,
                'humanness': 1 - humanness_degree[idx],
            }
            if self.use_aa_similarity:
                penalties['blosum'] = blosum_sum(changes) * (-0.04)
            all_candidates.append(SequenceChange(changes, value=sum(penalties.values()), values=penalties))
        for candidate_change in all_candidates:
            if is_change_less(candidate_change, best_change):
                best_change = candidate_change
        return best_change, all_candidates

    def _calc_v_gene_metrics(self, current_seq: List[str], human_sample: Optional[str] = None,
                             prefer_human_sample: bool = False) -> Tuple[float, float]:
        _, v_gene_score = self._get_v_gene_score(current_seq, human_sample, prefer_human_sample)
        if self.wild_v_gene_scorer:
            _, wild_v_gene_score, _ = self.wild_v_gene_scorer.query(current_seq)[0]
        else:
            wild_v_gene_score = None
        return v_gene_score, wild_v_gene_score

    def _query_one(self, original_seq, cur_human_sample, chain_type, limit_delta: float, target_v_gene_score: float,
                   aligned_result: bool, prefer_human_sample: bool, change_batch_size: int,
                   limit_changes: int) -> Tuple[str, HumanizationDetails]:
        original_embedding = _get_embedding(original_seq)
        current_seq = original_seq.copy()
        logger.info(f"Used human sample: {cur_human_sample}, chain type: {chain_type}")
        iterations = []
        current_value = 0.0
        v_gene_score, wild_v_gene_score = self._calc_v_gene_metrics(current_seq, cur_human_sample, prefer_human_sample)
        humanness_score = self._get_random_forest_value([current_seq], chain_type)[0]
        iterations.append(IterationDetails(0, current_value, v_gene_score, wild_v_gene_score, humanness_score, None))
        logger.info(f"Start metrics: V Gene score = {v_gene_score}, wild V Gene score = {wild_v_gene_score}")
        for it in range(1, min(config.get(config_loader.MAX_CHANGES), limit_changes) + 1):
            logger.info(f"Iteration {it}. Current delta = {round(current_value, 6)}, "
                        f"V Gene score = {v_gene_score}, wild V Gene score = {wild_v_gene_score}")
            best_change, all_changes = self._find_best_change(current_seq, original_embedding, cur_human_sample,
                                                              chain_type, v_gene_score, wild_v_gene_score,
                                                              change_batch_size)
            if best_change.is_defined():
                best_change.apply(current_seq)
                best_value = best_change.value
                best_v_gene_score, best_wild_v_gene_score = \
                    self._calc_v_gene_metrics(current_seq, cur_human_sample, prefer_human_sample)
                logger.debug(f"Trying apply metric {round(best_value, 6)} and V Gene score {best_v_gene_score}")
                if best_value >= limit_delta:
                    best_change.unapply(current_seq)
                    logger.info(f"It {it}. Current metrics are best ({round(current_value, 6)})")
                    break
                logger.info(f"Best change: {best_change}")
                humanness_score = self._get_random_forest_value([current_seq], chain_type)[0]
                iterations.append(IterationDetails(it, best_value, best_v_gene_score, best_wild_v_gene_score,
                                                   humanness_score, best_change, all_changes))
                if best_value < limit_delta and is_v_gene_score_less(target_v_gene_score, best_v_gene_score):
                    logger.info(f"It {it}. Target metrics are reached"
                                f" (v_gene_score = {best_v_gene_score}, wild_v_gene_score = {best_wild_v_gene_score})")
                    if is_v_gene_score_less(best_wild_v_gene_score, best_v_gene_score):
                        logger.info(f"Wild v gene score {best_wild_v_gene_score} is less human {best_v_gene_score}")
                        break
                current_value, v_gene_score, wild_v_gene_score = best_value, best_v_gene_score, best_wild_v_gene_score
            else:
                logger.info(f"It {it}. No effective changes found."
                            f" Stop algorithm on model metric = {round(current_value, 6)}")
                break
        logger.info(f"Process took {len(iterations)} iterations")
        logger.debug(f"Humanness: {iterations[-1].humanness_score} (threshold: {self.models[chain_type].threshold})")
        return seq_to_str(current_seq, aligned_result), HumanizationDetails(iterations, chain_type)

    def query(self, sequence: str, limit_delta: float = 15, target_v_gene_score: float = 0.0, human_sample: str = None,
              human_chain_type: str = None, aligned_result: bool = False, prefer_human_sample: bool = False,
              change_batch_size: int = 1, limit_changes: int = 999,
              candidates_count: int = 3) -> List[Tuple[str, HumanizationDetails]]:
        general_type = GeneralChainType.HEAVY
        current_seq = annotate_single(sequence, self.get_annotation(), general_type)
        if current_seq is None:
            raise RuntimeError(f"{sequence} cannot be annotated")
        original_seq = [x for x in current_seq]
        logger.debug(f"Annotated sequence: {seq_to_str(current_seq, True)}")
        if not human_sample:
            logger.debug(f"Retrieve {candidates_count} human sample from V Gene scorer")
            v_gene_samples = self.v_gene_scorer.query(current_seq, candidates_count)
            human_samples = [(human_sample, vgs, ChainType.from_oas_type(human_chain_type))
                             for human_sample, vgs, human_chain_type in v_gene_samples]
        else:
            human_sample = annotate_single(human_sample, self.get_annotation(), general_type)
            _, v_gene_score = self._get_v_gene_score(current_seq, human_sample, prefer_human_sample=True)
            human_samples = [(human_sample, v_gene_score, ChainType.from_oas_type(human_chain_type))]
        result = []
        for i, (cur_human_sample, vgs, chain_type) in enumerate(human_samples):
            logger.debug(f"Processing {i + 1} of {len(human_samples)} human sample (v_gene_score = {vgs})")
            result.append(self._query_one(original_seq, cur_human_sample, chain_type, limit_delta, target_v_gene_score,
                                          aligned_result, prefer_human_sample, change_batch_size, limit_changes))
        return result


def process_sequences(v_gene_scorer=None, models=None, wild_v_gene_scorer=None, sequences=None, limit_delta=16.0,
                      human_sample=None, human_chain_type=None, deny_use_aa=utils.TABOO_INSERT_AA,
                      deny_change_aa=utils.TABOO_DELETE_AA, deny_change_pos='', target_v_gene_score=None,
                      aligned_result=False, prefer_human_sample=False, change_batch_size=1, limit_changes=999,
                      candidates_count=3):
    humanizer = InnovativeAntibertaHumanizer(v_gene_scorer, wild_v_gene_scorer, models, parse_list(deny_use_aa),
                                             parse_list(deny_change_aa), parse_list(deny_change_pos))
    results = run_humanizer(sequences, humanizer, limit_delta, target_v_gene_score, human_sample, human_chain_type,
                            aligned_result, prefer_human_sample, change_batch_size, limit_changes, candidates_count)
    return results


def main(input_file, model_dir, dataset_file, wild_dataset_file, deny_use_aa, deny_change_aa, deny_change_pos,
         human_sample, human_chain_type, limit_changes, change_batch_size, candidates_count, report, output_file):
    sequences = read_sequences(input_file)
    v_gene_scorer = build_v_gene_scorer(ChothiaHeavy(), dataset_file)
    wild_v_gene_scorer = build_v_gene_scorer(ChothiaHeavy(), wild_dataset_file)
    assert v_gene_scorer is not None
    general_type = GeneralChainType.HEAVY
    models = load_all_models(model_dir, general_type) if model_dir else None
    results = process_sequences(
        v_gene_scorer, models, wild_v_gene_scorer, sequences, limit_delta=15.0,
        human_sample=human_sample, human_chain_type=human_chain_type,
        deny_use_aa=deny_use_aa, deny_change_aa=deny_change_aa,  deny_change_pos=deny_change_pos,
        target_v_gene_score=0.85, change_batch_size=change_batch_size, limit_changes=limit_changes,
        candidates_count=candidates_count
    )
    if report is not None:
        generate_report(report, results)
    write_sequences(output_file, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Greedy antiberta humanizer''')
    parser.add_argument('--input', type=str, required=False, help='Path to input fasta file')
    parser.add_argument('--output', type=str, required=False, help='Path to output fasta file')
    parser.add_argument('--models', type=str, help='Path to directory with random forest models')
    parser.add_argument('--human-sample', type=str, required=False,
                        help='Human sample used for creation chimeric sequence')
    parser.add_argument('--human-chain-type', type=str, required=False,
                        help='Type of provided human sample')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    parser.add_argument('--wild-dataset', type=str, required=False, help='Path to dataset for wildness calculation')
    parser.add_argument('--deny-use-aa', type=str, default=utils.TABOO_INSERT_AA, required=False,
                        help='Amino acids that could not be used')
    parser.add_argument('--deny-change-aa', type=str, default=utils.TABOO_DELETE_AA, required=False,
                        help='Amino acids that could not be changed')
    parser.add_argument('--deny-change-pos', type=str, default='', required=False,
                        help='Positions that could not be changed (fwr1_12, fwr2_2, etc.)')
    parser.add_argument('--change-batch-size', type=int, default=1, required=False,
                        help='Count of changes that will be applied in one iteration')
    parser.add_argument('--limit-changes', type=int, default=30, required=False, help='Limit count of changes')
    parser.add_argument('--candidates-count', type=int, default=3, required=False, help='Count of used references')
    parser.add_argument('--report', type=str, default=None, required=False, help='Path to report file')

    args = parser.parse_args()

    main(input_file=args.input,
         model_dir=args.models,
         dataset_file=args.dataset,
         wild_dataset_file=args.wild_dataset,
         deny_use_aa=args.deny_use_aa,
         deny_change_aa=args.deny_change_aa,
         deny_change_pos=args.deny_change_pos,
         human_sample=args.human_sample,
         human_chain_type=args.human_chain_type,
         limit_changes=args.limit_changes,
         change_batch_size=args.change_batch_size,
         candidates_count=args.candidates_count,
         report=args.report,
         output_file=args.output)
