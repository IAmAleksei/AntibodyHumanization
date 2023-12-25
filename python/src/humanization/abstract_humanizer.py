import traceback
from abc import ABC
from typing import List, Tuple, NamedTuple, Optional, Callable

from humanization import config_loader
from humanization.annotations import GeneralChainType
from humanization.models import ModelWrapper
from humanization.utils import BLOSUM62, configure_logger
from humanization.v_gene_scorer import calc_score, VGeneScorer


config = config_loader.Config()
logger = configure_logger(config, "Abstract humanizer")


class SequenceChange(NamedTuple):
    position: Optional[int]
    old_aa: Optional[str]
    aa: Optional[str]
    value: float

    def is_defined(self):
        return self.position is not None

    def __repr__(self):
        if self.is_defined():
            return f"Position {self.position}: {self.old_aa} -> {self.aa}"
        else:
            return "Undefined"


def is_change_less(left: SequenceChange, right: SequenceChange, use_aa_similarity: bool):
    left_value, right_value = left.value, right.value
    if use_aa_similarity:
        if left.is_defined():
            left_value += 0.001 * min(0, BLOSUM62[left.old_aa][left.aa])
        if right.is_defined():
            right_value += 0.001 * min(0, BLOSUM62[right.old_aa][right.aa])
    return left_value < right_value


class IterationDetails(NamedTuple):
    index: int
    model_metric: float
    v_gene_score: Optional[float]
    change: Optional[SequenceChange]

    def __repr__(self):
        if self.v_gene_score is not None:
            v_gene_score = f"v gene score = {round(self.v_gene_score, 5)}, "
        else:
            v_gene_score = ''
        if self.change is not None:
            change = f"change = [{self.change}]"
        else:
            change = ''
        return f"Iteration {self.index}: " \
               f"model metric = {round(self.model_metric, 5)}, " \
               f"{v_gene_score}" \
               f"{change}"


def seq_to_str(sequence: List[str], with_x: bool) -> str:
    return "".join(c for c in sequence if c != 'X' or with_x)


class AbstractHumanizer(ABC):
    def __init__(self, model_wrapper: ModelWrapper, v_gene_scorer: Optional[VGeneScorer]):
        self.model_wrapper = model_wrapper
        self.v_gene_scorer = v_gene_scorer

    def _get_v_gene_score(self, current_seq: List[str], human_sample: Optional[str] = None,
                          prefer_human_sample: bool = False) -> Tuple[Optional[str], Optional[float]]:
        if self.v_gene_scorer is not None and (not prefer_human_sample or human_sample is None):
            human_sample, v_gene_score, _ = self.v_gene_scorer.query(current_seq)[0]
            return human_sample, v_gene_score
        elif human_sample is not None:
            return human_sample, calc_score(current_seq, human_sample, self.model_wrapper.annotation)
        else:
            return None, None

    def _calc_metrics(self, current_seq: List[str], human_sample: Optional[str] = None,
                      prefer_human_sample: bool = False) -> Tuple[float, float]:
        current_value = self.model_wrapper.model.predict_proba(current_seq)[1]
        _, v_gene_score = self._get_v_gene_score(current_seq, human_sample, prefer_human_sample)
        return current_value, v_gene_score

    def query(self, sequence: str, target_model_metric: float,
              target_v_gene_score: Optional[float]) -> List[Tuple[str, List[IterationDetails]]]:
        pass


def run_humanizer(sequences: List[Tuple[str, str]], humanizer: AbstractHumanizer,
                  *args) -> List[Tuple[str, str, List[IterationDetails]]]:
    results = []
    for name, sequence in sequences:
        logger.info(f"Processing {name}")
        try:
            result_one = humanizer.query(sequence, *args)
        except RuntimeError as _:
            traceback.print_exc()
            result_one = [("", [])]
        for i, (result, its) in enumerate(result_one):
            results.append((f"{name}_cand{i + 1}", result, its))
    return results


def read_humanizer_options(dataset_file):
    general_chain_type = GeneralChainType(input("Enter chain type (H, K or L): "))
    v_gene_type = input(f"V gene type {general_chain_type.available_specific_types()}")
    chain_type = general_chain_type.specific_type(v_gene_type)
    target_model_metric = float(input("Enter target model metric: "))
    if dataset_file is not None:
        target_v_gene_score = float(input("Enter target V gene score: "))
    else:
        target_v_gene_score = None
    return chain_type, target_model_metric, target_v_gene_score


def abstract_humanizer_parser_options(parser):
    parser.add_argument('models', type=str, help='Path to directory with models')
    parser.add_argument('--skip-positions', required=False, default="", help='Positions that could not be changed')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    parser.add_argument('--use-aa-similarity', action='store_true', help='Use blosum table while search best change')
    parser.add_argument('--ignore-aa-similarity', dest='use_aa_similarity', action='store_false')
    parser.set_defaults(use_aa_similarity=True)
    return parser
