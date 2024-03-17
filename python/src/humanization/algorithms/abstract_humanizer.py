import traceback
from abc import ABC
from typing import List, Tuple, NamedTuple, Optional, Dict

from humanization.common import config_loader
from humanization.common.annotations import GeneralChainType, Annotation
from humanization.common.utils import BLOSUM62, configure_logger
from humanization.common.v_gene_scorer import calc_score, VGeneScorer
from humanization.humanness_calculator.model_wrapper import ModelWrapper

config = config_loader.Config()
logger = configure_logger(config, "Abstract humanizer")


class InnerChange(NamedTuple):
    position: int
    old_aa: str
    aa: str


class SequenceChange(NamedTuple):
    changes: List[InnerChange]
    value: float
    values: Optional[Dict[str, float]] = None

    def is_defined(self):
        return self.changes is not None

    def apply(self, seq: List[str]):
        for ch in self.changes:
            seq[ch.position] = ch.aa

    def unapply(self, seq: List[str]):
        for ch in self.changes:
            seq[ch.position] = ch.old_aa

    def __repr__(self):
        if self.is_defined():
            values_str = ""
            if self.values:
                values_str = f" (from {[k + ':' + str(round(v, 2)) for k, v in self.values.items()]})"
            pos = [f'#{str(ch.position)} {ch.old_aa}->{ch.aa}' for ch in self.changes]
            return f"{'|'.join(pos)} with value {round(self.value, 5)}{values_str}"
        else:
            return "Undefined"


def blosum_sum(change: List[InnerChange]) -> float:
    if change is None:
        return 0.0
    return sum(BLOSUM62[ch.old_aa][ch.aa] for ch in change)


def is_change_less(left: SequenceChange, right: SequenceChange, use_aa_similarity: bool):
    left_value, right_value = left.value, right.value
    if use_aa_similarity:
        if left.is_defined():
            left_value += 0.001 * min(0.0, blosum_sum(left.changes))
        if right.is_defined():
            right_value += 0.001 * min(0.0, blosum_sum(right.changes))
    return left_value < right_value


class IterationDetails(NamedTuple):
    index: int
    model_metric: float
    v_gene_score: Optional[float] = None
    change: Optional[SequenceChange] = None
    all_changes: Optional[List[SequenceChange]] = None

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


def seq_to_str(sequence: List[str], with_x: bool, sep: str = "") -> str:
    return sep.join(c for c in sequence if c != 'X' or with_x)


class BaseHumanizer(ABC):
    def __init__(self, v_gene_scorer: Optional[VGeneScorer]):
        self.v_gene_scorer = v_gene_scorer

    def get_annotation(self) -> Annotation:
        pass

    def _get_v_gene_score(self, current_seq: List[str], human_sample: Optional[str] = None,
                          prefer_human_sample: bool = False) -> Tuple[Optional[str], Optional[float]]:
        if self.v_gene_scorer is not None and (not prefer_human_sample or human_sample is None):
            human_sample, v_gene_score, _ = self.v_gene_scorer.query(current_seq)[0]
            return human_sample, v_gene_score
        elif human_sample is not None:
            return human_sample, calc_score(current_seq, human_sample, self.get_annotation())
        else:
            return None, None

    def query(self, sequence: str, target_model_metric: float,
              target_v_gene_score: Optional[float]) -> List[Tuple[str, List[IterationDetails]]]:
        pass


class AbstractHumanizer(BaseHumanizer):
    def __init__(self, model_wrapper: ModelWrapper, v_gene_scorer: Optional[VGeneScorer]):
        super().__init__(v_gene_scorer)
        self.model_wrapper = model_wrapper

    def get_annotation(self) -> Annotation:
        return self.model_wrapper.annotation

    def _calc_metrics(self, current_seq: List[str], human_sample: Optional[str] = None,
                      prefer_human_sample: bool = False) -> Tuple[float, float]:
        current_value = self.model_wrapper.model.predict_proba(current_seq)[1]
        _, v_gene_score = self._get_v_gene_score(current_seq, human_sample, prefer_human_sample)
        return current_value, v_gene_score


def run_humanizer(sequences: List[Tuple[str, str]], humanizer: BaseHumanizer,
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
    v_gene_type = input(f"V gene type {general_chain_type.available_specific_types()}: ")
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
