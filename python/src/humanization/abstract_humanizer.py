from abc import ABC
from typing import List, Tuple, NamedTuple, Optional, Callable

from humanization.models import LightChainType, HeavyChainType, ModelWrapper
from humanization.utils import BLOSUM62
from humanization.v_gene_scorer import calc_score, VGeneScorer


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


def seq_to_str(sequence: List[str], with_x: bool) -> str:
    return "".join(c for c in sequence if c != 'X' or with_x)


class AbstractHumanizer(ABC):
    def __init__(self, model_wrapper: ModelWrapper, v_gene_scorer: Optional[VGeneScorer]):
        self.model_wrapper = model_wrapper
        self.v_gene_scorer = v_gene_scorer

    def _get_v_gene_score(self, current_seq: List[str], human_sample: Optional[str] = None) -> float:
        if self.v_gene_scorer is not None:
            human_sample, v_gene_score = self.v_gene_scorer.query(current_seq)
            return v_gene_score
        elif human_sample is not None:
            return calc_score(current_seq, human_sample, self.model_wrapper.annotation)
        else:
            return 1.0

    def _calc_metrics(self, current_seq: List[str], human_sample: Optional[str] = None) -> Tuple[float, float]:
        current_value = self.model_wrapper.model.predict_proba(current_seq)[1]
        v_gene_score = self._get_v_gene_score(current_seq, human_sample)
        return current_value, v_gene_score

    def query(self, sequence: str, target_model_metric: float,
              target_v_gene_score: float) -> Tuple[str, List[IterationDetails]]:
        pass


def run_humanizer(sequences: List[Tuple[str, str]],
                  humanization: Callable[[str], Tuple[str, List[IterationDetails]]]) -> List[Tuple[str, str]]:
    results = []
    for name, sequence in sequences:
        try:
            result, _ = humanization(sequence)
        except RuntimeError as _:
            result = ""
        results.append((name, result))
    return results


def read_humanizer_options(dataset_file):
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
    return chain_type, target_model_metric, target_v_gene_score


def abstract_humanizer_parser_options(parser):
    parser.add_argument('models', type=str, help='Path to directory with models')
    parser.add_argument('--skip-positions', required=False, default="", help='Positions that could not be changed')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    parser.add_argument('--annotated-data', action='store_true', help='Data is annotated')
    parser.add_argument('--raw-data', dest='annotated_data', action='store_false')
    parser.set_defaults(annotated_data=True)
    parser.add_argument('--use-aa-similarity', action='store_true', help='Use blosum table while search best change')
    parser.add_argument('--ignore-aa-similarity', dest='use_aa_similarity', action='store_false')
    parser.set_defaults(use_aa_similarity=True)
    return parser
