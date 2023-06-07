from typing import List, Tuple, Optional

from humanization import config_loader
from humanization.annotations import Annotation
from humanization.dataset_preparer import read_human_samples
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "V Gene Scorer")


def calc_score(seq_1: List[str], seq_2: str, annotation: Annotation) -> float:
    same, total = 0, 0
    for i in range(annotation.v_gene_end + 1):
        if seq_1[i] != 'X' or seq_2[i] != 'X':
            total += 1
            if seq_1[i] == seq_2[i]:
                same += 1
    return same / total


def is_v_gene_score_less(first: Optional[float], second: Optional[float]) -> bool:
    if first is None or second is None:
        return True
    return first < second


class VGeneScorer:
    def __init__(self, annotation: Annotation, human_samples: List[str]):
        self.annotation = annotation
        self.human_samples = human_samples

    def query(self, sequence: List[str]) -> Tuple[str, float]:
        best_sample_idx, best_v_gene_score = None, 0
        for idx, human_sample in enumerate(self.human_samples):
            v_gene_score = calc_score(sequence, human_sample, self.annotation)
            if v_gene_score > best_v_gene_score:
                best_sample_idx = idx
                best_v_gene_score = v_gene_score
        return self.human_samples[best_sample_idx], best_v_gene_score


def build_v_gene_scorer(annotation: Annotation, dataset_file: str, annotated_data: bool) -> Optional[VGeneScorer]:
    human_samples = read_human_samples(dataset_file, annotated_data, annotation)
    if human_samples is not None:
        v_gene_scorer = VGeneScorer(annotation, human_samples)
        return v_gene_scorer
    else:
        return None
