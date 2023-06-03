from typing import List, Tuple

from humanization import config_loader
from humanization.annotations import Annotation
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "V Gene Scorer")


class VGeneScorer:
    def __init__(self, annotation: Annotation, human_samples: List[str]):
        self.annotation = annotation
        self.human_samples = human_samples

    def _calc_score(self, seq_1: List[str], seq_2: str) -> float:
        same, total = 0, 0
        for i in range(self.annotation.v_gene_end + 1):
            if seq_1[i] != 'X' or seq_2[i] != 'X':
                total += 1
                if seq_1[i] == seq_2[i]:
                    same += 1
        return same / total

    def query(self, sequence: List[str]) -> Tuple[str, float]:
        best_sample_idx, best_v_gene_score = None, 0
        for idx, human_sample in enumerate(self.human_samples):
            v_gene_score = self._calc_score(sequence, human_sample)
            if v_gene_score > best_v_gene_score:
                best_sample_idx = idx
                best_v_gene_score = v_gene_score
        return self.human_samples[best_sample_idx], best_v_gene_score
