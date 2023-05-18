from typing import List, Tuple

from humanization import config_loader
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "V Gene Scorer")


class VGeneScorer:
    def __init__(self, human_samples: List[str]):
        self.human_samples = human_samples

    @staticmethod
    def _calc_affinity(seq_1: str, seq_2: str) -> int:
        return sum(c1 == c2 and c1 != 'X' for c1, c2 in zip(seq_1, seq_2))

    def query(self, sequence: List[str]) -> Tuple[str, float]:
        str_sequence = "".join(sequence)
        best_sample_idx, best_affinity = None, -1
        for idx, human_sample in enumerate(self.human_samples):
            affinity = self._calc_affinity(str_sequence, human_sample)
            if affinity > best_affinity:
                best_sample_idx = idx
                best_affinity = affinity
        v_gene_score = best_affinity / len(str_sequence)
        return self.human_samples[best_sample_idx], v_gene_score
