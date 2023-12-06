from multiprocessing import Pool
from typing import List, Tuple, Optional

from humanization import config_loader
from humanization.annotations import Annotation, ChainType
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


def calc_score_wrapper(sample):
    return calc_score(sequence, sample, annotation)


def worker_init(seq, ann):
    global sequence, annotation
    sequence = seq
    annotation = ann


class VGeneScorer:
    def __init__(self, annotation: Annotation, human_samples: List[str]):
        self.annotation = annotation
        self.human_samples = human_samples

    def query(self, sequence: List[str]) -> List[Tuple[str, float]]:
        worker_args = sequence, self.annotation
        with Pool(processes=config.get(config_loader.NCPU), initializer=worker_init, initargs=worker_args) as pool:
            v_gene_scores = pool.map(calc_score_wrapper, self.human_samples)
        result = []
        for idx, v_gene_score in sorted(enumerate(v_gene_scores), key=lambda x: x[1], reverse=True)[:3]:
            logger.info(f"{idx + 1} candidate: {v_gene_score}")
            result.append((self.human_samples[idx], v_gene_score))
        return result


def build_v_gene_scorer(annotation: Annotation, dataset_file: str, annotated_data: bool,
                        v_type: ChainType) -> Optional[VGeneScorer]:
    human_samples = read_human_samples(dataset_file, annotated_data, annotation, v_type)
    if human_samples is not None:
        v_gene_scorer = VGeneScorer(annotation, human_samples)
        logger.info(f"Created VGeneScorer with {len(human_samples)} samples")
        return v_gene_scorer
    else:
        return None
