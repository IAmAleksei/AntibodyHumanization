import argparse
from multiprocessing import Pool
from typing import List, Tuple, Optional

from humanization import config_loader
from humanization.annotations import Annotation, ChainType, GeneralChainType, annotate_single, ChothiaHeavy
from humanization.dataset_preparer import read_v_gene_dataset
from humanization.utils import configure_logger, read_sequences, write_sequences

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
    def __init__(self, annotation: Annotation, human_samples: List[str], labels: List[str]):
        self.annotation = annotation
        self.human_samples = human_samples
        self.labels = labels
        if len(self.human_samples) != len(self.labels):
            raise RuntimeError(f"Lengths are different. Samples: {len(self.human_samples)}, labels: {len(self.labels)}")

    def query(self, sequence: List[str]) -> List[Tuple[str, float, str]]:
        worker_args = sequence, self.annotation
        with Pool(processes=config.get(config_loader.NCPU), initializer=worker_init, initargs=worker_args) as pool:
            v_gene_scores = pool.map(calc_score_wrapper, self.human_samples)
        result = []
        for idx, v_gene_score in sorted(enumerate(v_gene_scores), key=lambda x: x[1], reverse=True)[:2]:
            result.append((self.human_samples[idx], v_gene_score, self.labels[idx]))
        return result


def build_v_gene_scorer(annotation: Annotation, dataset_file: str, only_human: bool = True,
                        v_type: ChainType = None) -> Optional[VGeneScorer]:
    human_dataset = read_v_gene_dataset(dataset_file, annotation, only_human, v_type)
    if human_dataset is not None:
        v_gene_scorer = VGeneScorer(annotation, human_dataset[0], human_dataset[1])
        logger.info(f"Created VGeneScorer with {len(human_dataset[0])} samples")
        return v_gene_scorer
    else:
        return None


def get_similar_samples(annotation: Annotation, dataset_file: str, sequences: List[str], only_human: bool = True,
                        chain_type: GeneralChainType = None) -> List[Optional[List[Tuple[str, float, str]]]]:
    v_gene_scorer = build_v_gene_scorer(annotation, dataset_file)
    assert v_gene_scorer is not None
    result = []
    for seq in sequences:
        aligned_seq = annotate_single(seq, annotation, chain_type)
        result.append(v_gene_scorer.query(aligned_seq) if aligned_seq is not None else None)
    return result


def main(input_file, dataset_file, output_file):
    sequences = read_sequences(input_file)
    similar_samples = get_similar_samples(ChothiaHeavy(), dataset_file, sequences, only_human=False)
    out_sequences = []
    for idx, sample in enumerate(similar_samples):
        name = sequences[idx][0]
        if sample is None:
            logger.debug(f"Error while evaluating sample#{idx + 1} '{name}'")
        else:
            for nearest_seq, score, tp in sample:
                out_sequences.append((f"{name} Score={score}", nearest_seq, None))
    write_sequences(output_file, out_sequences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''VGeneScore calculator''')
    parser.add_argument('--input', type=str, required=False, help='Path to input fasta file')
    parser.add_argument('--output', type=str, required=False, help='Path to output fasta file')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')

    args = parser.parse_args()

    main(input_file=args.input,
         dataset_file=args.dataset,
         output_file=args.output)
