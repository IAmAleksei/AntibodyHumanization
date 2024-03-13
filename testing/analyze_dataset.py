import argparse

from tqdm import tqdm

from humanization.common import utils
from humanization.common.annotations import GeneralChainType, load_annotation
from humanization.dataset.dataset_reader import read_any_dataset


def main(dataset_dir):
    annotation = load_annotation("chothia", GeneralChainType.HEAVY.kind())
    X, y = read_any_dataset(dataset_dir, annotation)
    for label in tqdm(["IGHV1", "IGHV2", "IGHV3", "IGHV4", "IGHV5", "IGHV6", "IGHV7", "NOT_HUMAN"]):
        seqs = X.loc[y == label]
        with open(f'aa_stat_{label}.csv', 'w') as f:
            f.write(','.join(utils.AA_ALPHABET) + "\n")
            for pos in annotation.segmented_positions:
                counts = seqs[pos].value_counts(normalize=True)
                res = []
                for aa in utils.AA_ALPHABET:
                    res.append(('%.5f' % counts[aa]) if aa in counts else "0.0")
                f.write(','.join(res) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Benchmark direct humanizer''')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    args = parser.parse_args()
    main(dataset_dir=args.dataset)
