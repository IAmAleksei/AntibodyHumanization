import argparse

import numpy as np
from matplotlib import pyplot as plt

from humanization.common.annotations import load_annotation, ChainKind, GeneralChainType
from humanization.common.utils import AA_ALPHABET
from humanization.humanness_calculator.model_wrapper import load_all_models


def main(model_dir):
    annotation = load_annotation("chothia", ChainKind.HEAVY)
    positions_count = len(annotation.segmented_positions)
    model_wrappers = load_all_models(model_dir, GeneralChainType.HEAVY)
    for chain_type, model_wrapper in model_wrappers.items():
        if model_wrapper.library() == 'sklearn':
            raw_importance = model_wrapper.model.feature_importances_
            blocked_importance = raw_importance.reshape((positions_count, len(AA_ALPHABET)))
            importance = np.sum(blocked_importance, axis=1)
        else:
            importance = model_wrapper.model.get_feature_importance()
        plt.figure(figsize=(16, 10), dpi=400)
        for i in range(positions_count):
            color = 'g' if annotation.segmented_positions[i].startswith('cdr') else 'b'
            plt.bar(i, importance[i], color=color, alpha=0.5)
        plt.xlim((-1, len(importance) + 1))
        plt.tight_layout()
        plt.savefig(f"f_imp_{chain_type}.jpg")
        # plt.show()
        print(f"Finished {chain_type}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''CM models feature importance''')
    parser.add_argument('--models', type=str, default='../sklearn_models2', help='Path to directory with models')
    args = parser.parse_args()
    main(args.models)
