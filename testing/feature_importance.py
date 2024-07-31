import argparse

from matplotlib import pyplot as plt

from humanization.common.annotations import HeavyChainType, load_annotation, ChainKind
from humanization.humanness_calculator.model_wrapper import load_model


def main(model_dir):
    annotation = load_annotation("chothia", ChainKind.HEAVY)
    model_wrapper = load_model(model_dir, HeavyChainType.V1)
    importance = model_wrapper.model.get_feature_importance()
    plt.figure(figsize=(16, 10), dpi=400)
    for i in range(len(importance)):
        color = 'g' if annotation.segmented_positions[i].startswith('cdr') else 'b'
        plt.bar(i, importance[i], color=color, alpha=0.5)
    plt.xlim((-1, len(importance) + 1))
    plt.tight_layout()
    plt.savefig("fimp.jpg")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''CM models feature importance''')
    parser.add_argument('models', type=str, help='Path to directory with models')
    args = parser.parse_args()
    main(args.models)
