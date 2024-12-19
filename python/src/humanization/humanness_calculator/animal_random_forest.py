import argparse

from sklearn.model_selection import train_test_split

from humanization.common import config_loader
from humanization.common.annotations import load_annotation, ChainKind, HeavyChainType
from humanization.common.utils import configure_logger
from humanization.dataset.dataset_preparer import NA_SPECIES
from humanization.dataset.dataset_reader import read_any_dataset
from humanization.dataset.one_hot_encoder import one_hot_encode
from humanization.humanness_calculator import abstract_random_forest
from humanization.humanness_calculator.abstract_random_forest import configure_abstract_parser, log_data_stats
from humanization.humanness_calculator.model_wrapper import save_model, ModelWrapper

config = config_loader.Config()
logger = configure_logger(config, "Animal RF")


def make_model(input_dir: str, schema: str, species: str,
               metric: str, iterative_learning: bool, print_metrics: bool, tree_lib: str) -> ModelWrapper:
    marker = lambda x: x != NA_SPECIES
    annotation = load_annotation(schema, ChainKind.HEAVY)
    X, y = read_any_dataset(input_dir, annotation, species=species)
    X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.07, shuffle=True, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.07, shuffle=True, random_state=42)
    log_data_stats(X_train, y_train, X_val, y_val, X_test, y_test)
    test_pool = one_hot_encode(annotation, X_test, cat_features=X_test.columns.tolist())
    return abstract_random_forest.make_model(X_train, y_train, X_val, y_val, test_pool, y_test, annotation,
                                             HeavyChainType.V1, metric, iterative_learning, print_metrics,
                                             marker, tree_lib)


def main(input_dir, schema, metric, output_dir, species, iterative_learning, print_metrics, tree_lib):
    wrapped_model = make_model(input_dir, schema, species, metric, iterative_learning, print_metrics, tree_lib)
    save_model(output_dir, wrapped_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Heavy chain RF generator''')
    configure_abstract_parser(parser)
    parser.add_argument('--species', type=str, default='mouse', help='Positive species for training')
    args = parser.parse_args()
    main(input_dir=args.input, schema=args.schema, metric=args.metric, output_dir=args.output, species=args.species,
         iterative_learning=args.iterative_learning, print_metrics=args.print_metrics, tree_lib=args.tree_lib)
