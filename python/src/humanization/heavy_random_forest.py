import argparse
from typing import Tuple, Generator

import numpy as np
from catboost import Pool
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import config_loader
from humanization import abstract_random_forest
from humanization.annotations import load_annotation
from humanization.dataset import make_binary_target, format_confusion_matrix
from humanization.dataset_preparer import read_any_heavy_dataset
from humanization.models import ModelWrapper, HeavyChainType, save_model
from humanization.stats import plot_roc_auc, find_optimal_threshold, brute_force_threshold, plot_thresholds, \
    plot_comparison
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Heavy chain RF")


def get_threshold(metric, y, y_pred_proba, axis) -> Tuple[float, float]:
    threshold_points = brute_force_threshold(metric, y, y_pred_proba)
    threshold, metric_score = find_optimal_threshold(threshold_points)
    if axis is not None:
        plot_thresholds(threshold_points, metric, threshold, metric_score, axis[0])
        plot_roc_auc(y, y_pred_proba, axis[1])
    return threshold, metric_score


def plot_metrics(name: str, train_metrics: dict, val_metrics: dict, ax):
    ax.set_title(name)
    plot_comparison(name, train_metrics, "Train", val_metrics, "Validation", ax)


def build_tree(X_train, y_train_raw, X_val, y_val_raw, v_type: int, metric: str,
               iterative_learning: bool = False, print_metrics: bool = True):
    y_train = make_binary_target(y_train_raw, v_type)
    y_val = make_binary_target(y_val_raw, v_type)
    logger.debug(f"Dataset for V{v_type} tree contains {np.count_nonzero(y_train == 1)} positive samples")

    val_pool = Pool(X_val, y_val, cat_features=X_val.columns.tolist())
    logger.debug(f"Validation pool prepared")

    final_model = abstract_random_forest.build_tree(X_train, y_train, val_pool, iterative_learning)
    # TODO: Add train metrics

    y_val_pred_proba = final_model.predict_proba(X_val)[:, 1]
    if print_metrics:
        val_metrics = final_model.eval_metrics(data=val_pool, metrics=['Logloss', 'AUC'])
        logger.debug(f"Metrics evaluated")

        figure, axis = plt.subplots(2, 2, figsize=(9, 9))
        plt.suptitle(f'Tree IGHV{v_type}')
        plot_metrics('Logloss', {}, val_metrics, axis[0, 0])
        plot_metrics('AUC', {}, val_metrics, axis[0, 1])
        threshold, metric_score = get_threshold(metric, y_val, y_val_pred_proba, axis[1, :])
        plt.tight_layout()
        plt.show()
    else:
        threshold, metric_score = get_threshold(metric, y_val, y_val_pred_proba, None)
    logger.info(f"Optimal threshold is {threshold}, metric score = {metric_score}")
    return final_model, threshold


def build_trees(input_dir: str, schema: str, metric: str, annotated_data: bool,
                iterative_learning: bool, print_metrics: bool) -> Generator[ModelWrapper, None, None]:
    annotation = load_annotation(schema)
    X, y = read_any_heavy_dataset(input_dir, annotated_data, annotation)
    if X.isna().sum().sum() > 0 or y.isna().sum() > 0:
        raise RuntimeError("Found nans")
    X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.1, shuffle=True, random_state=42)
    logger.info(f"Train dataset: {X_.shape[0]} rows")
    logger.debug(f"Statistics:\n{y_.value_counts()}")
    logger.info(f"Test dataset: {X_test.shape[0]} rows")
    logger.debug(f"Statistics:\n{y_test.value_counts()}")
    for v_type in range(1, 8):
        logger.debug(f"Tree for V{v_type} is building...")
        model, threshold = build_tree(X_train, y_train, X_val, y_val, v_type, metric, iterative_learning, print_metrics)
        logger.debug(f"Tree for V{v_type} was built")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = np.where(y_pred_proba >= threshold, 1, 0)
        logger.info(format_confusion_matrix(make_binary_target(y_test, v_type), y_pred))
        logger.info(f"Tree for V{v_type} tested.")
        wrapped_model = ModelWrapper(HeavyChainType(str(v_type)), model, annotation, threshold)
        yield wrapped_model


def main(input_dir, schema, metric, output_dir, annotated_data, iterative_learning, print_metrics):
    for wrapped_model in build_trees(input_dir, schema, metric, annotated_data, iterative_learning, print_metrics):
        save_model(output_dir, wrapped_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''RF generator''')
    parser.add_argument('input', type=str, help='Path to directory where all .csv (or .csv.gz) are listed')
    parser.add_argument('output', type=str, help='Output models location')
    parser.add_argument('--annotated-data', action='store_true', help='Data is annotated')
    parser.add_argument('--raw-data', dest='annotated_data', action='store_false')
    parser.set_defaults(annotated_data=True)
    parser.add_argument('--iterative-learning', action='store_true', help='Iterative learning using data batches')
    parser.add_argument('--single-batch-learning', dest='iterative_learning', action='store_false')
    parser.set_defaults(iterative_learning=True)
    parser.add_argument('--schema', type=str, default="chothia", help='Annotation schema')
    parser.add_argument('--metric', type=str, default="youdens", help='Threshold optimized metric')
    parser.add_argument('--print-metrics', action='store_true', help='Print learning metrics')
    parser.set_defaults(print_metrics=False)
    args = parser.parse_args()

    main(input_dir=args.input,
         schema=args.schema,
         metric=args.metric,
         output_dir=args.output,
         annotated_data=args.annotated_data,
         iterative_learning=args.iterative_learning,
         print_metrics=args.print_metrics)
