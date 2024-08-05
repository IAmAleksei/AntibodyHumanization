import argparse
from typing import Tuple, Generator, Callable, Union

import numpy as np
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches

from humanization.common import config_loader
from humanization.common.annotations import Annotation, load_annotation, GeneralChainType, ChainType
from humanization.common.utils import configure_logger
from humanization.dataset.dataset_preparer import format_confusion_matrix, make_binary_target
from humanization.dataset.dataset_reader import read_any_dataset
from humanization.dataset.one_hot_encoder import one_hot_encode
from humanization.humanness_calculator.model_wrapper import ModelWrapper
from humanization.humanness_calculator.stats import brute_force_threshold, find_optimal_threshold, plot_thresholds, \
    plot_roc_auc, plot_comparison

config = config_loader.Config()
logger = configure_logger(config, "Abstract chain RF")


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


def log_data_stats(X_, y_, X_test, y_test):
    logger.info(f"Train dataset: {X_.shape[0]} rows")
    logger.debug(f"Statistics:\n{y_.value_counts()}")
    logger.info(f"Test dataset: {X_test.shape[0]} rows")
    logger.debug(f"Statistics:\n{y_test.value_counts()}")


def build_tree_impl(X_train, y_train, val_pool, annotation: Annotation, tree_lib: str,
                    iterative_learning: bool = False) -> Union[RandomForestClassifier, CatBoostClassifier]:
    if iterative_learning:
        # Cast to list for length calculation
        batches = list(gen_batches(X_train.shape[0], config.get(config_loader.LEARNING_BATCH_SIZE)))
    else:
        batches = [slice(X_train.shape[0])]
    cnt_batches = len(batches)
    batch_estimators = config.get(config_loader.TOTAL_ESTIMATORS) // cnt_batches
    final_model: Union[RandomForestClassifier, CatBoostClassifier] = None
    if tree_lib == 'sklearn':
        final_model = RandomForestClassifier(n_estimators=0, warm_start=True)
    for idx, batch in enumerate(batches):
        logger.debug(f"Model training. Batch {idx + 1} of {cnt_batches}")
        X_train_batch = X_train[batch]
        y_train_batch = y_train[batch]

        count_unique = len(np.unique(y_train_batch))
        if count_unique <= 1:
            logger.info("Skip batch with bad diverse target values (at most 1 unique value)")
            continue  # Otherwise, forest will be bad

        train_pool = one_hot_encode(annotation, X_train_batch, y_train_batch, tree_lib,
                                    cat_features=X_train_batch.columns.tolist())
        if tree_lib == 'catboost':
            model = CatBoostClassifier(
                depth=config.get(config_loader.TREE_DEPTH),
                loss_function='Logloss',
                grow_policy='Depthwise',
                used_ram_limit=config.get(config_loader.MEMORY_LIMIT),
                learning_rate=config.get(config_loader.TREE_LEARNING_RATE),
                verbose=config.get(config_loader.VERBOSE_FREQUENCY),
                max_ctr_complexity=config.get(config_loader.MAX_CTR_COMPLEXITY),
                n_estimators=batch_estimators,
                min_data_in_leaf=config.get(config_loader.MIN_DATA_IN_LEAF),
            )
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=10, init_model=final_model)
            final_model = model
        else:
            final_model.n_estimators += batch_estimators
            final_model.fit(train_pool, y_train_batch)

        del train_pool, X_train_batch, y_train_batch

    return final_model


def build_tree(X_train, y_train_raw, X_val, y_val_raw, marker: Callable[[str], bool], label: str, metric: str,
               annotation: Annotation, tree_lib: str, iterative_learning: bool = False, print_metrics: bool = True):
    y_train = make_binary_target(y_train_raw, marker)
    y_val = make_binary_target(y_val_raw, marker)
    logger.debug(f"Dataset for {label} tree contains {np.count_nonzero(y_train == 1)} positive samples")

    val_pool = one_hot_encode(annotation, X_val, y_val, cat_features=X_val.columns.tolist())
    logger.debug(f"Validation pool prepared")

    final_model = build_tree_impl(X_train, y_train, val_pool, annotation, tree_lib, iterative_learning)

    y_val_pred_proba = final_model.predict_proba(val_pool)[:, 1]
    if print_metrics and tree_lib == 'catboost':
        val_metrics = final_model.eval_metrics(data=val_pool, metrics=['Logloss', 'AUC'])
        logger.debug(f"Metrics evaluated")

        # TODO: Add train metrics
        figure, axis = plt.subplots(2, 2, figsize=(9, 9))
        plt.suptitle(f'Tree IG{label}')
        plot_metrics('Logloss', {}, val_metrics, axis[0, 0])
        plot_metrics('AUC', {}, val_metrics, axis[0, 1])
        threshold, metric_score = get_threshold(metric, y_val, y_val_pred_proba, axis[1, :])
        plt.tight_layout()
        plt.show()
    else:
        threshold, metric_score = get_threshold(metric, y_val, y_val_pred_proba, None)
    del val_pool
    logger.info(f"Optimal threshold is {threshold}, metric score = {metric_score}")
    return final_model, threshold


def make_model(X_train, y_train, X_val, y_val, test_pool, y_test, annotation: Annotation, v_type: ChainType,
               metric: str, iterative_learning: bool, print_metrics: bool, marker=None, tree_lib: str = 'catboost'):
    if not marker:
        marker = lambda x: x == v_type.oas_type()

    label = v_type.full_type()
    logger.debug(f"Tree for {label} is building...")
    model, threshold = build_tree(X_train, y_train, X_val, y_val, marker, label,
                                  metric, annotation, tree_lib, iterative_learning, print_metrics)
    logger.debug(f"Tree for {label} was built")
    y_pred_proba = model.predict_proba(test_pool)[:, 1]
    y_pred = np.where(y_pred_proba >= threshold, 1, 0)
    logger.info(format_confusion_matrix(make_binary_target(y_test, marker), y_pred))
    logger.info(f"Tree for {label} tested.")
    return ModelWrapper(v_type, model, annotation, threshold)


def make_models(input_dir: str, schema: str, chain_type: GeneralChainType,
                metric: str, iterative_learning: bool, print_metrics: bool,
                tree_types: str = None, tree_lib: str = None) -> Generator[ModelWrapper, None, None]:
    annotation = load_annotation(schema, chain_type.kind())
    X, y = read_any_dataset(input_dir, annotation)
    X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.07, shuffle=True, random_state=42)
    log_data_stats(X_, y_, X_test, y_test)
    X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.07, shuffle=True, random_state=42)
    used_types = tree_types.split(",") if tree_types else chain_type.available_specific_types()
    logger.debug(f"Forests for types {used_types} will be built")
    test_pool = one_hot_encode(annotation, X_test, tree_lib, cat_features=X_test.columns.tolist())
    for v_type in used_types:
        yield make_model(X_train, y_train, X_val, y_val, test_pool, y_test, annotation,
                         chain_type.specific_type(v_type), metric, iterative_learning, print_metrics, tree_lib=tree_lib)


def configure_abstract_parser(parser: argparse.ArgumentParser):
    parser.add_argument('input', type=str, help='Path to directory where all .csv (or .csv.gz) are listed')
    parser.add_argument('output', type=str, help='Output models location')
    parser.add_argument('--iterative-learning', action='store_true', help='Iterative learning using data batches')
    parser.add_argument('--single-batch-learning', dest='iterative_learning', action='store_false')
    parser.set_defaults(iterative_learning=True)
    parser.add_argument('--schema', type=str, default="chothia", choices=['chothia'], help='Annotation schema')
    parser.add_argument('--metric', type=str, default="youdens",
                        choices=['youdens', 'matthews'], help='Threshold optimized metric')
    parser.add_argument('--tree-lib', type=str, default="catboost",
                        choices=['catboost', 'sklearn'], help='Decision tree library')
    parser.add_argument('--print-metrics', action='store_true', help='Print learning metrics')
    parser.set_defaults(print_metrics=False)
    parser.add_argument('--types', type=str, default=None, help='Build only specified types')
