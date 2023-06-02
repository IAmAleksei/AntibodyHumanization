import argparse
from typing import List, Tuple

import numpy as np
from catboost import CatBoostClassifier, Pool
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import config_loader
from humanization.annotations import load_annotation
from humanization.dataset import make_binary_target
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
    plot_thresholds(threshold_points, metric, threshold, metric_score, axis[0])
    plot_roc_auc(y, y_pred_proba, axis[1])
    return threshold, metric_score


def plot_metrics(name: str, train_metrics: dict, val_metrics: dict, ax):
    ax.set_title(name)
    plot_comparison(name, train_metrics, "Train", val_metrics, "Validation", ax)


def build_tree(X, y_raw, v_type: int, metric: str) -> Tuple[CatBoostClassifier, float]:
    y = make_binary_target(y_raw, v_type)
    logger.debug(f"Dataset for V{v_type} tree contains {np.count_nonzero(y == 1)} positive samples")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=True)

    model = CatBoostClassifier(iterations=300,
                               depth=3,
                               loss_function='Logloss',
                               learning_rate=0.06,
                               verbose=25)
    model.fit(X_train, y_train, cat_features=X.columns.tolist(), eval_set=(X_val, y_val))

    train_pool = Pool(X_train, y_train, cat_features=X.columns.tolist())
    val_pool = Pool(X_val, y_val, cat_features=X.columns.tolist())
    train_metrics = model.eval_metrics(data=train_pool, metrics=['Logloss', 'AUC'])
    val_metrics = model.eval_metrics(data=val_pool, metrics=['Logloss', 'AUC'])
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]

    figure, axis = plt.subplots(2, 2, figsize=(9, 9))
    plt.suptitle(f'Tree IGHV{v_type}')
    plot_metrics('Logloss', train_metrics, val_metrics, axis[0, 0])
    plot_metrics('AUC', train_metrics, val_metrics, axis[0, 1])
    threshold, metric_score = get_threshold(metric, y_val, y_val_pred_proba, axis[1, :])
    logger.info(f"Optimal threshold is {threshold}, metric score = {metric_score}")
    plt.tight_layout()
    plt.show()
    return model, threshold


def build_trees(input_dir: str, schema: str, metric: str, annotated_data: bool) -> List[ModelWrapper]:
    annotation = load_annotation(schema)
    X, y = read_any_heavy_dataset(input_dir, annotated_data, annotation)
    X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)
    logger.info(f"Train dataset: {X_.shape[0]} rows")
    logger.debug(f"Statistics:\n{y_.value_counts()}")
    logger.info(f"Test dataset: {X_test.shape[0]} rows")
    logger.debug(f"Statistics:\n{y_test.value_counts()}")
    models = []
    for v_type in range(1, 8):
        logger.debug(f"Tree for V{v_type} is building...")
        model, threshold = build_tree(X_, y_, v_type, metric)
        logger.debug(f"Tree for V{v_type} was built")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = np.where(y_pred_proba >= threshold, 1, 0)
        tn, fp, fn, tp = confusion_matrix(make_binary_target(y_test, v_type), y_pred).ravel()
        logger.info(f"Tree for V{v_type} tested.")
        logger.info(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}. "
                    f"Recall={round(tp / (tp + fn), 5)}. "
                    f"Precision={round(tp / (tp + fp), 5)}. "
                    f"Accuracy={round((tp + tn) / (tp + tn + fp + fn), 5)}")
        wrapped_model = ModelWrapper(HeavyChainType(str(v_type)), model, annotation, threshold)
        models.append(wrapped_model)
    return models


def main(input_dir, schema, metric, output_dir, annotated_data):
    wrapped_models = build_trees(input_dir, schema, metric, annotated_data)
    for wrapped_model in wrapped_models:
        save_model(output_dir, wrapped_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''RF generator''')
    parser.add_argument('input', type=str, help='Path to directory where all .csv (or .csv.gz) are listed')
    parser.add_argument('output', type=str, help='Output models location')
    parser.add_argument('--annotated-data', action='store_true', help='Data is annotated')
    parser.add_argument('--raw-data', dest='annotated_data', action='store_false')
    parser.set_defaults(annotated_data=True)
    parser.add_argument('--schema', type=str, default="chothia", help='Annotation schema')
    parser.add_argument('--metric', type=str, default="youdens", help='Threshold optimized metric')
    args = parser.parse_args()

    main(input_dir=args.input,
         schema=args.schema,
         metric=args.metric,
         output_dir=args.output,
         annotated_data=args.annotated_data)
