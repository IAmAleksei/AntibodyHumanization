import argparse
from typing import List

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import config_loader
from humanization.annotations import load_annotation
from humanization.dataset import make_binary_target
from humanization.dataset_preparer import read_any_heavy_dataset
from humanization.models import ModelWrapper, HeavyChainType, save_model
from humanization.stats import plot_roc_auc, find_optimal_threshold
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Heavy chain RF")


def build_tree(X, y, v_type=1) -> CatBoostClassifier:
    y = make_binary_target(y, v_type)
    logger.debug(f"Dataset for V{v_type} tree contains {np.count_nonzero(y == 1)} positive samples")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

    model = CatBoostClassifier(iterations=250,
                               depth=3,
                               learning_rate=1,
                               loss_function='Logloss',
                               verbose=False)
    model.fit(X_train, y_train,
              cat_features=X.columns.tolist(),
              eval_set=(X_val, y_val))

    return model


def build_trees(input_dir: str, schema: str, annotated_data: bool) -> List[ModelWrapper]:
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
        model = build_tree(X_, y_, v_type)
        logger.debug(f"Tree for V{v_type} was built")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_test_binary = make_binary_target(y_test, v_type)
        threshold, metric_score = find_optimal_threshold("youdens", y_test_binary, y_pred_proba)
        logger.info(f"Optimal threshold is {threshold}, metric score = {metric_score}")
        plot_roc_auc(y_test_binary, y_pred_proba)
        y_pred = np.where(y_pred_proba >= threshold, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred).ravel()
        logger.info(f"Tree for V{v_type} tested.")
        logger.info(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}. "
                    f"Recall={round(tp / (tp + fn), 5)}. "
                    f"Precision={round(tp / (tp + fp), 5)}. "
                    f"Accuracy={round((tp + tn) / (tp + tn + fp + fn), 5)}")
        wrapped_model = ModelWrapper(HeavyChainType(str(v_type)), model, annotation, threshold)
        models.append(wrapped_model)
    return models


def main(input_dir, schema, output_dir, annotated_data):
    wrapped_models = build_trees(input_dir, schema, annotated_data)
    for wrapped_model in wrapped_models:
        save_model(output_dir, wrapped_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''RF generator''')
    parser.add_argument('input', type=str, help='Path to file where all .csv (or .csv.gz) are listed')
    parser.add_argument('schema', type=str, help='Annotation schema')
    parser.add_argument('output', type=str, help='Output models location')
    parser.add_argument('--annotated-data', action='store_true', help='Data is annotated')
    parser.add_argument('--raw-data', dest='annotated_data', action='store_false')
    parser.set_defaults(annotated_data=True)
    args = parser.parse_args()

    main(input_dir=args.input,
         schema=args.schema,
         output_dir=args.output,
         annotated_data=args.annotated_data)
