import argparse
from typing import List

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import config_loader
from humanization.dataset import read_prepared_heavy_dataset, make_binary_target
from humanization.models import ModelWrapper, HeavyChainType, save_model
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


def build_trees() -> List[ModelWrapper]:
    X, y = read_prepared_heavy_dataset()
    X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)
    logger.info(f"Train dataset: {X_.shape[0]} rows")
    logger.debug(f"Statistics:\n{y_.value_counts()}")
    logger.info(f"Test dataset: {X_test.shape[0]} rows")
    logger.debug(f"Statistics:\n{y_test.value_counts()}")
    models = []
    for v_type in range(1, 8):
        logger.debug(f"Tree for V{v_type} is building...")
        model = build_tree(X_, y_, v_type)
        y_pred = model.predict(X_test)
        logger.debug(f"Tree for V{v_type} was built")
        tn, fp, fn, tp = confusion_matrix(make_binary_target(y_test, v_type), y_pred).ravel()
        logger.info(f"Tree for V{v_type} tested.")
        logger.info(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}. "
                    f"Recall={round(tp / (tp + fn), 5)}. "
                    f"Precision={round(tp / (tp + fp), 5)}. "
                    f"Accuracy={round((tp + tn) / (tp + tn + fp + fn), 5)}")
        wrapped_model = ModelWrapper(HeavyChainType(str(v_type)), model)
        models.append(wrapped_model)
    return models


def main(output_directory):
    wrapped_models = build_trees()
    for wrapped_model in wrapped_models:
        save_model(output_directory, wrapped_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''RF generator''')
    parser.add_argument('input', type=str, help='Path to file where all .csv (or .csv.gz) are listed')
    parser.add_argument('output', type=str, help='Output models location')
    args = parser.parse_args()

    main(args.output)
