import argparse
import numpy as np
import os

from humanization.common import config_loader
from humanization.common.annotations import GeneralChainType, load_annotation, ChainKind
from humanization.common.utils import configure_logger
from humanization.dataset.dataset_reader import read_any_dataset
from humanization.dataset.one_hot_encoder import one_hot_encode
from humanization.humanness_calculator.model_wrapper import load_all_models
from humanization.humanness_calculator.stats import calc_learning_stats

config = config_loader.Config()
logger = configure_logger(config, "Ada analyzer")


def main(input_dir, schema, model_dir):
    model_wrappers = load_all_models(model_dir, GeneralChainType.HEAVY)
    annotation = load_annotation(schema, ChainKind.HEAVY)
    test_path = os.path.join(input_dir, "test")
    X_test, y_test = read_any_dataset(test_path, annotation)
    test_pool = one_hot_encode(annotation, X_test, lib=next(iter(model_wrappers.values())).library(), cat_features=X_test.columns.tolist())
    metrics = ['pr_auc', 'roc_auc', 'f1', 'matthews', 'balanced_accuracy']
    print([""] + metrics, sep=',')
    for key in model_wrappers.keys():
        model = model_wrappers[key]
        y_pred_proba = model.predict_proba(test_pool)[:, 1]
        y_pred = np.where(y_pred_proba >= model.threshold, 1, 0)
        learning_statistics = calc_learning_stats(y_test, y_pred_proba, y_pred)
        metrics_values = [learning_statistics[m] for m in metrics]
        print(key, *metrics_values, sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to directory where all .csv (or .csv.gz) are listed')
    parser.add_argument('--models', type=str, default='../sklearn_models2', help='Path to directory with models')
    parser.add_argument('--schema', type=str, default="imgt_humatch", choices=['chothia', 'imgt_humatch'], help='Annotation schema')
    args = parser.parse_args()

    main(args.input, args.schema, args.models)
