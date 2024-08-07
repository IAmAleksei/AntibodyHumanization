import math
from typing import List, Tuple, Callable, NoReturn, Optional

import numpy as np
from sklearn import metrics

from humanization.common import config_loader
from humanization.common.utils import configure_logger


config = config_loader.Config()
logger = configure_logger(config, "Stats calculator")


def calc_matthews_correlation(confusion_matrix: List[List[int]]) -> float:
    if confusion_matrix[0][0] == 0 or confusion_matrix[1][1] == 0:
        return -1.0  # Bad corner case
    n = confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[1][0] + confusion_matrix[0][1]
    s = (confusion_matrix[1][1] + confusion_matrix[1][0]) / n
    p = (confusion_matrix[1][1] + confusion_matrix[0][1]) / n
    return (confusion_matrix[1][1] / n - s * p) / math.sqrt(p * s * (1 - s) * (1 - p))


def calc_youdens_j_statistic(confusion_matrix: List[List[int]]) -> float:
    if confusion_matrix[0][0] == 0 or confusion_matrix[1][1] == 0:
        return -1.0  # Bad corner case
    sensitivity = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
    specificity = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    return sensitivity + specificity - 1


def get_metric_function(metric_name: str) -> Callable[[List[List[int]]], float]:
    if metric_name == "matthews":
        return calc_matthews_correlation
    elif metric_name == "youdens":
        return calc_youdens_j_statistic
    else:
        raise RuntimeError("Unrecognized metric name")


def proba_distribution(values: np.ndarray) -> np.ndarray:
    return np.histogram(values, bins=10, range=(0.0, 1.0))[0]


def print_distribution(y_pred_proba: np.ndarray, y_true: Optional[np.ndarray] = None) -> NoReturn:
    total_counts = proba_distribution(y_pred_proba)
    ones_counts = None
    if y_true is not None:
        ones_counts = proba_distribution(y_pred_proba[y_true == 1])
    str_dists = []
    for i in range(10):
        str_count = f"{ones_counts[i]} / " if ones_counts is not None else ""
        str_dists.append(f"{round(i * 0.1, 1)} - {round((i + 1) * 0.1, 1)}: {str_count}{total_counts[i]}")
    logger.info("Sample distribution:\n" + "\n".join(str_dists))


def brute_force_threshold(metric_name: str, y_true: np.ndarray,
                          y_pred_proba: np.ndarray) -> List[Tuple[float, float]]:
    print_distribution(y_pred_proba, y_true)
    metric_function = get_metric_function(metric_name)
    count = len(y_true)
    y_ = list(zip(y_pred_proba, y_true))
    y_.sort()
    ones_count = np.count_nonzero(y_true)
    confusion_matrix = [[0, count - ones_count], [0, ones_count]]  # [[tn, fp], [fn, tp]]
    best_threshold, best_score = 0.0, metric_function(confusion_matrix)
    threshold_points = [(best_threshold, best_score)]
    for i in range(count):
        value = y_[i][1]
        confusion_matrix[value][1] -= 1
        confusion_matrix[value][0] += 1
        if i == count - 1 or y_[i][0] < y_[i + 1][0]:
            threshold = (y_[i][0] + y_[i + 1][0]) / 2 if i < count - 1 else 1.1
            score = metric_function(confusion_matrix)
            threshold_points.append((threshold, score))
    return threshold_points


def find_optimal_threshold(threshold_points: List[Tuple[float, float]]) -> Tuple[float, float]:
    return max(threshold_points, key=lambda x: x[1])


def plot_thresholds(threshold_points: List[Tuple[float, float]], metric: str,
                    best_threshold: float, metric_score: float, ax) -> NoReturn:
    x, y = zip(*threshold_points)
    ax.set_title('Thresholds')
    ax.plot(x, y, color='#160773')
    ax.plot(best_threshold, metric_score, marker='o', color='#7C005A',
            label=f"{metric} score = {round(metric_score, 2)}")
    ax.legend(loc=4)


def plot_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray, ax):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_proba)
    auc = metrics.roc_auc_score(y_true, y_pred_proba)
    ax.set_title('ROC AUC')
    ax.plot(fpr, tpr, color="#160773", label="AUC = " + str(round(auc, 4)))
    ax.legend(loc=4)


def plot_comparison(key: str, metrics1: dict, name1: str, metrics2: dict, name2: str, ax):
    if len(metrics1) > 0:
        ax.plot(np.array(metrics1[key]), color="#7C005A", label=name1)
    ax.plot(np.array(metrics2[key]), color="#006B53", label=name2)
    ax.legend(loc=1)
