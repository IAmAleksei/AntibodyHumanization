import math
from typing import List, Tuple, Callable

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics


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


def find_optimal_threshold(metric_name: str, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, float]:
    metric_function = get_metric_function(metric_name)
    count = len(y_true)
    y_ = list(zip(y_pred_proba, y_true))
    y_.sort()
    ones_count = np.count_nonzero(y_true)
    confusion_matrix = [[0, count - ones_count], [0, ones_count]]  # [[tn, fp], [fn, tp]]
    best_threshold, best_score = 0.0, metric_function(confusion_matrix)
    threshold_points = [best_threshold]
    score_points = [best_score]
    for i in range(count):
        value = y_[i][1]
        confusion_matrix[value][1] -= 1
        confusion_matrix[value][0] += 1
        if i == count - 1 or y_[i][0] < y_[i + 1][0]:
            threshold = (y_[i][0] + y_[i + 1][0]) / 2 if i < count - 1 else 1.1
            score = metric_function(confusion_matrix)
            threshold_points.append(threshold)
            score_points.append(score)
            if score > best_score:
                best_threshold, best_score = threshold, score
    plt.plot(threshold_points, score_points)
    plt.show()
    return best_threshold, best_score


def plot_roc_auc(y_true, y_pred_proba):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_proba)
    auc = metrics.roc_auc_score(y_true, y_pred_proba)
    plt.plot(fpr, tpr, label="AUC = " + str(round(auc, 4)))
    plt.legend(loc=4)
    plt.show()
