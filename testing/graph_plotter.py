import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

prettifier_m = {
    "WDist": "Количество замен",
    "HuVGS": "V-gene score с ближайшим человеческим референсом",
    "WBerta": "AntiBERTa2",
    "WAbody": "AbodyBuilder2",
    "WAblang": "AbLang2",
    "OASId": "BioPhi OASis Identity",
    "RMSD": "RMSD",
}

prettifier_t = {
    "Wild.": "Животное",
    "Therap.": "Терапевт.",
    "i": "РИ\n(1 замена)",
    "i3": "РИ\n(3 замены)",
    "HuMab": "Hu-mAb",
    "Sapiens1": "BioPhi\nSapiens",
    "Sapiens3": "BioPhi\nSapiens 3x",
}


def main(metric, filename, with_wild, no_thera):
    df = pd.read_csv(filename)
    types = ["i", "i3", "HuMab", "Sapiens1", "Sapiens3"]
    if not no_thera:
        types = ["Therap."] + types
    if with_wild:
        types = ["Wild."] + types
    labels = [prettifier_t[t] for t in types]
    boxes = [df[df["Type"] == t][metric] for t in types]
    plt.figure(figsize=(4 + len(types), 6), dpi=200)
    plt.rcParams.update({'font.size': 16})
    plt.title(prettifier_m[metric])
    if metric == "HuVGS":
        plt.axhline(y=0.85, color='r', alpha=0.35, linestyle='--')
    plt.boxplot(boxes, labels=labels, whis=(0, 100))
    plt.tight_layout()
    for i, t in enumerate(types):
        xs = np.random.normal(i + 1, 0.02, len(boxes[i]))
        plt.scatter(xs, boxes[i], c='r', alpha=0.2)
    plt.savefig(metric + ".jpg")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Plotter''')
    parser.add_argument('metric', type=str, help='Metric to visualize')
    parser.add_argument('filename', type=str, help='Input .csv')
    parser.add_argument('--with-wild', action='store_true', default=False, help='Include wild type')
    parser.add_argument('--no-thera', action='store_true', default=False, help='Include therapeutic type')
    args = parser.parse_args()
    main(args.metric, args.filename, args.with_wild, args.no_thera)
