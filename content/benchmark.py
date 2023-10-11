import argparse
import json
import os.path
from collections import defaultdict
from typing import Dict, List

from termcolor import colored

from humanization import humanizer
from humanization.abstract_humanizer import IterationDetails
from humanization.annotations import ChainType


def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


AA_GROUPS = {
    'positive': 'KRH',
    'negative': 'DE',
    'hydrophobic': 'VMILA',
    'hydrophilic': 'QNST',
    'aromatic': 'WFY',
    'others': 'CGP',
}


def same_group_aa(aa1, aa2):
    for values in AA_GROUPS.values():
        if aa1 in values and aa2 in values:
            return True
    return False


def remove_minuses(seq_dict, key):
    seq_dict[key] = seq_dict[key].replace('-', '')


SUMMARY = []


def analyze(seq_dict: Dict[str, str]):
    def remove_xs(sss):
        return sss.replace('X', '')

    remove_minuses(seq_dict, 'sequ')
    remove_minuses(seq_dict, 'ther')
    remove_minuses(seq_dict, 'hu_m')

    if len(remove_xs(seq_dict['tl_1'])) != len(seq_dict['ther']) or len(seq_dict['ther']) != len(seq_dict['sequ']):
        print('Bad alignment')
        return

    diff_poses = set([i for i in range(len(seq_dict['ther'])) if seq_dict['ther'][i] != seq_dict['sequ'][i]])

    def print_hamming_distance(key2):
        seq1, seq2, seq3 = seq_dict['ther'], remove_xs(seq_dict[key2]), seq_dict['sequ']
        if len(seq1) != len(seq2):
            print(f"Different lengths between `{key2}`")
            print()
            return
        eq, group = 0, 0
        snd_str = []
        insert_errors, delete_errors = defaultdict(int), defaultdict(int)
        for i in range(len(seq1)):
            c1, c2, c3 = seq1[i], seq2[i], seq3[i]
            if c1 == c2:
                eq += 1
                snd_str.append(colored(c2, 'green'))
            elif same_group_aa(c1, c2):
                group += 1
                snd_str.append(colored(c2, 'blue'))
            else:
                snd_str.append(colored(c2, 'red'))
                if same_group_aa(c1, c3):
                    delete_errors[c3] += 1
                insert_errors[c2] += 1
        beautiful_str = "".join(snd_str)
        print(beautiful_str, key2, eq, group)
        return eq, beautiful_str

    def analyze_its(i):
        sq = seq_dict[f"tl_{i}"]
        details: List[IterationDetails] = seq_dict[f"ti_{i}"]
        result = ""
        for itr in details:
            if itr.change:
                changed_pos = itr.change.position - sq[:itr.change.position].count('X')
                ch_ther = seq_dict['ther'][changed_pos] != seq_dict['sequ'][changed_pos]
                result += "+" if ch_ther else "-"
        print(result)
        return result, details[-1]

    print(seq_dict['ther'], len(diff_poses))
    print_hamming_distance('sequ')
    print_hamming_distance('hu_m')
    max_eq = 0
    max_r = "", ""
    for i in range(1, 8):
        eq, b = print_hamming_distance(f'tl_{i}')
        res, last_det = analyze_its(i)
        if eq > max_eq:
            max_r = b, res, last_det
            max_eq = eq
    SUMMARY.append((seq_dict['sequ'], seq_dict['ther'], max_r))
    # print_hamming_distance('ther', 'tl_2')


def analyze_summary():
    mx_len = max(len(SUMMARY[i][2][1]) for i in range(len(SUMMARY)))
    positions_correct = [0 for _ in range(mx_len)]
    positions_count = [0 for _ in range(mx_len)]
    v_gene_scores = []
    for t in SUMMARY:
        sequ, ther, (beautiful_result, result, last_det) = t
        print("Seq.", sequ)
        print("Exp.", ther)
        print("Hum.", beautiful_result)
        metric = round(result.count('+') / len(result), 2)
        print(metric, result, last_det.v_gene_score)
        if last_det.v_gene_score is not None:
            v_gene_scores.append(last_det.v_gene_score)
        for i, r in enumerate(result):
            if r == "+":
                positions_correct[i] += 1
            positions_count[i] += 1
    sm_correct, sm_count = sum(positions_correct), sum(positions_count)
    correct_fraction = [round(positions_correct[i] / positions_count[i] * 100, 1) for i in range(mx_len)]
    print("Correct positions", sm_correct, "of", sm_count, f"({round(sm_correct / sm_count * 100, 1)}%)")
    print("Count of correct", positions_correct, "and of all", positions_count)
    print("Fraction of correct", correct_fraction)
    if len(v_gene_scores) > 0:
        print(f"Average v gene score", sum(v_gene_scores) / len(v_gene_scores))
    else:
        print("V gene score not calculated")


def main(models_dir, dataset_dir):
    models_dir = os.path.abspath(models_dir)
    with open('thera_antibodies.json', 'r') as fp:
        samples = json.load(fp)

    for i in range(1, 8):
        print(f'Processing HV{i}')
        prep_seqs = []
        for antibody in samples:
            remove_minuses(antibody['heavy'], 'sequ')
            prep_seqs.append((antibody['name'], antibody['heavy']['sequ']))
        ans = humanizer.process_sequences(
            models_dir, prep_seqs, ChainType.from_full_type(f'HV{i}'),
            0.9, dataset_file=dataset_dir, annotated_data=True, aligned_result=True
        )
        for j, antibody in enumerate(samples):
            _, res1, its = ans[j]
            antibody['heavy'][f"tl_{i}"] = res1
            antibody['heavy'][f"ti_{i}"] = its

    print()
    print("Analyze")
    for antibody in samples:
        print('---')
        print(f'Processing antibody {antibody["name"]} {antibody["heavy"]["type"]}')
        analyze(antibody["heavy"])

    print()
    print("Summary")
    analyze_summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Benchmark direct humanizer''')
    parser.add_argument('--models', type=str, default="../models", help='Path to directory with models')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    args = parser.parse_args()
    main(models_dir=args.models, dataset_dir=args.dataset)
