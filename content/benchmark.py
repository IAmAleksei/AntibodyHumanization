import argparse
import json
import os.path
from typing import Dict, List

from termcolor import colored

from humanization import humanizer, reverse_humanizer
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


def pretty_key(key):
    mp = {"sequ": "Original",
          "hu_m": "Hu-Mab  ",
          "ther": "Therap. "}
    if key in mp:
        return mp[key]
    return f"TClass-{key[-1]}"


# def get_diff_indirect(seq_dict, i):
#     pass


def analyze_its_direct(seq_dict, i):
    sq = seq_dict[f"tl_{i}"]
    details: List[IterationDetails] = seq_dict[f"ti_{i}"]
    result = ""
    for itr in details:
        if itr.change:
            changed_pos = itr.change.position - sq[:itr.change.position].count('X')
            ch_ther = seq_dict['ther'][changed_pos] != seq_dict['sequ'][changed_pos]
            result += "+" if ch_ther else "-"
    return result, details[-1]


def remove_xs(sss):
    return sss.replace('X', '')


def print_hamming_distance(seq_dict, key2, pluses=""):
    seq1, seq2, seq3 = seq_dict['ther'], remove_xs(seq_dict[key2]), seq_dict['sequ']
    if len(seq1) != len(seq2):
        print(f"Different lengths between `{key2}`")
        print()
        return -1, ""
    eq, group = 0, 0
    snd_str = []
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
    beautiful_str = "".join(snd_str)
    print(pretty_key(key2), beautiful_str, f"={eq}", f"~{group}", pluses)
    return eq, beautiful_str


def analyze(seq_dict: Dict[str, str]):

    remove_minuses(seq_dict, 'sequ')
    remove_minuses(seq_dict, 'ther')
    remove_minuses(seq_dict, 'hu_m')

    if len(seq_dict['ther']) != len(seq_dict['sequ']):
        print('Bad alignment')
        return

    diff_poses = set([i for i in range(len(seq_dict['ther'])) if seq_dict['ther'][i] != seq_dict['sequ'][i]])

    sequ_eq, _ = print_hamming_distance(seq_dict, 'sequ')
    print(pretty_key('ther'), seq_dict['ther'], f"Diff={len(diff_poses)}")
    print_hamming_distance(seq_dict, 'hu_m')
    max_eq = 0
    max_r = None
    for i in range(1, 8):
        if f'tl_{i}' not in seq_dict:
            continue
        if len(remove_xs(seq_dict[f'tl_{i}'])) != len(seq_dict['ther']):
            print('Bad alignment')
            continue
        res, last_det = analyze_its_direct(seq_dict, i)
        eq, b = print_hamming_distance(seq_dict, f'tl_{i}', res)
        if eq > max_eq:
            max_r = b, res, last_det, (eq - sequ_eq)
            max_eq = eq
    SUMMARY.append((seq_dict['sequ'], seq_dict['ther'], max_r))


def analyze_summary():
    mx_len = max(len(SUMMARY[i][2][1]) for i in range(len(SUMMARY)))
    positions_correct = [0 for _ in range(mx_len)]
    positions_count = [0 for _ in range(mx_len)]
    v_gene_scores = []
    for t in SUMMARY:
        sequ, ther, ress = t
        if ress is None:
            print("Skipped summary")
            continue
        beautiful_result, result, last_det, inc_eq = ress
        print(pretty_key("sequ"), sequ)
        print(pretty_key("ther"), ther)
        print(pretty_key("hu_m"), beautiful_result)
        print(result, f"VGeneScore={last_det.v_gene_score}", f"+{inc_eq}")
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


def main(models_dir, dataset_dir, humanizer_type):
    models_dir = os.path.abspath(models_dir)
    with open('thera_antibodies.json', 'r') as fp:
        samples = json.load(fp)

    if humanizer_type == "direct":
        for i in range(1, 8):
            print(f'Processing HV{i}')
            print("Humanizer type", humanizer_type)
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
            print('***')
            print(f'Processing antibody {antibody["name"]} {antibody["heavy"]["type"]}')
            analyze(antibody["heavy"])

        print()
        print("Summary")
        analyze_summary()
    else:
        for antibody in samples:
            heavy_chain = antibody['heavy']
            tp = heavy_chain['type']
            remove_minuses(heavy_chain, 'ther')
            remove_minuses(heavy_chain, 'sequ')
            seq = (antibody['name'], heavy_chain['sequ'])
            ans = reverse_humanizer.process_sequences(
                models_dir, [seq], ChainType.from_full_type(tp),
                0.5, target_v_gene_score=0.85, dataset_file=dataset_dir, annotated_data=True, aligned_result=True
            )[0]
            _, res1, its = ans
            res1 = remove_xs(res1)
            heavy_chain[f"tl_{tp}"] = res1
            eq, hum = print_hamming_distance(heavy_chain, f"tl_{tp}")
            if eq == -1:
                continue
            changes = 0
            ch_ther = 0
            for i in range(len(heavy_chain['ther'])):
                if res1[i] != heavy_chain['sequ'][i]:
                    changes += 1
                    if heavy_chain[f"ther"][i] != heavy_chain['sequ'][i]:
                        ch_ther += 1
            print(pretty_key("sequ"), heavy_chain['sequ'])
            print(pretty_key("ther"), heavy_chain['ther'])
            print(pretty_key("hu_m"), hum)
            print(f"VGeneScore={its[-1].v_gene_score}", f"Changes={changes}",
                  f"ChangesFracWithTher={round(ch_ther / changes * 100, 1)}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Benchmark direct humanizer''')
    parser.add_argument('--models', type=str, default="../models", help='Path to directory with models')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    parser.add_argument('--humanizer', type=str, default='direct', choices=["direct", "reverse"], help='Humanizer type')
    args = parser.parse_args()
    main(models_dir=args.models, dataset_dir=args.dataset, humanizer_type=args.humanizer)
