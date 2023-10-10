import json
import os.path
from collections import defaultdict

from termcolor import colored
from typing import Dict, List

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


def process_sequence(models_dir: str, name: str, seq_dict: Dict[str, str], target_model_metric: float):
    def remove_minuses(key):
        seq_dict[key] = seq_dict[key].replace('-', '')

    remove_minuses('sequ')
    remove_minuses('ther')
    remove_minuses('hu_m')

    # chain_type = ChainType.from_full_type(seq_dict['type'])
    for i in range(1, 8):
        _, res1, its = humanizer.process_sequences(models_dir, [(name, seq_dict['sequ']), ],
                                              ChainType.from_full_type(f'HV{i}'), target_model_metric)[0]
        seq_dict[f"tl_{i}"] = res1
        seq_dict[f"ti_{i}"] = its
    # _, res2 = reverse_humanizer.process_sequences(models_dir, [(name, seq_dict['sequ']), ], chain_type, target_model_metric)[0]
    # seq_dict["tl_2"] = res2


def analyze(seq_dict: Dict[str, str]):
    def print_hamming_distance(key2):
        seq1, seq2, seq3 = seq_dict['ther'], seq_dict[key2], seq_dict['sequ']
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
        print("".join(snd_str), key2, eq, group, delete_errors, insert_errors)
        # print(f"Distance between `{key1}` and `{key2}`:", "eq =", eq, "; group =", group)
        # print()

    # def analyze_its(i):
    #     details: List[IterationDetails] = seq_dict[f"ti_{i}"]
    #     for itr in details:
    #         if itr.change:
    #             print(itr.change.position, itr.change.old_aa, itr.change.aa)

    print(seq_dict['ther'])
    print_hamming_distance('sequ')
    print_hamming_distance('hu_m')
    for i in range(1, 8):
        print_hamming_distance(f'tl_{i}')
        # analyze_its(i)
    # print_hamming_distance('ther', 'tl_2')


def main():
    models_dir = os.path.abspath("../models")
    with open('thera_antibodies.json', 'r') as fp:
        samples = json.load(fp)

    for antibody in samples[1:]:
        name = antibody["name"]
        print(f'Processing antibody {name}')
        process_sequence(models_dir, name, antibody["heavy"], 0.99)
        # process_sequence(models_dir, name, antibody["light"], 0.99)
    print()
    print("Analyze")
    for antibody in samples[1:]:
        name = antibody["name"]
        print('---')
        print(f'Processing antibody {name}')
        analyze(antibody["heavy"])


if __name__ == '__main__':
    main()

