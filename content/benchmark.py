import json
import os.path
from termcolor import colored
from typing import Dict

from humanization import humanizer, reverse_humanizer
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

    chain_type = ChainType.from_full_type(seq_dict['type'])
    _, res1 = humanizer.process_sequences(models_dir, [(name, seq_dict['sequ']), ], chain_type, target_model_metric)[0]
    seq_dict["tl_1"] = res1
    # _, res2 = reverse_humanizer.process_sequences(models_dir, [(name, seq_dict['sequ']), ], chain_type, target_model_metric)[0]
    # seq_dict["tl_2"] = res2


def analyze(seq_dict: Dict[str, str]):
    def print_hamming_distance(key1, key2):
        seq1, seq2 = seq_dict[key1], seq_dict[key2]
        if len(seq1) != len(seq2):
            print(f"Different lengths between `{key1}` and `{key2}`")
            print()
            return
        print(seq1)
        eq, group = 0, 0
        snd_str = []
        for c1, c2 in zip(seq1, seq2):
            if c1 == c2:
                eq += 1
                snd_str.append(colored(c2, 'green'))
            elif same_group_aa(c1, c2):
                group += 1
                snd_str.append(colored(c2, 'blue'))
            else:
                snd_str.append(colored(c2, 'red'))
        print("".join(snd_str))
        print(f"Distance between `{key1}` and `{key2}`:", "eq =", eq, "; group =", group)
        print()

    print_hamming_distance('ther', 'sequ')
    print_hamming_distance('ther', 'hu_m')
    print_hamming_distance('ther', 'tl_1')
    # print_hamming_distance('ther', 'tl_2')


def main():
    models_dir = os.path.abspath("../models")
    with open('thera_antibodies.json', 'r') as fp:
        samples = json.load(fp)

    for antibody in samples:
        name = antibody["name"]
        print(f'Processing antibody {name}')
        process_sequence(models_dir, name, antibody["heavy"], 0.99)
        # process_sequence(models_dir, name, antibody["light"], 0.99)
    print()
    print("Analyze")
    for antibody in samples:
        name = antibody["name"]
        print('---')
        print(f'Processing antibody {name}')
        analyze(antibody["heavy"])


if __name__ == '__main__':
    main()

