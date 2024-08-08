import argparse
from collections import defaultdict
from termcolor import colored

from Bio import SeqIO


AA_GROUPS = {
    'positive': 'KRH',
    'negative': 'DE',
    'hydrophobic': 'VMILA',
    'hydrophilic': 'QNST',
    'aromatic': 'WFY',
    'others': 'CGP',
}


def same_group_aa(aa1, aa2):
    return any(aa1 in values and aa2 in values for values in AA_GROUPS.values())


def main(files):
    seqs = defaultdict(list)
    for file in files:
        for seq in SeqIO.parse(file, 'fasta'):
            parts = seq.name.split("_")
            seqs[parts[0]].append((parts[2], str(seq.seq).replace('X', '')))
    for mab, lst in seqs.items():
        thera = next(seq for way, seq in lst if "Therap." == way)
        wild = next(seq for way, seq in lst if "Wild" == way)
        print()
        print(mab)
        if len(thera) != len(wild):
            print("Skipped")
            continue
        print(wild, "Wild")
        for way, seq in lst:
            if way in ["Therap.", "Wild"]:
                continue
            colored_seq = []
            for i, aa in enumerate(seq):
                c_aa = aa
                if aa != wild[i]:
                    if aa == thera[i]:
                        c_aa = colored(aa, 'green')
                    elif same_group_aa(aa, thera[i]):
                        c_aa = colored(aa, 'blue')
                    else:
                        c_aa = colored(aa, 'red')
                colored_seq.append(c_aa)
            print("".join(colored_seq), way)
        colored_thera = []
        for i, aa in enumerate(thera):
            c_aa = aa
            if aa != wild[i]:
                c_aa = colored(aa, 'yellow')
            colored_thera.append(c_aa)
        print("".join(colored_thera), "Thera")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Comparator of sequences''')
    parser.add_argument('files', metavar='file', type=str, nargs='+', help='Name of files')
    args = parser.parse_args()
    main(args.files)
