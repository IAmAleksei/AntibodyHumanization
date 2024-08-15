import argparse
from collections import defaultdict
from termcolor import colored

from Bio import SeqIO

from humanization.common.annotations import load_annotation, ChainKind, annotate_single, GeneralChainType
from humanization.common.v_gene_scorer import build_v_gene_scorer

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


def get_colored_seq(seq, wild, thera):
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
    return colored_seq


def main(files, dataset_dir, only_first):
    annotation = load_annotation("chothia", ChainKind.HEAVY)
    v_gene_scorer = build_v_gene_scorer(annotation, dataset_dir)
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
        reference = v_gene_scorer.query(annotate_single(wild, annotation, GeneralChainType.HEAVY), count=1)
        print("".join(get_colored_seq(reference[0][0], wild, thera)), f"Ref {reference[0][2]}")
        used = ["Therap.", "Wild"]
        for way, seq in lst:
            if way in used:
                continue
            if only_first:
                used.append(way)
            print("".join(get_colored_seq(seq, wild, thera)), way)
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
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    parser.add_argument('--only-first', type=bool, action='store_true', default=False,
                        help='Process only first seq from tool')
    args = parser.parse_args()
    main(args.files, args.dataset, args.only_first)
