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
    chs = [0, 0, 0]
    for i, aa in enumerate(seq):
        c_aa = aa
        if aa != wild[i] and aa != "X":
            if aa == thera[i]:
                c_aa = colored(aa, 'green')
                chs[2] += 1
            elif same_group_aa(aa, thera[i]):
                c_aa = colored(aa, 'blue')
                chs[1] += 1
            else:
                c_aa = colored(aa, 'red')
                chs[0] += 1
        colored_seq.append(c_aa)
    return "".join(colored_seq), "/".join(map(str, chs))


def print_reference(v_gene_scorer, annotation, wild, thera):
    annotated_wild = annotate_single(wild, annotation, GeneralChainType.HEAVY)
    ref = v_gene_scorer.query(annotated_wild, count=1)[0]
    ref_seq = [aa1 for aa1, aa2 in zip(ref[0], annotated_wild) if aa2 != 'X']
    colored_seq, chs = get_colored_seq(ref_seq, wild, thera)
    print(colored_seq, f"Ref {ref[2]}", chs)


def main(files, dataset_dir, only_first):
    annotation = load_annotation("imgt_humatch", ChainKind.HEAVY)
    v_gene_scorer = build_v_gene_scorer(annotation, dataset_dir)
    seqs = defaultdict(list)
    for file in files:
        for seq in SeqIO.parse(file, 'fasta'):
            parts = seq.name.split("_")
            name = parts[0]
            tool = parts[1] if len(parts) < 3 else parts[2]
            seqs[name].append((tool, str(seq.seq).replace('X', '')))
    for mab, lst in seqs.items():
        thera = next(seq for way, seq in lst if "Therap." == way)
        wild = next(seq for way, seq in lst if "Wild" == way)
        print()
        print(mab)
        if len(thera) != len(wild):
            print("Skipped")
            continue
        print(wild, "Wild")
        print_reference(v_gene_scorer, annotation, wild, thera)
        used = ["Therap.", "Wild"]
        for way, seq in lst:
            if way in used:
                continue
            if only_first:
                used.append(way)
            colored_seq, chs = get_colored_seq(seq, wild, thera)
            print(colored_seq, way, chs)
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
    parser.add_argument('--only-first', action='store_true', default=False,
                        help='Process only first seq from tool')
    args = parser.parse_args()
    main(args.files, args.dataset, args.only_first)
