import argparse
from collections import defaultdict

from Bio import SeqIO

from humanization.algorithms import greedy_humanizer
from humanization.common import config_loader
from humanization.common.annotations import annotate_single, GeneralChainType, HumatchNumbering
from humanization.common.utils import configure_logger
from humanization.common.v_gene_scorer import build_v_gene_scorer
from humanization.humanness_calculator.model_wrapper import load_all_models

config = config_loader.Config()
logger = configure_logger(config, "xxx")


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

def remove_xs(seq):
    return "".join([x for x in seq if x != 'X'])

def identity(seq1, seq2):
    return sum([1 for c1, c2 in zip(seq1, seq2) if c1 == c2]) / max(len(seq1), len(seq2))

def main(file, models_dir, dataset, wild_dataset):
    annotation = HumatchNumbering()
    seqs = defaultdict(list)
    for seq in SeqIO.parse(file, 'fasta'):
        orig_seq = str(seq.seq)
        annotated_seq = annotate_single(orig_seq, annotation, GeneralChainType.HEAVY)
        seqs[seq.name] = annotated_seq
        assert len(orig_seq) == len(remove_xs(annotated_seq))
    if models_dir is not None:
        models = load_all_models(models_dir, GeneralChainType.HEAVY)
        v_gene_scorer = build_v_gene_scorer(annotation, dataset)
        wild_v_gene_scorer = build_v_gene_scorer(annotation, wild_dataset)
        our_seqs = {}
        for seq_name, seq in seqs.items():
            if not seq_name.endswith('Wild'):
                continue
            result = greedy_humanizer.process_sequences(
                v_gene_scorer, models, wild_v_gene_scorer, [("_", remove_xs(seq))],
                aligned_result=True, candidates_count=1)[0][1]
            our_seqs[seq_name.replace('Wild', 'Our')] = result
            print(result)
        seqs = seqs | our_seqs
    for t in ["HuMab", "Sapiens1", "Sapiens3", "Our"]:
        non_homological_number = [0] * len(annotation.positions)
        homological_number = [0] * len(annotation.positions)
        identities_with_wild = []
        identities_with_therap = []
        for seq_name, seq in seqs.items():
            if not seq_name.endswith(t):
                continue
            wild_seq = seqs[seq_name.replace(t, 'Wild')]
            therap_seq = seqs[seq_name.replace(t, 'Therap')]
            for i in range(len(annotation.positions)):
                if seq[i] == wild_seq[i]:
                    continue
                if same_group_aa(seq[i], therap_seq[i]):
                    homological_number[i] += 1
                else:
                    non_homological_number[i] += 1
            identities_with_wild.append(identity(remove_xs(seq), remove_xs(wild_seq)))
            identities_with_therap.append(identity(remove_xs(seq), remove_xs(therap_seq)))
        print("--------")
        print("Type:", t)
        print("Non-homological changes:", non_homological_number)
        print("Homological changes:", homological_number)
        print(f"Identities {t} with wild:", ",".join(map(str, identities_with_wild)))
        print(f"Identities {t} with therap:", ",".join(map(str, identities_with_therap)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Comparator of sequences''')
    parser.add_argument('db_file', type=str, help='Name of file for sequences')
    parser.add_argument('--models', type=str, required=False, help='Path to trained models')
    parser.add_argument('--dataset', type=str, required=False, help='')
    parser.add_argument('--wild-dataset', type=str, required=False, help='')
    args = parser.parse_args()
    main(args.db_file, args.models, args.dataset, args.wild_dataset)
