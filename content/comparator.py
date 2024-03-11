import argparse
from collections import defaultdict

import edit_distance
from Bio import SeqIO

from humanization import config_loader
from humanization.ablang_utils import get_ablang_embedding
from humanization.annotations import ChothiaHeavy, annotate_single, GeneralChainType
from humanization.antiberta_utils import get_antiberta_embedding, diff_embeddings
from humanization.biophi_utils import get_humanness, get_oasis_humanness
from humanization.immunebuilder_utils import get_immunebuilder_embedding
from humanization.sapiens_utils import get_sapiens_embedding
from humanization.utils import configure_logger
from humanization.v_gene_scorer import build_v_gene_scorer

config = config_loader.Config()
logger = configure_logger(config, "Comparator")


def optional_v_gene_score(v_gene_scorer, seq: str):
    aligned_seq = annotate_single(seq, ChothiaHeavy(), GeneralChainType.HEAVY)
    if aligned_seq is None:
        return -1.0
    return v_gene_scorer.query(aligned_seq)[0][1]


def main(files, dataset, oas_db):
    v_gene_scorer = build_v_gene_scorer(ChothiaHeavy(), dataset)
    seqs = defaultdict(list)
    for file in files:
        for seq in SeqIO.parse(file, 'fasta'):
            mab = seq.name.split("_")[0]
            way = seq.name.split("_")[2]
            seqs[mab].append((way, str(seq.seq).replace('X', '')))
    print("Seq", "Type", "ThDist", "WDist", "VGScore", "ThBerta", "WBerta", "ThAbody", "WAbody",
          "ThSap", "WSap", "ThAblang", "WAblang", sep=",")
    for mab, lst in seqs.items():
        print()
        print(mab)
        thera = None
        wild = None
        for way, seq in lst:
            if "Therap." == way:
                thera = seq
            if "Wild" == way:
                wild = seq
        if thera is None or wild is None:
            print("No therapeutic or wild seq")
            continue
        seq_emb_thera = get_antiberta_embedding(" ".join(thera))
        seq_emb_wild = get_antiberta_embedding(" ".join(wild))
        struct_emb_thera = get_immunebuilder_embedding(thera)
        struct_emb_wild = get_immunebuilder_embedding(wild)
        sap_emb_thera = get_sapiens_embedding(thera)
        sap_emb_wild = get_sapiens_embedding(wild)
        abl_emb_thera = get_ablang_embedding(thera)
        abl_emb_wild = get_ablang_embedding(wild)
        oasis_ident_thera = get_oasis_humanness(oas_db, thera)
        print(wild[:25] + "...", "Wild.", sep=",")
        print(thera[:25] + "...", "Therap.", "", edit_distance.SequenceMatcher(wild, thera).distance(),
              round(optional_v_gene_score(v_gene_scorer, thera), 2), "",
              round(diff_embeddings(seq_emb_thera, seq_emb_wild), 2), "",
              round(diff_embeddings(struct_emb_thera, struct_emb_wild), 2), "",
              round(diff_embeddings(sap_emb_thera, sap_emb_wild), 2), "",
              round(diff_embeddings(abl_emb_thera, abl_emb_wild), 2), "",
              round(oasis_ident_thera.get_oasis_identity(0.5), 2),
              round(oasis_ident_thera.get_oasis_percentile(0.5), 2),
              sep=",")
        for i, (way, seq) in enumerate(lst):
            if way in ["Therap.", "Wild"]:
                continue
            sm_thera = edit_distance.SequenceMatcher(seq, thera)
            sm_wild = edit_distance.SequenceMatcher(seq, wild)
            seq_emb_seq = get_antiberta_embedding(" ".join(seq))
            struct_emb_seq = get_immunebuilder_embedding(seq)
            sap_emb_seq = get_sapiens_embedding(seq)
            abl_emb_seq = get_ablang_embedding(seq)
            oasis_ident_seq = get_oasis_humanness(oas_db, seq)
            print(seq[:25] + "...", way, sm_thera.distance(), sm_wild.distance(),
                  round(optional_v_gene_score(v_gene_scorer, seq), 2),
                  round(diff_embeddings(seq_emb_thera, seq_emb_seq), 2),
                  round(diff_embeddings(seq_emb_wild, seq_emb_seq), 2),
                  round(diff_embeddings(struct_emb_thera, struct_emb_seq), 2),
                  round(diff_embeddings(struct_emb_wild, struct_emb_seq), 2),
                  round(diff_embeddings(sap_emb_thera, sap_emb_seq), 2),
                  round(diff_embeddings(sap_emb_wild, sap_emb_seq), 2),
                  round(diff_embeddings(abl_emb_thera, abl_emb_seq), 2),
                  round(diff_embeddings(abl_emb_wild, abl_emb_seq), 2),
                  round(oasis_ident_seq.get_oasis_identity(0.5), 2),
                  round(oasis_ident_seq.get_oasis_percentile(0.5), 2),
                  sep=",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Comparator of sequences''')
    parser.add_argument('files', metavar='file', type=str, nargs='+', help='Name of files')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    parser.add_argument('--oas-db', type=str, required=False, default=None, help='Path to OAS db')
    args = parser.parse_args()
    main(args.files, args.dataset, args.oas_db)
