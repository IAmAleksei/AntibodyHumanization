import argparse
from collections import defaultdict

import edit_distance
from Bio import SeqIO

from humanization.common import config_loader
from humanization.common.annotations import ChothiaHeavy, annotate_single, GeneralChainType, ChainType, HeavyChainType
from humanization.common.utils import configure_logger
from humanization.common.v_gene_scorer import build_v_gene_scorer
from humanization.external_models.ablang_utils import get_ablang_embedding
from humanization.external_models.antiberta_utils import get_antiberta_embedding
from humanization.external_models.biophi_utils import get_oasis_humanness
from humanization.external_models.embedding_utils import diff_embeddings
from humanization.external_models.immunebuilder_utils import get_immunebuilder_embedding
from humanization.external_models.sapiens_utils import get_sapiens_embedding
from humanization.humanness_calculator.model_wrapper import load_all_models, ModelWrapper

config = config_loader.Config()
logger = configure_logger(config, "Comparator")


def optional_v_gene_score(v_gene_scorer, seq: str):
    aligned_seq = annotate_single(seq, ChothiaHeavy(), GeneralChainType.HEAVY)
    if aligned_seq is None:
        return -1.0
    return v_gene_scorer.query(aligned_seq)[0][1]


def v_gene_type(v_gene_scorer, seq: str) -> ChainType:
    aligned_seq = annotate_single(seq, ChothiaHeavy(), GeneralChainType.HEAVY)
    return ChainType.from_oas_type(v_gene_scorer.query(aligned_seq)[0][2])


def model_humanness_score(model: ModelWrapper, seq: str):
    return model.model.predict_proba(list(seq))[1]


def catboost_humanness_score(models, v_gene_scorer, seq: str):
    chain_type = v_gene_type(v_gene_scorer, seq)
    return model_humanness_score(models[chain_type], seq)


COLUMNS = ["Seq", "Type", "ThDist", "WDist", "HuVGS", "WVGS", "ThBerta", "WBerta", "ThAbody", "WAbody",
           "ThSap", "WSap", "ThAblang", "WAblang", "OASId", "OASPerc", "HumCM", "MurCM"]


def print_info(seq: str, way: str, v_gene_scorer, wild_v_gene_scorer, biophi_path, models, wild_models,
               seq_emb_thera=None, seq_emb_wild=None, struct_emb_thera=None, struct_emb_wild=None,
               sap_emb_thera=None, sap_emb_wild=None, abl_emb_thera=None, abl_emb_wild=None,
               wild: str = None, thera: str = None):
    seq_emb_seq = get_antiberta_embedding(" ".join(seq))
    struct_emb_seq = get_immunebuilder_embedding(seq)
    sap_emb_seq = get_sapiens_embedding(seq)
    abl_emb_seq = get_ablang_embedding(seq)
    oasis_ident_seq = get_oasis_humanness(biophi_path, seq)

    args = [
        seq[:25] + "...",
        way,
        edit_distance.SequenceMatcher(seq, thera) if thera is not None else "",
        edit_distance.SequenceMatcher(seq, wild) if wild is not None else "",
        round(optional_v_gene_score(v_gene_scorer, seq), 3),
        round(optional_v_gene_score(wild_v_gene_scorer, seq), 3),
        round(diff_embeddings(seq_emb_thera, seq_emb_seq), 3) if thera is not None else "",
        round(diff_embeddings(seq_emb_wild, seq_emb_seq), 3) if wild is not None else "",
        round(diff_embeddings(struct_emb_thera, struct_emb_seq), 4) if thera is not None else "",
        round(diff_embeddings(struct_emb_wild, struct_emb_seq), 4) if wild is not None else "",
        round(diff_embeddings(sap_emb_thera, sap_emb_seq), 3) if thera is not None else "",
        round(diff_embeddings(sap_emb_wild, sap_emb_seq), 3) if wild is not None else "",
        round(diff_embeddings(abl_emb_thera, abl_emb_seq), 4) if thera is not None else "",
        round(diff_embeddings(abl_emb_wild, abl_emb_seq), 4) if wild is not None else "",
        round(oasis_ident_seq.get_oasis_identity(0.5), 2),
        round(oasis_ident_seq.get_oasis_percentile(0.5), 2),
        round(catboost_humanness_score(models, v_gene_scorer, seq), 2),
    ]

    args.extend([model_humanness_score(model, seq) for model in wild_models])

    print(*args, sep=",")


def main(files, dataset, wild_dataset, biophi_path, model_dir, wild_model_dir):
    models = load_all_models(model_dir, GeneralChainType.HEAVY)
    wild_models = [load_all_models(wild_model_dir, GeneralChainType.HEAVY)[HeavyChainType.V1]]
    v_gene_scorer = build_v_gene_scorer(ChothiaHeavy(), dataset)
    wild_v_gene_scorer = build_v_gene_scorer(ChothiaHeavy(), wild_dataset)
    seqs = defaultdict(list)
    for file in files:
        for seq in SeqIO.parse(file, 'fasta'):
            mab = seq.name.split("_")[0]
            way = seq.name.split("_")[2]
            seqs[mab].append((way, str(seq.seq).replace('X', '')))
    print(*COLUMNS, sep=",")
    for mab, lst in seqs.items():
        print()
        print(mab)
        thera = next(seq for way, seq in lst if "Therap." == way)
        wild = next(seq for way, seq in lst if "Wild" == way)
        seq_emb_thera, seq_emb_wild = get_antiberta_embedding(" ".join(thera)), get_antiberta_embedding(" ".join(wild))
        struct_emb_thera, struct_emb_wild = get_immunebuilder_embedding(thera), get_immunebuilder_embedding(wild)
        sap_emb_thera, sap_emb_wild = get_sapiens_embedding(thera), get_sapiens_embedding(wild)
        abl_emb_thera, abl_emb_wild = get_ablang_embedding(thera), get_ablang_embedding(wild)
        print_info(wild, "Wild.", v_gene_scorer, wild_v_gene_scorer, biophi_path, models, wild_models,
                   seq_emb_thera, seq_emb_wild, struct_emb_thera, struct_emb_wild, sap_emb_thera, sap_emb_wild,
                   abl_emb_thera, abl_emb_wild, None, None)
        print_info(thera, "Wild.", v_gene_scorer, wild_v_gene_scorer, biophi_path, models, wild_models,
                   seq_emb_thera, seq_emb_wild, struct_emb_thera, struct_emb_wild, sap_emb_thera, sap_emb_wild,
                   abl_emb_thera, abl_emb_wild, wild, None)
        for i, (way, seq) in enumerate(lst):
            if way in ["Therap.", "Wild"]:
                continue
            print_info(seq, way, v_gene_scorer, wild_v_gene_scorer, biophi_path, models, wild_models,
                       seq_emb_thera, seq_emb_wild, struct_emb_thera, struct_emb_wild, sap_emb_thera, sap_emb_wild,
                       abl_emb_thera, abl_emb_wild, wild, thera)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Comparator of sequences''')
    parser.add_argument('files', metavar='file', type=str, nargs='+', help='Name of files')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset for humanness calculation')
    parser.add_argument('--wild-dataset', type=str, required=False, help='Path to dataset for wildness calculation')
    parser.add_argument('--biophi-path', type=str, required=False, default=None, help='Path to BioPhi dir')
    parser.add_argument('--models', type=str, help='Path to directory with human random forest models')
    parser.add_argument('--wild-models', type=str, help='Path to directory with murine random forest models')
    args = parser.parse_args()
    main(args.files, args.dataset, args.wild_dataset, args.biophi_path, args.models, args.wild_models)
