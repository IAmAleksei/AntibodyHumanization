import argparse

from matplotlib import pyplot as plt

from humanization.common import config_loader
from humanization.common.annotations import load_annotation, ChainKind, GeneralChainType, annotate_single
from humanization.common.utils import AA_ALPHABET, configure_logger
from humanization.external_models.antiberta_utils import get_antiberta_embedding, get_antiberta_embeddings
from humanization.external_models.embedding_utils import diff_embeddings


config = config_loader.Config()
logger = configure_logger(config, "Embedding analyzer")


def main():
    raw_seq = input("Sequence: ")
    annotation = load_annotation("chothia", ChainKind.HEAVY)
    annotated_seq = annotate_single(raw_seq, annotation, GeneralChainType.HEAVY)
    pos_label = [an for an, aa in zip(annotation.segmented_positions, annotated_seq) if aa != 'X']
    assert len(pos_label) == len(raw_seq)
    logger.info("Annotated sequence")
    seq = [c for c in raw_seq]
    start_emb = get_antiberta_embedding(" ".join(seq))
    differences = []
    for pos in range(len(seq)):
        logger.info(f"Processed {pos}/{len(seq)} positions")
        aa_backup = seq[pos]
        alt_seqs = []
        for aa in AA_ALPHABET:
            if aa != 'X' and aa != aa_backup:
                seq[pos] = aa
                alt_seqs.append(" ".join(seq))
        seq[pos] = aa_backup
        alt_embs = get_antiberta_embeddings(alt_seqs)
        pos_diffs = [diff_embeddings(start_emb, alt_emb) for alt_emb in alt_embs]
        differences.append(pos_diffs)
    logger.info("Got embeddings")
    plt.figure(figsize=(16, 10), dpi=400)
    for i in range(len(seq)):
        xs = [i] * len(differences[i])
        color = 'g' if pos_label[i].startswith('cdr') else 'b'
        plt.scatter(xs, differences[i], c=color, alpha=0.2, s=20)
    plt.tight_layout()
    plt.savefig("emb_" + raw_seq[:10] + ".jpg")
    plt.show()
    logger.info("Finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Embedding diff analyzer''')
    args = parser.parse_args()
    main()
