import argparse

from transformers import RoFormerForMaskedLM, RoFormerTokenizer, pipeline

from humanization import config_loader, humanizer
from humanization.annotations import load_annotation, ChainKind, GeneralChainType, ChainType
from humanization.humanizer import common_parser_options
from humanization.models import load_model
from humanization.utils import configure_logger
from humanization.v_gene_scorer import get_similar_human_samples

config = config_loader.Config()
logger = configure_logger(config, "AntiBERTa2 humanizer")


def mask_sequence(args, s: str) -> str:
    annotation = load_annotation("chothia", ChainKind.HEAVY)
    human_samples = get_similar_human_samples(annotation, args.dataset, [s], GeneralChainType.HEAVY)
    tp = ChainType.from_oas_type(human_samples[0][0][2])
    model_wrapper = load_model(args.models, tp)
    direct_result = humanizer._process_sequences(model_wrapper, None, [("S", s)], 0.999, aligned_result=True,
                                                 limit_changes=1)[0][1]
    s = " ".join(s)
    direct_result = " ".join(direct_result)
    diff_pos = -1
    for i, (c1, c2) in enumerate(zip(s, direct_result)):
        if c1 != c2:
            diff_pos = i
    if diff_pos >= 0:
        logger.info(f"Found diff position: {diff_pos}")
        return s[:diff_pos] + "[MASK]" + s[diff_pos + 1:]
    else:
        return s


def humanize(seq: str) -> str:
    tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2")
    model = RoFormerForMaskedLM.from_pretrained("alchemab/antiberta2")
    filler = pipeline(model=model, tokenizer=tokenizer)
    result = filler(seq)
    return result


def main(args):
    sequence = input("Enter sequence:")
    last_sequence = ""
    while last_sequence != sequence:
        logger.info("New iteration")
        last_sequence = sequence
        masked_sequence = mask_sequence(args, sequence)
        sequence = humanize(masked_sequence)
    print(sequence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''AntiBERTa2 humanizer''')
    common_parser_options(parser)
    args = parser.parse_args()

    main(args)
