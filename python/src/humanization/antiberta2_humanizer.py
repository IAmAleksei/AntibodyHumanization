import argparse
import traceback
from typing import Any

from transformers import RoFormerForMaskedLM, RoFormerTokenizer, pipeline

from humanization import config_loader, humanizer
from humanization.annotations import load_annotation, ChainKind, GeneralChainType, ChainType
from humanization.humanizer import common_parser_options
from humanization.utils import configure_logger
from humanization.v_gene_scorer import get_similar_human_samples

config = config_loader.Config()
logger = configure_logger(config, "AntiBERTa2 humanizer")


def mask_sequence(models, dataset, sequence: str) -> str:
    annotation = load_annotation("chothia", ChainKind.HEAVY)
    human_samples = get_similar_human_samples(annotation, dataset, [sequence], GeneralChainType.HEAVY)
    str_type = human_samples[0][0][2]
    _, result, its = humanizer.process_sequences(models, [("S", sequence)], ChainType.from_oas_type(str_type),
                                                 0.999, aligned_result=True, limit_changes=1)[0]

    diff_pos = -1
    if its[-1].change is not None and its[-1].change.position is not None:
        diff_pos = its[-1].change.position
    if diff_pos >= 0:
        logger.info(f"Found diff position: {diff_pos}")
        sequence_list = list(sequence)
        sequence_list[diff_pos] = "[MASK]"
        result = " ".join(filter(lambda aa: aa != "X", sequence_list))
        logger.debug(f"Result: {result}")
        return result
    else:
        logger.info(f"No diff position")
        return sequence


def humanize(seq: str) -> Any:
    tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2")
    model = RoFormerForMaskedLM.from_pretrained("alchemab/antiberta2")
    filler = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    result = filler(seq)
    return result


def process_sequence(models, dataset, sequence):
    last_sequence = ""
    while last_sequence != sequence:
        logger.info("New iteration")
        last_sequence = sequence
        masked_sequence = mask_sequence(models, dataset, sequence)
        candidates = humanize(masked_sequence)
        sequence = candidates[0]['sequence'].replace(' ', '')
    return sequence


def process_sequences(models, dataset, sequences):
    results = []
    for name, sequence in sequences:
        logger.info(f"Processing {name}")
        try:
            result_one = [process_sequence(models, dataset, sequence)]
        except RuntimeError as _:
            traceback.print_exc()
            result_one = [""]
        for i, result in enumerate(result_one):
            results.append((f"{name}_cand{i + 1}", result))
    return results


def main(args):
    sequence = input("Enter sequence:")
    print(process_sequence(args.models, args.dataset, sequence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''AntiBERTa2 humanizer''')
    common_parser_options(parser)
    args = parser.parse_args()

    main(args)
