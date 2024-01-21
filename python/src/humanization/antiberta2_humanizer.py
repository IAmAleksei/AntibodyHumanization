import argparse
import traceback
from typing import Any, List

from transformers import RoFormerForMaskedLM, RoFormerTokenizer, pipeline

from humanization import config_loader, humanizer
from humanization.annotations import load_annotation, ChainKind, GeneralChainType, ChainType
from humanization.humanizer import common_parser_options
from humanization.models import load_model
from humanization.utils import configure_logger
from humanization.v_gene_scorer import get_similar_human_samples, build_v_gene_scorer

config = config_loader.Config()
logger = configure_logger(config, "AntiBERTa2 humanizer")


def mask_sequence(model_wrapper, v_gene_scorer, sequence: str, limit_changes):
    annotation = load_annotation("chothia", ChainKind.HEAVY)
    skip_positions = []
    for i in range(min(limit_changes, config.get(config_loader.ANTIBERTA_CANDIDATES))):
        _, result, its = humanizer._process_sequences(model_wrapper, v_gene_scorer, [(f"S_{i + 1}", sequence)],
                                                      0.999, skip_positions=",".join(skip_positions),
                                                      aligned_result=True, limit_changes=1)[0]
        if its[-1].change is not None and its[-1].change.position is not None:
            diff_pos = its[-1].change.position
            logger.info(f"Found diff position: {diff_pos}")
            result_list = list(result)
            result_list[diff_pos] = "[MASK]"
            masked = " ".join(filter(lambda aa: aa != "X", result_list))
            # logger.debug(f"Masked: {masked}")
            skip_positions.append(annotation.segmented_positions[diff_pos])
            yield masked
        else:
            logger.info(f"No diff position")
            break


def humanize(seq: str) -> str:
    tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2")
    model = RoFormerForMaskedLM.from_pretrained("alchemab/antiberta2")
    filler = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    result = filler(seq)
    # logger.debug(f"Selected candidate: {result[0]['token_str']}")
    return result[0]['sequence'].replace(' ', '').replace('Ḣ', 'H')


def process_sequence(model_wrapper, v_gene_scorer, sequence, limit_changes):
    last_sequence = ""
    its = 0
    while last_sequence != sequence:
        its += 1
        logger.info(f"New iteration. Current sequence: {sequence}")
        last_sequence = sequence
        for i, masked_sequence in enumerate(mask_sequence(model_wrapper, v_gene_scorer, sequence, limit_changes)):
            humanized_sequence = humanize(masked_sequence)
            if humanized_sequence != last_sequence:
                logger.info(f"Created new sequence for {i + 1} tries")
                sequence = humanized_sequence
                break
    logger.info(f"Changes established for {its} iterations")
    return sequence


def process_sequences(model_wrapper, v_gene_scorer, sequences, limit_changes):
    results = []
    for name, sequence in sequences:
        logger.info(f"Processing {name}")
        try:
            result_one = [process_sequence(model_wrapper, v_gene_scorer, sequence, limit_changes)]
        except RuntimeError as _:
            traceback.print_exc()
            result_one = [""]
        for i, result in enumerate(result_one):
            results.append((f"{name}_cand{i + 1}", result))
    return results


def main(args):
    sequence = input("Enter sequence:")
    annotation = load_annotation("chothia", ChainKind.HEAVY)
    human_samples = get_similar_human_samples(annotation, args.dataset, [sequence], GeneralChainType.HEAVY)
    chain_type = ChainType.from_oas_type(human_samples[0][0][2])
    model_wrapper = load_model(args.models, chain_type)
    v_gene_scorer = build_v_gene_scorer(annotation, args.dataset, chain_type)
    print(process_sequence(model_wrapper, v_gene_scorer, sequence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''AntiBERTa2 humanizer''')
    common_parser_options(parser)
    args = parser.parse_args()

    main(args)
