import argparse
import traceback

from humanization.algorithms import direct_humanizer
from humanization.common import config_loader
from humanization.common.annotations import load_annotation, ChainKind, GeneralChainType, ChainType
from humanization.common.utils import configure_logger
from humanization.common.v_gene_scorer import get_similar_samples, build_v_gene_scorer
from humanization.external_models.antiberta_utils import fill_mask
from humanization.humanness_calculator.model_wrapper import load_model

config = config_loader.Config()
logger = configure_logger(config, "AntiBERTa2 humanizer")


def mask_sequence(model_wrapper, v_gene_scorer, sequence: str, limit_changes):
    annotation = load_annotation("chothia", ChainKind.HEAVY)
    skip_positions = []
    for i in range(min(limit_changes, config.get(config_loader.ANTIBERTA_CANDIDATES))):
        _, result, its = direct_humanizer._process_sequences(model_wrapper, v_gene_scorer, [(f"S_{i + 1}", sequence)],
                                                             0.999, skip_positions=",".join(skip_positions),
                                                             aligned_result=True, limit_changes=1)[0]
        if its[-1].change is not None and its[-1].change.position is not None:
            diff_pos = its[-1].change.position
            logger.info(f"Found diff position: {diff_pos}")
            result_list = list(result)
            result_list[diff_pos] = "[MASK]"
            masked = " ".join(filter(lambda aa: aa != "X", result_list))
            skip_positions.append(annotation.segmented_positions[diff_pos])
            yield masked
        else:
            logger.info(f"No diff position")
            break


def process_sequence(model_wrapper, v_gene_scorer, sequence, limit_changes):
    last_sequence = ""
    its = 0
    while last_sequence != sequence:
        its += 1
        logger.info(f"New iteration. Current sequence: {sequence}")
        last_sequence = sequence
        for i, masked_sequence in enumerate(mask_sequence(model_wrapper, v_gene_scorer, sequence, limit_changes)):
            humanized_sequence = fill_mask(masked_sequence)
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
    human_samples = get_similar_samples(annotation, args.dataset, [sequence], chain_type=GeneralChainType.HEAVY)
    chain_type = ChainType.from_oas_type(human_samples[0][0][2])
    model_wrapper = load_model(args.models, chain_type)
    v_gene_scorer = build_v_gene_scorer(annotation, args.dataset, chain_type)
    print(process_sequence(model_wrapper, v_gene_scorer, sequence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''AntiBERTa2 humanizer''')
    direct_humanizer.common_parser_options(parser)
    args = parser.parse_args()

    main(args)
