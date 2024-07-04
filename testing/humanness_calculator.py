import argparse
import json
import random

from humanization.common import config_loader
from humanization.common.annotations import HeavyChainType, annotate_single, ChothiaHeavy, GeneralChainType
from humanization.common.utils import configure_logger
from humanization.humanness_calculator.model_wrapper import load_model
from humanization.humanness_calculator.stats import print_distribution


config = config_loader.Config()
logger = configure_logger(config, "Humanness calculator")


def main(model_dir):
    with open('thera_antibodies.json', 'r') as fp:
        samples = json.load(fp)
    with open('extra_thera_antibodies.json', 'r') as fp:
        samples += json.load(fp)
    logger.info(f"Found {len(samples)} antibodies")
    test_set = []
    for sample in samples:
        name = sample["name"]
        source = sample["heavy"]["sequ"].replace("-", "")
        thera = sample["heavy"]["ther"].replace("-", "")
        if len(source) != len(thera):
            logger.debug(name + " skipped")
            continue
        diff_positions = [i for i, (sc, tc) in enumerate(zip(source, thera)) if sc != tc]
        for num_changes in range(1, len(diff_positions) + 1):
            random.shuffle(diff_positions)
            subsample_positions = diff_positions[:num_changes]
            test_seq = [c for c in source]
            for i in subsample_positions:
                test_seq[i] = thera[i]
            aligned_seq = annotate_single("".join(test_seq), ChothiaHeavy(), GeneralChainType.HEAVY)
            if not aligned_seq:
                logger.debug("Bad alignment. Skip")
                continue
            test_set.append(aligned_seq)
    logger.info(f"{len(test_set)} antibodies generated")
    model_wrapper = load_model(model_dir, HeavyChainType.V1)
    y_pred_proba = model_wrapper.model.predict_proba(test_set)
    logger.info(f"Got predictions")
    print_distribution(y_pred_proba, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''CM models tester''')
    parser.add_argument('models', type=str, help='Path to directory with models')
    args = parser.parse_args()

    main(args.models)
