import argparse

import pandas
from catboost import CatBoostClassifier

from humanization import config_loader, utils
from humanization.models import load_model, HeavyChainType, LightChainType
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Humanizer")


def get_column_name(segment_name: str, pos: int):
    return f'{segment_name}_{pos}'


def split_chain(sequence: str):
    segments = sequence.split("|")
    cols = []
    for segment, (segment_name, segment_len) in zip(segments, utils.SEGMENTS):
        for j in range(segment_len):
            col = pandas.DataFrame({get_column_name(segment_name, j + 1): [segment[j] if j < len(segment) else '']})
            cols.append(col)
    result = pandas.concat(cols, axis=1)
    return result


def process_query(sequence: str, model: CatBoostClassifier, target: float) -> str:
    df = split_chain(sequence)
    for it in range(config.get(config_loader.MAX_CHANGES)):
        current_value = model.predict_proba(df)[0][1]
        logger.debug(f"Iteration {it + 1}. Current humanness = {current_value}")
        best_change, best_value = None, current_value
        for segment_name, segment_len in utils.SEGMENTS:
            for pos in range(1, segment_len + 1):
                aa_backup = df.iloc[0][get_column_name(segment_name, pos)]
                if aa_backup == '':
                    continue  # No inserts
                for new_aa in utils.AA_ALPHABET:  # TODO: make it batched
                    df[get_column_name(segment_name, pos)] = new_aa
                    new_value = model.predict_proba(df)[0][1]
                    if new_value > best_value:
                        best_value = new_value
                        best_change = segment_name, pos, new_aa
                df[get_column_name(segment_name, pos)] = aa_backup
            logger.debug(f"Segment {segment_name} proceed. Current best humanness = {best_value}")
        if best_change is not None:
            best_segment_name, best_segment_pos, best_new_aa = best_change
            prev_aa = df.iloc[0][get_column_name(best_segment_name, best_segment_pos)]
            df[get_column_name(best_segment_name, best_segment_pos)] = best_new_aa
            logger.debug(f"Best change: humanness = {current_value} -> {best_value}")
            logger.debug(f"Position {best_segment_name}-{best_segment_pos}: {prev_aa}->{best_new_aa}")
            if best_value > target:
                logger.info(f"Target humanness is reached ({best_value})")
                break
        else:
            logger.info(f"No effective changes found. Stop algorithm on humanness = {current_value}")
    result = "".join(df.iloc[0].tolist())
    return result


def main(models_dir):
    sequence = input("Enter aligned sequence (split FR and CDR segments by |): ")
    chain_type_str = input("Enter chain type (heavy or light): ")
    if chain_type_str == "heavy":
        chain_type_class = HeavyChainType
        v_gene_type = input("V gene type (1-7): ")
    elif chain_type_str == "light":
        chain_type_class = LightChainType
        v_gene_type = input("V gene type (kappa or lambda): ")
    else:
        raise RuntimeError(f"Unknown chain type: {chain_type_str}")
    chain_type = chain_type_class(v_gene_type)
    target = float(input("Enter target humanness: "))
    model = load_model(models_dir, chain_type)
    result = process_query(sequence, model, target)
    print(f"Result: {result}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Humanizer''')
    parser.add_argument('models', type=str, help='Path to directory with models')
    args = parser.parse_args()

    main(args.models)
