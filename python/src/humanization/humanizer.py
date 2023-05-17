import argparse
from typing import List

import pandas
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from humanization import config_loader, utils
from humanization.annotations import segments_to_columns, annotate_batch
from humanization.dataset import read_prepared_heavy_dataset
from humanization.models import load_model, HeavyChainType, LightChainType, ModelWrapper
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Humanizer")


def test_single_change(df: pandas.DataFrame, model_wrapper: ModelWrapper, column_name: str):
    aa_backup = df.iloc[0][column_name]
    if aa_backup == 'X':
        return None, 0.0  # No inserts
    best_change, best_value = None, 0.0
    for new_aa in utils.AA_ALPHABET:  # TODO: make it batched
        df[column_name] = new_aa
        new_value = model_wrapper.model.predict_proba(df)[0][1]
        if new_value > best_value:
            best_change, best_value = new_aa, new_value
    df[column_name] = aa_backup
    return best_change, best_value


def get_str_sequence(df: pandas.Series) -> str:
    result = "".join(filter(lambda x: x != "X", df.tolist()))
    return result


def calc_humanness(df: pandas.DataFrame, current: pandas.Series, aa_columns: List[str]) -> float:
    def calc_affinity(row: pandas.Series):
        matched = [
            row[aa_column] == current[aa_column] and row[aa_column] != 'X'
            for aa_column in aa_columns
        ]  # TODO: use only first 105 positions
        return sum(matched)

    seq_len = sum(aa != 'X' for aa in get_str_sequence(current))
    affinity = df.apply(calc_affinity, axis=1)
    nearest_idx = affinity.idxmax()
    nearest = affinity[nearest_idx]
    result = nearest / seq_len
    logger.debug(f"Nearest human sample: {get_str_sequence(df.iloc[nearest_idx])}. Humanness = {result}")
    return result


def process_query(sequence: str, model_wrapper: ModelWrapper, target_model: float, target_humanness: float,
                  dataset: pandas.DataFrame = None, modify_cdr: bool = True) -> str:
    aa_columns = segments_to_columns(model_wrapper.annotation)
    _, annotated_seq = annotate_batch([sequence], model_wrapper.annotation)
    if len(annotated_seq) == 0:
        logger.error(f"Cannot annotate `{sequence}`")
        return sequence
    df = pandas.DataFrame(annotated_seq, columns=aa_columns)
    for it in range(config.get(config_loader.MAX_CHANGES)):
        current_value = model_wrapper.model.predict_proba(df)[0][1]
        logger.debug(f"Iteration {it + 1}. Current model metric = {current_value}")
        best_change, best_value = None, current_value
        for idx, column_name in enumerate(aa_columns):
            if column_name.startswith('CDR') and not modify_cdr:
                continue
            candidate_change, candidate_value = test_single_change(df, model_wrapper, column_name)
            if candidate_value > best_value:
                best_value = candidate_value
                best_change = column_name, candidate_change
        if best_change is not None:
            best_column_name, best_new_aa = best_change
            prev_aa = df.iloc[0][best_column_name]
            df[best_column_name] = best_new_aa
            logger.debug(f"Best change: model metric = {current_value} -> {best_value}")
            if dataset is not None:
                humanness = calc_humanness(dataset, df.iloc[0], aa_columns)
                logger.debug(f"After change humanness = {humanness}")
            else:
                humanness = 0.0
            logger.debug(f"Position {best_column_name}: {prev_aa}->{best_new_aa}")
            if best_value > target_model and (dataset is None or humanness > target_humanness):
                logger.info(f"Target metrics are reached ({best_value})")
                break
        else:
            logger.info(f"No effective changes found. Stop algorithm on model metric = {current_value}")
            break
    return get_str_sequence(df.iloc[0])


def read_sequences(input_file):
    if not input_file:
        sequence = input("Enter sequence: ")
        result = [("CONSOLE", sequence)]
    else:
        result = [(seq.name, str(seq.seq)) for seq in SeqIO.parse(input_file, 'fasta')]
    return result


def write_sequences(output_file, sequences):
    if not output_file:
        for name, result in sequences:
            print(f'>{name}')
            print(result)
    else:
        seqs = [SeqRecord(Seq(seq), id=name, description='') for name, seq in sequences]
        SeqIO.write(seqs, output_file, 'fasta')


def main(models_dir, input_file, dataset_file, modify_cdr, output_file):
    sequences = read_sequences(input_file)
    chain_type_str = input("Enter chain type (heavy or light): ")
    if chain_type_str.lower() in ["h", "heavy"]:
        chain_type_class = HeavyChainType
        v_gene_type = input("V gene type (1-7): ")
    elif chain_type_str.lower() in ["l", "light"]:
        chain_type_class = LightChainType
        v_gene_type = input("V gene type (kappa or lambda): ")
    else:
        raise RuntimeError(f"Unknown chain type: {chain_type_str}")
    chain_type = chain_type_class(v_gene_type)
    target_model = float(input("Enter target model metric: "))
    model_wrapper = load_model(models_dir, chain_type)
    if dataset_file is not None:
        target_humanness = float(input("Enter target humanness: "))
        X, y = read_prepared_heavy_dataset(dataset_file, model_wrapper.annotation)
        dataset = X[y != 'NOT_HUMAN'].reset_index(drop=True)
    else:
        target_humanness = 0.0
        dataset = None
    results = []
    for name, sequence in sequences:
        result = process_query(sequence, model_wrapper, target_model, target_humanness,
                               dataset=dataset, modify_cdr=modify_cdr)
        results.append((name, result))
    write_sequences(output_file, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Humanizer''')
    parser.add_argument('models', type=str, help='Path to directory with models')
    parser.add_argument('--input', type=str, required=False, help='Path to input fasta file')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset')
    parser.add_argument('--modify-cdr', action='store_true', help='Allowing CDR modifications')
    parser.add_argument('--skip-cdr', dest='modify_cdr', action='store_false')
    parser.set_defaults(modify_cdr=True)
    parser.add_argument('--output', type=str, required=False, help='Path to output fasta file')
    args = parser.parse_args()

    main(models_dir=args.models,
         input_file=args.input,
         dataset_file=args.dataset,
         modify_cdr=args.modify_cdr,
         output_file=args.output)
