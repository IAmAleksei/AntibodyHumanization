import argparse

import pandas
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from catboost import CatBoostClassifier

from humanization import config_loader, utils
from humanization.annotations import segments_to_columns, annotate_batch
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


def process_query(sequence: str, model_wrapper: ModelWrapper, target: float) -> str:
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
            candidate_change, candidate_value = test_single_change(df, model_wrapper, column_name)
            if candidate_value > best_value:
                best_value = candidate_value
                best_change = column_name, candidate_change
        if best_change is not None:
            best_column_name, best_new_aa = best_change
            prev_aa = df.iloc[0][best_column_name]
            df[best_column_name] = best_new_aa
            logger.debug(f"Best change: model metric = {current_value} -> {best_value}")
            logger.debug(f"Position {best_column_name}: {prev_aa}->{best_new_aa}")
            if best_value > target:
                logger.info(f"Target model metric is reached ({best_value})")
                break
        else:
            logger.info(f"No effective changes found. Stop algorithm on model metric = {current_value}")
    result = "".join(filter(lambda x: x != "X", df.iloc[0].tolist()))
    return result


def read_sequences(input_file):
    if not input_file:
        sequence = input("Enter sequence: ")
        result = [("CONSOLE", sequence)]
    else:
        result = [(seq.name, seq.seq) for seq in SeqIO.parse(input_file, 'fasta')]
    return result


def write_sequences(output_file, sequences):
    if not output_file:
        for name, result in sequences:
            print(f'>{name}')
            print(result)
    else:
        seqs = [SeqRecord(Seq(seq), id=name, description='') for name, seq in sequences]
        SeqIO.write(seqs, output_file, 'fasta')


def main(models_dir, input_file, output_file):
    sequences = read_sequences(input_file)
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
    target = float(input("Enter target model metric: "))
    model_wrapper = load_model(models_dir, chain_type)
    results = []
    for name, sequence in sequences:
        result = process_query(sequence, model_wrapper, target)
        results.append((name, result))
    write_sequences(output_file, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Humanizer''')
    parser.add_argument('models', type=str, help='Path to directory with models')
    parser.add_argument('--input', type=str, required=False, help='Path to input fasta file')
    parser.add_argument('--output', type=str, required=False, help='Path to output fasta file')
    args = parser.parse_args()

    main(args.models, args.input, args.output)
