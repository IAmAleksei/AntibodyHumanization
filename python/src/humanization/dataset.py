import json
import os
from typing import List, Tuple, Any, NoReturn

import numpy as np
import pandas

from humanization import config_loader
from humanization.annotations import load_annotation, Annotation, segments_to_columns, annotate_batch
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Dataset reader")


def read_file(csv_path: str, requested_columns: List[str] = None) -> Tuple[pandas.DataFrame, Any]:
    metadata = json.loads(','.join(pandas.read_csv(csv_path, nrows=0).columns))
    df: pandas.DataFrame = pandas.read_csv(csv_path, skiprows=1)  # Drop row with running info
    df = df[requested_columns].dropna()
    return df, metadata


def correct_v_call(df: pandas.DataFrame, metadata: Any) -> NoReturn:
    if metadata['Species'] != 'human':
        df['v_call'] = 'NOT_HUMAN'
    else:
        df['v_call'] = df['v_call'].str.slice(stop=5)


def read_dataset(
        dataset_list_path: str = "/Users/alex-shishkin/PycharmProjects/science/bash/h_data.txt",
        requested_columns: List[str] = None) -> pandas.DataFrame:
    requested_columns = requested_columns if requested_columns is not None else ['sequence_alignment_aa']
    logger.info("Dataset reading...")
    dfs = []
    original_data_size = 0
    with open(dataset_list_path, 'r') as heavy_data_list:
        dataset_list = heavy_data_list.readlines()
        for dataset_path in dataset_list:
            try:
                dataset_path = dataset_path.strip()
                if len(dataset_path) == 0:
                    continue  # Drop empty lines from dataset paths
                df, metadata = read_file(dataset_path, requested_columns)
                correct_v_call(df, metadata)
                dfs.append(df)
                original_data_size += df.shape[0]
                logger.debug(f"{dataset_path} has been read ({df.shape[0]} rows)")
            except Exception as e:
                logger.error(e, exc_info=True)
    df = pandas.concat(dfs, axis=0, ignore_index=True).drop_duplicates(ignore_index=True)
    logger.debug(f"Original dataset: {original_data_size} rows")
    logger.info(f"Dataset: {df.shape[0]} rows (duplicates removed)")
    return df


def make_annotated_df(df: pandas.DataFrame, annotation: Annotation) -> pandas.DataFrame:
    aa_columns = segments_to_columns(annotation)
    annotated_indexes, annotated_list = annotate_batch(df['sequence_alignment_aa'].tolist(), annotation)
    X = pandas.DataFrame(annotated_list, columns=aa_columns)  # Make column for every aa
    y = df['v_call'][annotated_indexes]
    dataset = pandas.concat([X, y], axis=1)
    return dataset


def filter_df(df: pandas.DataFrame) -> pandas.DataFrame:
    mask1 = df['fwr1_23'] == "C"
    mask2 = df['fwr2_15'] == "W"
    mask3 = df['fwr3_39'] == "C"
    return df[mask1 & mask2 & mask3]


def read_prepared_heavy_dataset(input_directory: str, annotation: Annotation):
    dataset = read_dataset(input_directory, requested_columns=['sequence_alignment_aa', 'v_call'])
    logger.debug(f"Dataset columns: {segments_to_columns(annotation)}")
    dataset = make_annotated_df(dataset, annotation)
    dataset = filter_df(dataset)
    X = dataset.drop(['v_call'], axis=1)
    y = dataset['v_call']
    logger.debug(f"Dataset: {X.shape[0]} rows (cysteine/tryptophan errors removed)")
    return X, y


def make_binary_target(y, target_v_type):
    return np.where(y.apply(lambda x: x == f"IGHV{target_v_type}"), 1, 0)


def prepare_file(csv_file: str, annotation: Annotation) -> str:
    df, metadata = read_file(csv_file, ['sequence_alignment_aa', 'v_call'])
    correct_v_call(df, metadata)
    df = make_annotated_df(df, annotation)
    df = filter_df(df)
    csv_dir, csv_name = os.path.split(csv_file)
    result_name = f"proc_{csv_name}"
    result_file = os.path.join(csv_dir, result_name)
    df.to_csv(result_file, index=False)
    return result_file


def prepare_files(csv_files: List[str], schema: str) -> NoReturn:
    annotation = load_annotation(schema)
    for file in csv_files:
        prepare_file(file, annotation)


if __name__ == '__main__':
    # read_dataset()
    prepare_files(["/Users/alex-shishkin/PycharmProjects/science/bash/heavy_dataset/1279051_1_Heavy_IGHG.csv.gz"],
                  "chothia")
