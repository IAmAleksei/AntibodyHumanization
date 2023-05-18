import json
import os
from typing import List, Tuple, Any, NoReturn, Callable

import numpy as np
import pandas
from tqdm import tqdm

from humanization import config_loader
from humanization.annotations import Annotation, annotate_batch
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Dataset reader")


def read_file(csv_path: str, requested_columns: List[str]) -> Tuple[pandas.DataFrame, Any]:
    metadata = json.loads(','.join(pandas.read_csv(csv_path, nrows=0).columns))
    df: pandas.DataFrame = pandas.read_csv(csv_path, skiprows=1)  # Drop row with running info
    if requested_columns is not None:
        df = df[requested_columns]
    df.dropna(inplace=True)
    return df, metadata


def correct_v_call(df: pandas.DataFrame, metadata: Any) -> NoReturn:
    if metadata['Species'] != 'human':
        df['v_call'] = 'NOT_HUMAN'
    else:
        df['v_call'] = df['v_call'].str.slice(stop=5)


def make_annotated_df(df: pandas.DataFrame, annotation: Annotation) -> pandas.DataFrame:
    aa_columns = annotation.segmented_positions
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


def read_heavy_dataset(input_dir: str, read_function: Callable[[str], pandas.DataFrame]):
    logger.info("Dataset reading...")
    dfs = []
    original_data_size = 0
    file_names = os.listdir(input_dir)
    for input_file_name in tqdm(file_names):
        input_file_path = os.path.join(input_dir, input_file_name)
        df: pandas.DataFrame = read_function(input_file_path)
        dfs.append(df)
        original_data_size += df.shape[0]
    dataset = pandas.concat(dfs, axis=0, ignore_index=True).drop_duplicates(ignore_index=True)
    logger.info(f"Original dataset: {original_data_size} rows")
    logger.info(f"Dataset: {dataset.shape[0]} rows (duplicates removed)")
    X = dataset.drop(['v_call'], axis=1)
    y = dataset['v_call']
    return X, y


def read_annotated_heavy_dataset(input_dir: str) -> Tuple[pandas.DataFrame, pandas.Series]:
    return read_heavy_dataset(input_dir, pandas.read_csv)


def merge_all_columns(df: pandas.DataFrame) -> List[str]:
    values: List[List[str]] = df.to_numpy().tolist()
    result = list(map("".join, values))
    return result


def make_binary_target(y, target_v_type):
    return np.where(y.apply(lambda x: x == f"IGHV{target_v_type}"), 1, 0)
