import json
import os
from itertools import accumulate
from typing import List, Tuple, Any, NoReturn, Callable

import numpy as np
import pandas
from tqdm import tqdm

from humanization import config_loader
from humanization.annotations import Annotation, annotate_batch
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Dataset reader")

CHUNK_SIZE = 200_000


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
    lst = df['sequence_alignment_aa'].tolist()
    annotated_indexes, annotated_list = [], []
    parts = np.array_split(lst, CHUNK_SIZE)
    for part in parts:
        a_i, a_l = annotate_batch(part, annotation)
        annotated_indexes.extend(a_i)
        annotated_list.extend(a_l)
    X = pandas.DataFrame(annotated_list, columns=aa_columns)  # Make column for every aa
    y = df['v_call'][annotated_indexes].reset_index(drop=True)
    dataset = pandas.concat([X, y], axis=1)
    nan_errors = dataset['v_call'].isna().sum()
    if nan_errors > 0:
        logger.error(f"Found {nan_errors} NaNs in target chain types")
    return dataset


def filter_df(df: pandas.DataFrame, annotation: Annotation) -> pandas.DataFrame:
    if len(annotation.required_positions) > 0:
        masks = [df[pos] == aa for pos, aa in annotation.required_positions.items()]
        mask = next(accumulate(masks, func=lambda a, b: a & b))
        return df[mask]
    return df


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
