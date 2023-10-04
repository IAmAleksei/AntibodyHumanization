import json
import math
import os
from itertools import accumulate
from typing import List, Tuple, Any, NoReturn, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from humanization import config_loader
from humanization.annotations import Annotation, annotate_batch
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Dataset reader")


def read_file(csv_path: str, requested_columns: List[str]) -> Tuple[pd.DataFrame, Any]:
    metadata = json.loads(','.join(pd.read_csv(csv_path, nrows=0).columns))
    df: pd.DataFrame = pd.read_csv(csv_path, skiprows=1)  # Drop row with running info
    if requested_columns is not None:
        df = df[requested_columns]
    df.dropna(inplace=True)
    return df, metadata


def correct_v_call(df: pd.DataFrame, metadata: Any) -> NoReturn:
    if metadata['Species'] != 'human':
        df['v_call'] = 'NOT_HUMAN'
    else:
        df['v_call'] = df['v_call'].str.slice(stop=5)


def make_annotated_df(df: pd.DataFrame, annotation: Annotation, metadata: Any = {}) -> pd.DataFrame:
    aa_columns = annotation.segmented_positions
    annotated_indexes, annotated_list = annotate_batch(
        df['sequence_alignment_aa'].tolist(), annotation, only_human=metadata['Species'] == 'human'
    )
    X = pd.DataFrame(annotated_list, columns=aa_columns)  # Make column for every aa
    y = df['v_call'][annotated_indexes]
    y.reset_index(drop=True, inplace=True)
    dataset = pd.concat([X, y], axis=1)
    nan_errors = dataset['v_call'].isna().sum()
    if nan_errors > 0:
        raise RuntimeError(f"Found {nan_errors} NaNs in target chain types")
    return dataset


def filter_df(df: pd.DataFrame, annotation: Annotation) -> pd.DataFrame:
    if len(annotation.required_positions) > 0:
        masks = [df[pos] == aa for pos, aa in annotation.required_positions.items()]
        mask = next(accumulate(masks, func=lambda a, b: a & b))
        return df[mask]
    return df


def read_heavy_datasets(input_dir: str, read_function: Callable[[str], pd.DataFrame]) -> List[pd.DataFrame]:
    logger.info("Dataset reading...")
    dfs = []
    original_data_size = 0
    file_names = os.listdir(input_dir)
    for input_file_name in tqdm(file_names[:5]):
        input_file_path = os.path.join(input_dir, input_file_name)
        df: pd.DataFrame = read_function(input_file_path)
        dfs.append(df)
        original_data_size += df.shape[0]
    logger.info(f"Original dataset: {original_data_size} rows")
    return dfs


def read_heavy_dataset(*args, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    dfs = read_heavy_datasets(*args, **kwargs)
    dataset = pd.concat(dfs, axis=0, ignore_index=True, copy=False)
    dataset.drop_duplicates(ignore_index=True, inplace=True)
    logger.info(f"Dataset: {dataset.shape[0]} rows (duplicates removed)")
    X = dataset.drop(['v_call'], axis=1)
    y = dataset['v_call']
    return X, y


def read_annotated_heavy_dataset(input_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
    return read_heavy_dataset(input_dir, pd.read_csv)


def merge_all_columns(df: pd.DataFrame) -> List[str]:
    values: List[List[str]] = df.to_numpy().tolist()
    result = list(map("".join, values))
    return result


def make_binary_target(y, target_v_type):
    return np.where(y.apply(lambda x: x == f"IGHV{target_v_type}"), 1, 0)
