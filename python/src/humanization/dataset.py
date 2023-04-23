import json
from typing import List

import numpy as np
import pandas

from humanization import config_loader, utils
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Dataset reader")


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
                metadata = json.loads(','.join(pandas.read_csv(dataset_path, nrows=0).columns))
                df: pandas.DataFrame = pandas.read_csv(dataset_path, skiprows=1)  # Drop row with running info
                original_data_size += df.shape[0]
                df['species'] = metadata['Species']
                df = df.dropna(subset=requested_columns)
                mask1 = df['fwr1_aa'].apply(lambda x: 24 <= len(x) <= 26 and x[21] == "C")  # TODO: Why is it shifted? (pos(C) = 23)
                mask2 = df['fwr3_aa'].apply(lambda x: len(x) < 39 or x[38] == "C")
                df = df[mask1 & mask2]
                df = df[requested_columns]
                dfs.append(df)
                logger.debug(f"{dataset_path} has been read ({df.shape[0]} rows)")
            except Exception as e:
                logger.error(e, exc_info=True)
    df = pandas.concat(dfs, axis=0, ignore_index=True).drop_duplicates(subset=['sequence_alignment_aa'])
    logger.debug(f"Original dataset: {original_data_size} rows")
    logger.debug(f"Filtered dataset: {sum(map(lambda df: df.shape[0], dfs))} rows (cysteine errors removed)")
    logger.info(f"Final dataset: {df.shape[0]} rows (duplicates removed)")
    # TODO: Generate sequence by expanded fwr1_aa cdr1_aa fwr2_aa cdr2_aa fwr3_aa cdr3_aa fwr4_aa
    return df


def read_prepared_heavy_dataset():
    dataset = read_dataset(requested_columns=[
        'sequence_alignment_aa', 'fwr1_aa', 'cdr1_aa', 'fwr2_aa', 'cdr2_aa', 'fwr3_aa', 'cdr3_aa', 'fwr4_aa',
        'species', 'v_call'
    ])
    segments = []
    for segment_name, _ in utils.SEGMENTS:
        df_col = dataset[f'{segment_name}_aa'].str.split('', expand=True)
        df_col = df_col.drop(columns=[0, len(df_col.columns) - 1]).add_prefix(f'{segment_name}_')
        logger.debug(f"Segment {segment_name} has {len(df_col.columns)} columns")
        # First and last columns are empty always
        segments.append(df_col)
    dataset['v_call'] = dataset['v_call'].str.slice(stop=5)
    dataset.loc[dataset['species'] != 'human', 'v_call'] = 'NOT_HUMAN'
    # X = dataset['sequence_alignment_aa'].str.split('', expand=True)  # Make column for every aa
    X = pandas.concat(segments, axis=1)  # Make column for every aa
    X = X.fillna(value='')
    y = dataset['v_call']
    return X, y


def make_binary_target(y, target_v_type):
    return np.where(y.apply(lambda x: x == f"IGHV{target_v_type}"), 1, 0)


if __name__ == '__main__':
    read_dataset()
