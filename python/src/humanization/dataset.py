import logging
from typing import NoReturn, List

import pandas

logger = logging.getLogger("Dataset reader")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)


def read_dataset(
        dataset_list_path: str = "/Users/alex-shishkin/PycharmProjects/science/bash/h_data.txt",
        requested_columns: List[str] = None) -> pandas.DataFrame:
    requested_columns = requested_columns if requested_columns is not None else ['sequence_alignment_aa']
    logger.debug("DATASET PROCESSING")
    dfs = []
    original_data_size = 0
    with open(dataset_list_path, 'r') as heavy_data_list:
        dataset_list = heavy_data_list.readlines()
        for dataset_path in dataset_list:
            try:
                dataset_path = dataset_path.strip()
                if len(dataset_path) == 0:
                    continue  # Drop empty lines from dataset paths
                df: pandas.DataFrame = pandas.read_csv(dataset_path, skiprows=1)  # Drop row with running info
                original_data_size += df.size
                mask1 = df['fwr1_aa'].apply(lambda x: len(x) >= 24 and x[21] == "C")  # TODO: Why is it shifted? (pos(C) = 23)
                mask2 = df['fwr3_aa'].apply(lambda x: len(x) < 39 or x[38] == "C")
                df = df[mask1 & mask2]
                df = df[requested_columns]
                dfs.append(df)
            except Exception as e:
                logger.error(e, exc_info=True)
    df = pandas.concat(dfs, axis=0, ignore_index=True).drop_duplicates(subset=['sequence_alignment_aa'])
    logger.debug(f"ORIGINAL DATASET SIZE: {original_data_size}")
    logger.debug(f"FILTERED DATASET SIZE: {sum(map(lambda df: df.size, dfs))} (cysteine errors removed)")
    logger.debug(f"FINAL DATASET SIZE: {df.size} (duplicates removed)")
    # TODO: Generate sequence by expanded fwr1_aa cdr1_aa fwr2_aa cdr2_aa fwr3_aa cdr3_aa fwr4_aa
    return df


if __name__ == '__main__':
    read_dataset()
