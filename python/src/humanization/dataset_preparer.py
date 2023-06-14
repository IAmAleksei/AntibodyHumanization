import argparse
import os
from typing import NoReturn, Tuple, Optional, List

import pandas

from humanization import config_loader
from humanization.annotations import load_annotation, Annotation
from humanization.dataset import read_file, correct_v_call, make_annotated_df, filter_df, read_heavy_dataset, \
    read_annotated_heavy_dataset, merge_all_columns
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Dataset preparer")


def read_and_annotate_file(csv_file: str, annotation: Annotation) -> pandas.DataFrame:
    df, metadata = read_file(csv_file, ['sequence_alignment_aa', 'v_call'])
    correct_v_call(df, metadata)
    df = make_annotated_df(df, annotation)
    df = filter_df(df, annotation)
    return df


def read_raw_heavy_dataset(input_dir: str, annotation: Annotation) -> Tuple[pandas.DataFrame, pandas.Series]:
    return read_heavy_dataset(input_dir, lambda csv_file: read_and_annotate_file(csv_file, annotation))


def read_any_heavy_dataset(input_dir: str, annotated_data: bool,
                           annotation: Annotation) -> Tuple[pandas.DataFrame, pandas.Series]:
    if annotated_data:
        logger.info(f"Use annotated-data mode")
        logger.info(f"Please check that `{annotation.name}` is defined correctly")
        return read_annotated_heavy_dataset(input_dir)
    else:
        logger.info(f"Use raw-data mode")
        return read_raw_heavy_dataset(input_dir, annotation)


def read_human_samples(dataset_file=None, annotated_data=None, annotation=None) -> Optional[List[str]]:
    if dataset_file is not None:
        X, y = read_any_heavy_dataset(dataset_file, annotated_data, annotation)
        df = X[y != 'NOT_HUMAN']
        df.reset_index(drop=True, inplace=True)
        human_samples = merge_all_columns(df)
        return human_samples
    else:
        return None


def main(input_dir: str, schema: str, output_dir: str, skip_existing: bool) -> NoReturn:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    annotation = load_annotation(schema)
    file_names = os.listdir(input_dir)
    logger.info(f"{len(file_names)} files found")
    for input_file_name in file_names:
        input_file_path = os.path.join(input_dir, input_file_name)
        output_file_path = os.path.join(output_dir, input_file_name)
        if skip_existing and os.path.exists(output_file_path):
            logger.debug(f"Processed {input_file_name} exists")
            continue
        logger.debug(f"Processing {input_file_name}...")
        try:
            df = read_and_annotate_file(input_file_path, annotation)
            df.to_csv(output_file_path, index=False)
            logger.debug(f"Result with {df.shape[0]} rows saved to {output_file_path}")
        except Exception as err:
            logger.error(f"Processing error: {str(err)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Dataset preparer''')
    parser.add_argument('input', type=str, help='Path to input folder with .csv files')
    parser.add_argument('schema', type=str, help='Annotation schema')
    parser.add_argument('output', type=str, help='Path to output folder')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing processed files')
    parser.add_argument('--process-existing', dest='skip_existing', action='store_false')
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()

    main(input_dir=args.input,
         schema=args.schema,
         output_dir=args.output,
         skip_existing=args.skip_existing)
