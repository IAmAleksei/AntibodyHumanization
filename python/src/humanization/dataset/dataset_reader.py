import argparse
import json
import os
from typing import NoReturn, Tuple, Optional, List, Dict

import pandas

from humanization.common import config_loader
from humanization.common.annotations import load_annotation, Annotation, ChainKind, ChainType
from humanization.common.utils import configure_logger
from humanization.dataset.dataset_preparer import read_oas_file, correct_v_call, make_annotated_df, filter_df, \
    merge_all_columns, read_annotated_dataset, read_dataset, read_imgt_file, NA_SPECIES, mark_another_species

config = config_loader.Config()
logger = configure_logger(config, "Dataset preparer")


def read_and_annotate_file(csv_file: str, annotation: Annotation) -> Tuple[pandas.DataFrame, Dict[str, str]]:
    if csv_file.endswith(".imgt"):
        df, species = read_imgt_file(csv_file)
    else:
        df, species = read_oas_file(csv_file, ['sequence_alignment_aa', 'v_call'])
    logger.debug(f"File contains {df.shape[0]} rows")
    correct_v_call(df)
    df = make_annotated_df(df, annotation)
    df = filter_df(df, annotation)
    metadata = {"Species": species}
    return df, metadata


def read_raw_dataset(input_dir: str, annotation: Annotation, species: str = 'human',
                     **kwargs) -> Tuple[pandas.DataFrame, pandas.Series]:
    def process_file(csv_file):
        df, metadata = read_and_annotate_file(csv_file, annotation)
        mark_another_species(df, metadata, species)
        return df

    return read_dataset(input_dir, process_file, **kwargs)


def read_any_dataset(input_dir: str, annotation: Annotation, species: str = 'human', drop_another: bool = False,
                     v_type: Optional[ChainType] = None) -> Tuple[pandas.DataFrame, pandas.Series]:
    annotated_data = "_annotated" in os.path.basename(input_dir)
    if annotated_data:
        logger.info(f"Use annotated-data mode")
        logger.info(f"Please check that `{annotation.name}` is defined correctly")
        X, y = read_annotated_dataset(input_dir, species=species, drop_another=drop_another, v_type=v_type)
    else:
        logger.info(f"Use raw-data mode")
        X, y = read_raw_dataset(input_dir, annotation, species=species, drop_another=drop_another, v_type=v_type)
    if X.isna().sum().sum() > 0 or y.isna().sum() > 0:
        raise RuntimeError("Found nans")
    if X.shape[0] != y.shape[0]:
        raise RuntimeError(f"X has shape {X.shape}, but y has shape {y.shape}")
    if v_type is not None and not y.eq(v_type.oas_type()).all():
        raise RuntimeError("If v gene is specified then all v_call values must be equal to that")
    return X, y


def read_v_gene_dataset(dataset_file=None, annotation=None, only_human=True,
                        v_type: ChainType = None) -> Optional[Tuple[List[str], List[str]]]:
    if dataset_file is not None:
        X, y = read_any_dataset(dataset_file, annotation, drop_another=only_human, v_type=v_type)
        X.reset_index(drop=True, inplace=True)
        human_samples = merge_all_columns(X)
        return human_samples, y.tolist()
    else:
        return None


def main(input_dir: str, chain_kind: ChainKind, schema: str, output_dir: str, skip_existing: bool) -> NoReturn:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    annotation = load_annotation(schema, chain_kind)
    file_names = os.listdir(input_dir)
    logger.info(f"{len(file_names)} files found")
    for input_file_name in file_names:
        input_file_path = os.path.join(input_dir, input_file_name)
        output_file_path = os.path.join(output_dir, input_file_name)
        if output_file_path.endswith('.gz'):
            output_file_path = output_file_path[:-3]
        if skip_existing and os.path.exists(output_file_path):
            logger.debug(f"Processed {input_file_name} exists")
            continue
        logger.debug(f"Processing {input_file_name}...")
        try:
            df, metadata = read_and_annotate_file(input_file_path, annotation)
            with open(output_file_path, 'w') as file:
                file.write(json.dumps(metadata) + "\n")
            df.to_csv(output_file_path, index=False, mode='a')
            logger.debug(f"Dataframe {metadata} with {df.shape[0]} rows saved to {output_file_path}")
        except Exception:
            logger.exception(f"Processing error")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Dataset preparer''')
    parser.add_argument('input', type=str, help='Path to input folder with .csv files')
    parser.add_argument('chain_kind', type=str, choices=[c_kind.value for c_kind in ChainKind], help='Chain kind')
    parser.add_argument('schema', type=str, help='Annotation schema')
    parser.add_argument('output', type=str, help='Path to output folder')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing processed files')
    parser.add_argument('--process-existing', dest='skip_existing', action='store_false')
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()

    main(input_dir=args.input,
         chain_kind=ChainKind(args.chain_kind),
         schema=args.schema,
         output_dir=args.output,
         skip_existing=args.skip_existing)
