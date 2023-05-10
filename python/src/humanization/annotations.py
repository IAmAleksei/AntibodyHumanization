from typing import List, Tuple

from anarci import run_anarci

from humanization import config_loader
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Annotations")


class Annotation:
    name = "-"
    positions = []
    segments = []


class Chothia(Annotation):
    name = "chothia"
    positions = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
        "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "31A", "31B", "32", "33", "34", "35",
        "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "52A",
        "52B", "52C", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68",
        "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "82A", "82B", "82C", "83",
        "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "100A",
        "100B", "100C", "100D", "100E", "100F", "100G", "100H", "100I", "100J", "100K", "101", "102", "103", "104",
        "105", "106", "107", "108", "109", "110", "111", "112", "113"
    ]
    segments = [
        (
            "fwr1",
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
             "18", "19", "20", "21", "22", "23", "24", "25"]
        ),
        (
            "cdr1",
            ["26", "27", "28", "29", "30", "31", "31A", "31B", "32"]
        ),
        (
            "fwr2",
            ["33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
             "50", "51"]
        ),
        (
            "cdr2",
            ["52", "52A", "52B", "52C", "53", "54", "55", "56"]
        ),
        (
            "fwr3",
            ["57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73",
             "74", "75", "76", "77", "78", "79", "80", "81", "82", "82A", "82B", "82C", "83", "84", "85", "86", "87",
             "88", "89", "90", "91", "92", "93", "94"]
        ),
        (
            "cdr3",
            ["95", "96", "97", "98", "99", "100", "100A", "100B", "100C", "100D", "100E", "100F", "100G", "100H",
             "100I", "100J", "100K", "101", "102"]
        ),
        (
            "fwr4",
            ["103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113"]
        ),
    ]


def load_annotation(schema: str) -> Annotation:
    if schema == "chothia":
        return Chothia()
    else:
        raise RuntimeError("Unrecognized annotation type")


def segments_to_columns(annotation: Annotation) -> List[str]:
    result = []
    for segment_name, segment_positions in annotation.segments:
        for idx, _ in enumerate(segment_positions):
            result.append(f"{segment_name}_{idx + 1}")
    return result


def annotate_batch(sequences: List[str], annotation: Annotation) -> Tuple[List[int], List[List[str]]]:
    logger.debug(f"Anarci run on {len(sequences)} rows")
    sequences_ = list(enumerate(sequences))
    numerated_sequences = run_anarci(sequences_, ncpu=4, scheme=annotation.name)[1]
    logger.debug(f"Anarci run is finished")
    index_results = []
    prepared_results = []
    for i, numerated_seq in enumerate(numerated_sequences):
        if numerated_seq is None:
            logger.warn(f"Bad sequence found")
        else:
            a = numerated_seq[0]
            b = a[0]
            seq_dict = {f"{idx}{letter.strip()}": aa for (idx, letter), aa in b if aa != "-"}
            result_seq = [
                seq_dict.get(position, "X")
                for segment_name, segment_positions in annotation.segments for position in segment_positions
            ]
            index_results.append(i)
            prepared_results.append(result_seq)
    return index_results, prepared_results
