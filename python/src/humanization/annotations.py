from typing import List, Tuple, Optional

import anarci

from humanization import patched_anarci

from humanization import config_loader
from humanization.models import GeneralChainType
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Annotations")


def segments_to_columns(segments: List[Tuple[str, List[str]]]) -> List[str]:
    result = []
    for segment_name, segment_positions in segments:
        for idx, _ in enumerate(segment_positions):
            result.append(f"{segment_name}_{idx + 1}")
    return result


class Annotation:
    name = "-"
    positions = []
    segments = []
    segmented_positions = []
    required_positions = {}
    v_gene_end = ""


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
    segmented_positions = segments_to_columns(segments)
    required_positions = {'fwr1_23': 'C', 'fwr2_15': 'W', 'fwr3_39': 'C'}
    v_gene_end = segmented_positions.index('fwr3_41')


class Imgt(Annotation):
    name = "imgt"
    positions = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
        '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
        '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55',
        '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73',
        '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91',
        '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108',
        '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124',
        '125', '126', '127', '128']
    segments = [
        (
            "fwr1",
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
             '20', '21', '22', '23', '24', '25']
        ),
        (
            "cdr1",
            ['26', '27', '28', '29', '30', '31', '32', '33']
        ),
        (
            "fwr2",
            ['34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50']
        ),
        (
            "cdr2",
            ['51', '52', '53', '54', '55', '56']
        ),
        (
            "fwr3",
            ['57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73',
             '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90',
             '91', '92']
        ),
        (
            "cdr3",
            ['93', '94', '95', '96', '97', '98', '99', '100', '101', '102']
        ),
        (
            "fwr4",
            ['103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117',
             '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128']
        ),
    ]
    segmented_positions = segments_to_columns(segments)


SEGMENTS_ORDER = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"]


def compare_positions(first: str, second: str):
    if first[:4] == second[:4]:  # AAAX_YY
        return int(first[5:]) < int(second[5:])
    else:
        return SEGMENTS_ORDER.index(first[:4]) < SEGMENTS_ORDER.index(second[:4])


def load_annotation(schema: str) -> Annotation:
    if schema == "chothia":
        return Chothia()
    elif schema == "imgt":
        return Imgt()
    else:
        raise RuntimeError("Unrecognized annotation type")


def annotate_batch(sequences: List[str], annotation: Annotation, chain_type: GeneralChainType,
                   is_human: bool = False) -> Tuple[List[int], List[List[str]]]:
    logger.debug(f"Anarci run on {len(sequences)} rows")
    sequences_ = list(enumerate(sequences))
    kwargs = {
        'ncpu': config.get(config_loader.ANARCI_NCPU),
        'scheme': annotation.name,
        'allow': {chain_type.value}
    }
    if is_human:
        kwargs['allowed_species'] = ['human']
    else:
        kwargs['allowed_species'] = ['mouse', 'rat', 'rabbit', 'rhesus', 'pig', 'alpaca']
    import sys
    sys.modules['anarci.anarci']._parse_hmmer_query = patched_anarci._parse_hmmer_query  # Monkey patching
    temp_res = anarci.run_anarci(sequences_, **kwargs)
    numerated_sequences = temp_res[1]
    logger.debug(f"Anarci run is finished")
    index_results = []
    prepared_results = []
    for i, numerated_seq in enumerate(numerated_sequences):
        assert sequences_[i][0] == temp_res[0][i][0]
        if numerated_seq is None:
            logger.warn(f"Bad sequence found #{i} {sequences[i]}")
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
    logger.debug(f"Anarci returned {len(index_results)} rows")
    return index_results, prepared_results


def annotate_single(sequence: str, annotation: Annotation, chain_type: GeneralChainType) -> Optional[List[str]]:
    _, annotated_seq = annotate_batch([sequence], annotation, chain_type)
    if len(annotated_seq) == 1:
        return annotated_seq[0]
    else:
        return None
