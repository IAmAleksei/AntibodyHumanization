import logging
import sys
from typing import List

import blosum
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from humanization import config_loader


AA_ALPHABET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
TABOO_INSERT_AA = 'C,P,X'
TABOO_DELETE_AA = 'C,P,X'
BLOSUM62 = blosum.BLOSUM(62)


def configure_logger(config: config_loader.Config, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(config.get(config_loader.LOGGING_LEVEL)))

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(logging.Formatter(fmt=config.get(config_loader.LOGGING_FORMAT)))

    logger.propagate = False
    logger.addHandler(ch)
    return logger


def parse_list(value: str) -> List[str]:
    return [x for x in value.split(",") if x != ""]


def read_sequences(input_file):
    if not input_file:
        sequence = input("Enter sequence: ")
        result = [("CONSOLE", sequence)]
    else:
        result = [(seq.name, str(seq.seq)) for seq in SeqIO.parse(input_file, 'fasta')]
    return result


def write_sequences(output_file, sequences):
    if not output_file:
        for name, result, _ in sequences:
            print(f'>{name}')
            print(result)
    else:
        seqs = [SeqRecord(Seq(seq), id=name, description='') for name, seq in sequences]
        SeqIO.write(seqs, output_file, 'fasta')
