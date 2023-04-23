import logging
import sys

from humanization import config_loader

SEGMENTS = [
    ('fwr1', 26), ('cdr1', 16), ('fwr2', 29), ('cdr2', 18),
    ('fwr3', 39), ('cdr3', 36), ('fwr4', 11)
]

AA_ALPHABET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '❤️']


def configure_logger(config: config_loader.Config, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(config.get(config_loader.LOGGING_LEVEL)))

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(logging.Formatter(fmt=config.get(config_loader.LOGGING_FORMAT)))

    logger.propagate = False
    logger.addHandler(ch)
    return logger


