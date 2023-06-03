import logging
import sys

from humanization import config_loader


AA_ALPHABET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
TABOO_INSERT_AA = ['C', 'P', 'X']
TABOO_DELETE_AA = ['C', 'P', 'X']


def configure_logger(config: config_loader.Config, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(config.get(config_loader.LOGGING_LEVEL)))

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(logging.Formatter(fmt=config.get(config_loader.LOGGING_FORMAT)))

    logger.propagate = False
    logger.addHandler(ch)
    return logger


