from abnumber.chain import Chain
from biophi.humanization.methods.humanness import ChainHumanness, OASisParams, get_chain_humanness


def get_oasis_humanness(db_path, seq):
    chain = Chain(seq, scheme="chothia", cdr_definition="chothia", name="protein")
    params = OASisParams(db_path, 0.5)
    return get_chain_humanness(chain, params)
