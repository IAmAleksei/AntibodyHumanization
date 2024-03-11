import os
import sys

from abnumber.chain import Chain


def get_oasis_humanness(biophi_path, seq):
    if biophi_path not in sys.path:
        sys.path.append(biophi_path)
    from biophi.humanization.methods.humanness import ChainHumanness, OASisParams, get_chain_humanness
    chain = Chain(seq, scheme="chothia", cdr_definition="chothia", name="protein")
    params = OASisParams(os.path.join(biophi_path, "OASis_9mers_v1.db"), 0.5)
    return get_chain_humanness(chain, params)
