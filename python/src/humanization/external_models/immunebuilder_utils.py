import numpy as np
from ImmuneBuilder import NanoBodyBuilder2


def get_immunebuilder_embedding(seq: str):
    antibody = NanoBodyBuilder2(numbering_scheme='chothia').predict({'H': seq})
    embedding = np.average(antibody.encodings[0], axis=0)
    return embedding
