import sapiens


def get_sapiens_embedding(seq: str):
    return sapiens.predict_sequence_embedding(seq, "H", layer=-1)
