import ablang2


ablang = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=1, device='cpu')


def get_ablang_embedding(seq: str):
    return ablang([seq, ''], mode='seqcoding')
