import ablang2
import numpy as np
import torch

ablang = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=1, device='cpu')


def get_ablang_embedding(seq: str):
    return ablang([seq, ''], mode='seqcoding')


def get_attentions(vh: str):
    length = len(vh)
    tokenized_seqs = ablang.tokenizer([f"<{vh}>|"], pad=True, w_extra_tkns=False)
    with torch.no_grad():
        attn_weights = ablang.AbLang.forward(tokenized_seqs, return_attn_weights=True)[-1][0]
        result = []
        for pos in range(length):
            token = tokenized_seqs[0][pos + 1]  # Ignoring first token '<'
            result.append(attn_weights[token - 1, 1:length + 1, pos + 1])
        return np.array(result)
