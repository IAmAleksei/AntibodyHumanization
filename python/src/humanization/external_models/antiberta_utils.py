from typing import List

import numpy as np
from transformers import pipeline, RoFormerForMaskedLM, RoFormerTokenizer

model_name = "alchemab/antiberta2-cssp"
antiberta_tokenizer = RoFormerTokenizer.from_pretrained(model_name)
antiberta_model = RoFormerForMaskedLM.from_pretrained(model_name)
filler = pipeline("fill-mask", model=antiberta_model, tokenizer=antiberta_tokenizer)


def get_antiberta_embeddings(seqs: List[str], get_attention: bool = False) -> np.array:
    inputs = antiberta_tokenizer(seqs, return_tensors="pt")
    outputs = antiberta_model(**inputs, output_hidden_states=True, output_attentions=True)
    embedding_2d = outputs.hidden_states[-1][:, 0, :].detach().numpy()  # layer, seq_no, token (CLS), embedding
    if get_attention:
        return embedding_2d, outputs.attentions
    else:
        return embedding_2d


def get_antiberta_embedding(seq: str) -> np.array:
    return get_antiberta_embeddings([seq])[0, :]


def fill_mask(seq: str) -> str:
    result = filler(seq)
    return result[0]['sequence'].replace(' ', '').replace('Ḣ', 'H')


def get_mask_values(seq: List[str]) -> List[str]:
    result = filler(seq)
    if len(seq) == 1:
        result = [result]
    return [res[0]['token_str'].replace('Ḣ', 'H') for res in result]


def get_mask_value(seq: str) -> str:
    return get_mask_values([seq])[0]
