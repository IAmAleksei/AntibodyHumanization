from catboost import Pool
from sklearn.preprocessing import OneHotEncoder

from humanization.common.annotations import Annotation
from humanization.common.utils import AA_ALPHABET


def get_encoder(a: Annotation) -> OneHotEncoder:
    return OneHotEncoder(categories=[AA_ALPHABET for _ in a.segmented_positions], handle_unknown='ignore')


def one_hot_encode(annotation, X, y=None, lib='catboost', **kwargs):
    if lib == 'catboost':
        return Pool(data=X, label=y, **kwargs)
    elif lib == 'sklearn':
        return get_encoder(annotation).transform(X)
    else:
        raise RuntimeError("Unrecognized library")
