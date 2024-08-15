from sklearn.preprocessing import OneHotEncoder

from humanization.common.annotations import Annotation
from humanization.common.utils import AA_ALPHABET


def get_encoder(a: Annotation) -> OneHotEncoder:
    return OneHotEncoder(categories=[AA_ALPHABET for _ in a.segmented_positions], handle_unknown='ignore')


def one_hot_encode(annotation, X, y=None, lib='catboost', **kwargs):
    if lib == 'catboost':
        from catboost import Pool

        return Pool(data=X, label=y, **kwargs)
    if lib == 'sklearn':
        return get_encoder(annotation).fit_transform(X)
    raise RuntimeError("Unrecognized library")


def one_hot_encode_pred(annotation, X, lib='catboost'):
    if lib == 'catboost':
        return X
    if lib == 'sklearn':
        return get_encoder(annotation).fit_transform(X)
    raise RuntimeError("Unrecognized library")
