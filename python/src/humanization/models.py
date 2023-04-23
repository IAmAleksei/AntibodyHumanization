import os.path
from enum import Enum

from catboost import CatBoostClassifier


class ChainType(Enum):
    pass


class HeavyChainType(ChainType):
    V1 = "1"
    V2 = "2"
    V3 = "3"
    V4 = "4"
    V5 = "5"
    V6 = "6"
    V7 = "7"


class LightChainType(ChainType):
    KAPPA = "kappa"
    LAMBDA = "lambda"


class ModelWrapper:
    def __init__(self, chain_type: ChainType, model: CatBoostClassifier):
        self.chain_type = chain_type
        self.model = model


def get_model_name(chain_type: ChainType) -> str:
    if isinstance(chain_type, HeavyChainType):
        return f"heavy_v{chain_type.value}.cbm"
    elif isinstance(chain_type, LightChainType):
        return f"light_{chain_type.value}.cbm"
    else:
        raise RuntimeError("Unrecognized chain type")


def save_model(model_dir: str, wrapped_model: ModelWrapper):
    model_path = os.path.join(model_dir, get_model_name(wrapped_model.chain_type))
    wrapped_model.model.save_model(model_path)


def load_model(model_dir, chain_type: ChainType) -> CatBoostClassifier:
    model_path = os.path.join(model_dir, get_model_name(chain_type))
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model
