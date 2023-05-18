import json
import os.path
from datetime import datetime
from enum import Enum

from catboost import CatBoostClassifier

from humanization.annotations import Annotation, load_annotation


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
    def __init__(self, chain_type: ChainType, model: CatBoostClassifier, annotation: Annotation):
        self.chain_type = chain_type
        self.model = model
        self.annotation = annotation


def get_model_name(chain_type: ChainType) -> str:
    if isinstance(chain_type, HeavyChainType):
        return f"heavy_v{chain_type.value}.cbm"
    elif isinstance(chain_type, LightChainType):
        return f"light_{chain_type.value}.cbm"
    else:
        raise RuntimeError("Unrecognized chain type")


def get_meta_name(chain_type: ChainType) -> str:
    if isinstance(chain_type, HeavyChainType):
        return f"heavy_v{chain_type.value}_meta.json"
    elif isinstance(chain_type, LightChainType):
        return f"light_{chain_type.value}_meta.json"
    else:
        raise RuntimeError("Unrecognized chain type")


def save_model(model_dir: str, wrapped_model: ModelWrapper):
    model_path = os.path.join(model_dir, get_model_name(wrapped_model.chain_type))
    meta_path = os.path.join(model_dir, get_meta_name(wrapped_model.chain_type))
    wrapped_model.model.save_model(model_path)
    with open(meta_path, 'w') as file:
        schema = wrapped_model.annotation.name
        json.dump({'schema': schema, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, file)


def load_model(model_dir, chain_type: ChainType) -> ModelWrapper:
    model_path = os.path.join(model_dir, get_model_name(chain_type))
    meta_path = os.path.join(model_dir, get_meta_name(chain_type))
    with open(meta_path, 'r') as file:
        d = json.load(file)
    schema = d['schema']
    annotation = load_annotation(schema)
    model = CatBoostClassifier()
    model.load_model(model_path)
    model_wrapper = ModelWrapper(chain_type, model, annotation)
    return model_wrapper

