import json
import os.path
from datetime import datetime
from enum import Enum

from catboost import CatBoostClassifier

from humanization.annotations import Annotation, load_annotation


class GeneralChainType(Enum):
    HEAVY = "H"
    KAPPA = "K"
    LAMBDA = "L"

    def specific_type(self, v_type):
        if self == GeneralChainType.HEAVY:
            return HeavyChainType(str(v_type))
        elif self == GeneralChainType.KAPPA:
            return KappaChainType(str(v_type))
        elif self == GeneralChainType.LAMBDA:
            return LambdaChainType(str(v_type))
        else:
            raise RuntimeError("Unrecognized chain type")


class ChainType(Enum):
    @classmethod
    def general_type(cls):
        ...

    def full_type(self):
        return f"{self.general_type().value}V{self.value}"


class HeavyChainType(ChainType):
    V1 = "1"
    V2 = "2"
    V3 = "3"
    V4 = "4"
    V5 = "5"
    V6 = "6"
    V7 = "7"

    def general_type(self):
        return GeneralChainType.HEAVY


class LightChainType(ChainType):
    pass


class KappaChainType(LightChainType):
    V1 = "1"

    def general_type(self):
        return GeneralChainType.KAPPA


class LambdaChainType(LightChainType):
    V1 = "1"

    def general_type(self):
        return GeneralChainType.LAMBDA


class ModelWrapper:
    def __init__(self, chain_type: ChainType, model: CatBoostClassifier, annotation: Annotation, threshold: float):
        self.chain_type = chain_type
        self.model = model
        self.annotation = annotation
        self.threshold = threshold


def get_model_name(chain_type: ChainType) -> str:
    return f"{chain_type.full_type()}.cbm"


def get_meta_name(chain_type: ChainType) -> str:
    return f"{chain_type.full_type()}_meta.json"


def save_model(model_dir: str, wrapped_model: ModelWrapper):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, get_model_name(wrapped_model.chain_type))
    meta_path = os.path.join(model_dir, get_meta_name(wrapped_model.chain_type))
    wrapped_model.model.save_model(model_path)
    with open(meta_path, 'w') as file:
        json.dump(
            {
                'schema': wrapped_model.annotation.name,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'threshold': wrapped_model.threshold
            }, file
        )


def load_model(model_dir, chain_type: ChainType) -> ModelWrapper:
    model_path = os.path.join(model_dir, get_model_name(chain_type))
    meta_path = os.path.join(model_dir, get_meta_name(chain_type))
    with open(meta_path, 'r') as file:
        desc = json.load(file)
    annotation = load_annotation(desc['schema'])
    model = CatBoostClassifier()
    model.load_model(model_path)
    model_wrapper = ModelWrapper(chain_type, model, annotation, desc['threshold'])
    return model_wrapper

