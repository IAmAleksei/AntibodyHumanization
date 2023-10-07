import json
import os.path
from datetime import datetime

from catboost import CatBoostClassifier

from humanization.annotations import Annotation, load_annotation, ChainType


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
                'chain_type': wrapped_model.chain_type.full_type(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'threshold': wrapped_model.threshold
            }, file
        )


def load_model(model_dir, chain_type: ChainType) -> ModelWrapper:
    model_path = os.path.join(model_dir, get_model_name(chain_type))
    meta_path = os.path.join(model_dir, get_meta_name(chain_type))
    with open(meta_path, 'r') as file:
        desc = json.load(file)
    annotation = load_annotation(desc['schema'], chain_type.general_type().kind())
    model = CatBoostClassifier()
    model.load_model(model_path)
    model_wrapper = ModelWrapper(chain_type, model, annotation, desc['threshold'])
    return model_wrapper
