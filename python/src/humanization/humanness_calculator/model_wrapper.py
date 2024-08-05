import json
import os.path
import pickle
from datetime import datetime
from typing import Dict, Union

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from humanization.common.annotations import Annotation, load_annotation, ChainType, GeneralChainType
from humanization.dataset.one_hot_encoder import one_hot_encode


class ModelWrapper:
    def __init__(self, chain_type: ChainType, model: Union[CatBoostClassifier, RandomForestClassifier],
                 annotation: Annotation, threshold: float):
        self.chain_type = chain_type
        self.model = model
        self.annotation = annotation
        self.threshold = threshold

    def library(self):
        return 'catboost' if isinstance(self.model, CatBoostClassifier) else 'sklearn'

    def predict_proba(self, data):
        return self.model.predict_proba(one_hot_encode(self.annotation, data, lib=self.library(),
                                                       cat_features=self.annotation.segmented_positions))


def get_model_name(chain_type: ChainType) -> str:
    return f"{chain_type.full_type()}.cbm"


def get_meta_name(chain_type: ChainType) -> str:
    return f"{chain_type.full_type()}_meta.json"


def save_model(model_dir: str, wrapped_model: ModelWrapper):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, get_model_name(wrapped_model.chain_type))
    meta_path = os.path.join(model_dir, get_meta_name(wrapped_model.chain_type))
    if wrapped_model.library() == 'catboost':
        wrapped_model.model.save_model(model_path)
    else:
        with open(model_path, 'wb') as f:
            pickle.dump(wrapped_model.model, f)
    with open(meta_path, 'w') as file:
        json.dump(
            {
                'schema': wrapped_model.annotation.name,
                'chain_type': wrapped_model.chain_type.full_type(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'threshold': wrapped_model.threshold,
                'library': wrapped_model.library(),
            }, file
        )


def load_model(model_dir, chain_type: ChainType) -> ModelWrapper:
    model_path = os.path.join(model_dir, get_model_name(chain_type))
    meta_path = os.path.join(model_dir, get_meta_name(chain_type))
    with open(meta_path, 'r') as file:
        desc = json.load(file)
    annotation = load_annotation(desc['schema'], chain_type.general_type().kind())
    if desc.get('library', 'catboost') == 'catboost':
        model = CatBoostClassifier()
        model.load_model(model_path)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    model_wrapper = ModelWrapper(chain_type, model, annotation, desc['threshold'])
    return model_wrapper


def load_all_models(model_dir, general_type: GeneralChainType) -> Dict[ChainType, ModelWrapper]:
    chain_types = [general_type.specific_type(v_type) for v_type in general_type.available_specific_types()]
    models = {}
    for chain_type in chain_types:
        try:
            models[chain_type] = load_model(model_dir, chain_type)
        except Exception:
            pass
    return models
