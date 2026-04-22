
from model_training.models.base_model import BaseSepsisModel

MODEL_REGISTRY = {}

def get_model(model_name, config, model_params, features) -> BaseSepsisModel:
    # Support XGBoost and LSTM lazily to avoid unused imports
    if model_name == "xgboost" and "xgboost" not in MODEL_REGISTRY:
        from .xgboost_model import XGBoostWrapper
        MODEL_REGISTRY["xgboost"] = XGBoostWrapper
    elif model_name == "lightgbm" and "lightgbm" not in MODEL_REGISTRY:
        from .lightgbm_model import LightGBMWrapper
        MODEL_REGISTRY["lightgbm"] = LightGBMWrapper
    elif model_name == "lstm" and "lstm" not in MODEL_REGISTRY:
        from .lstm_model import LSTMModelWrapper
        MODEL_REGISTRY["lstm"] = LSTMModelWrapper
    elif model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in factory registry.")

    return MODEL_REGISTRY[model_name](config, model_params, features)
