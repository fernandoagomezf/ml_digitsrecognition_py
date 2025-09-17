from domain.digits import HyperParameters
from domain.digits import Result
from domain.digits import KNNModel
import json 
import joblib
import sklearn

class ClassifierRepository():
    _model_path : str
    def __init__(self, model_path:str):
        if model_path is None or model_path == "":
            raise ValueError("The model path cannot be null or empty")        
        self._model_path = model_path

    def get_model_path(self) -> str:
        return self._model_path


class KNNRepository(ClassifierRepository):
    def __init__(self):
        super().__init__("knn_model.joblib")

    def get_model(self) -> KNNModel:
        payload = joblib.load(self.get_model_path())
        model = KNNModel()
        model._knn = payload.get("model")
        if (model._knn is None):
            raise ValueError("Failed to load the model from the specified path")
        model._params = payload.get("params")
        if (model._params is None):
            raise ValueError("Failed to load the parameters from the specified path")
        model._result = payload.get("result")
        if (model._result is None):
            raise ValueError("Failed to load the result from the specified path")

    def update(self, model:KNNModel) -> None:
        if model is None:
            raise ValueError("Model cannot be null")
        try:
            payload = { 
                "model": model._knn ,
                "params" : model._params,
                "result": model._result
            }
            joblib.dump(payload, self.get_model_path())
        except Exception as e:
            raise IOError(f"Failed to save model: {e}")
