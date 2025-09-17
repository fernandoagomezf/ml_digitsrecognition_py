from domain.digits import HyperParameters
from domain.digits import Result
from domain.digits import KNNModel
from keras.models import load_model
from json import dump
from json import load

class ClassifierRepository():
    _model_path : str
    _params_path : str
    _result_path : str
    def __init__(self, model_path:str, params_path:str, result_path:str):
        if model_path is None or model_path == "":
            raise ValueError("The model path cannot be null or empty")
        if params_path is None or params_path == "":
            raise ValueError("The parameters path cannot be null or empty")
        if result_path is None or result_path == "":
            raise ValueError("The result path cannot be null or empty")
        self._model_path = model_path
        self._params_path = params_path
        self._result_path = result_path

    def get_model_path(self) -> str:
        return self._model_path

    def get_params_path(self) -> str:
        return self._params_path

    def get_result_path(self) -> str:
        return self._result_path
    
    def _load_params(self) -> HyperParameters:
        try:
            with open(self.get_params_path(), "r", encoding="utf-8") as f:
                params_dict = load(f)
                return HyperParameters(**params_dict)
        except Exception as e:
            raise IOError(f"Failed to load parameters: {e}")
        
    def _load_result(self) -> Result:
        try:
            with open(self.get_result_path(), "r", encoding="utf-8") as f:
                result_dict = load(f)
                return Result(**result_dict)
        except Exception as e:
            raise IOError(f"Failed to load result: {e}")
    
    def _save_params(self, params: HyperParameters) -> None:
        if params is None:
            raise ValueError("Parameters cannot be null")
        try:
            with open(self.get_params_path(), "w", encoding="utf-8") as f:
                dump(params, f, ensure_ascii=False, indent=4)
        except Exception as e:
            raise IOError(f"Failed to save parameters: {e}")
        
    def _save_result(self, result: Result) -> None:
        if result is None:
            raise ValueError("Result cannot be null")
        try:
            with open(self.get_result_path(), "w", encoding="utf-8") as f:
                dump(result, f, ensure_ascii=False, indent=4)
        except Exception as e:
            raise IOError(f"Failed to save result: {e}")

class KNNRepository(ClassifierRepository):
    def __init__(self):
        super().__init__("knn_model.h5", "knn_params.json", "knn_results.json")

    def get_params(self) -> HyperParameters:
        return self._load_params()
    
    def get_results(self) -> Result:
        return self._load_result()

    def get_model(self) -> KNNModel:
        model = KNNModel()
        model._knn = load_model(self.get_model_path())
        if (model._knn is None):
            raise ValueError("Failed to load the model from the specified path")
        model._params = self._load_params()
        model._result = self._load_result()

    def update(self, model:KNNModel) -> None:
        if model is None:
            raise ValueError("Model cannot be null")
        try:
            model._knn.save(self.get_model_path())
            self._save_params(model._params)
            self._save_result(model._result)
        except Exception as e:
            raise IOError(f"Failed to save model: {e}")
