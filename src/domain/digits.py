from typing import Any

from datetime import datetime
from dataclasses import dataclass
from dataclasses import replace
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

@dataclass
class Data:
    set: Any
    target: Any
    train_features: Any
    train_target: Any
    test_features: Any
    test_target: Any

@dataclass(frozen=True)
class Result:
    normal_accuracy: float
    crossval_accuracy: float
    datetime: datetime
    feature_count: int
    total_records: int
    train_records: int
    test_records: int
    error_count: int
    confusion_matrix: Any

@dataclass(frozen=True)
class HyperParameters:
    seed:int
    test_size:float
    k_neighbors:int
    k_fold:int

class KNNModel():
    _params: HyperParameters
    _data: Data
    _result: Result
    _knn: KNeighborsClassifier

    def __init__(self):
        self._params = HyperParameters(
            seed=0, 
            test_size=0.3, 
            k_neighbors=3,
            k_fold=0
        )
        self._data = Data(None, None, None, None, None, None)
        self._knn = KNeighborsClassifier(n_neighbors=self._params.k_neighbors)
        self._result = None

    def read_data(self):
        digits = load_digits()
        self._data = replace(self._data, set=digits.data, target=digits.target)

    def get_seed(self) -> int:
        return self._params.seed

    def set_seed(self, seed:int) -> None:
        if seed > 0:
            self._params = replace(self._params, seed=seed)
        else:
            raise ValueError("El número para sembrar el generador aleatorio debe ser mayor a cero.")

    def get_test_size(self) -> float:
        return self._params.test_size

    def set_test_size(self, test_size:float) -> None:
        if 0.0 < test_size < 1.0:
            self._params = replace(self._params, test_size=test_size)
        else:
            raise ValueError("El tamaño de la prueba debe estar entre 0 y 1.")
        
    def get_k_neighbors(self) -> int:
        return self._params.k_neighbors
    
    def set_k_neighbors(self, k_neighbors:int) -> None:
        if k_neighbors > 0:
            self._params = replace(self._params, k_neighbors=k_neighbors)
            self._knn.set_params(n_neighbors=k_neighbors)
        else:
            raise ValueError("El número de vecinos debe ser mayor a cero.")
        
    def get_k_fold(self) -> int:
        return self._params.k_fold
    
    def set_k_fold(self, k_fold:int) -> None:
        if k_fold >= 0:
            self._params = replace(self._params, k_fold=k_fold)
        else:
            raise ValueError("El número de particiones (k-fold) debe ser cero o mayor.")
        
    def get_result(self) -> Result:
        return self._result
    
    def train(self):
        if self._data.set is None or self._data.target is None:
            raise ValueError("Los datos no han sido cargados. Por favor, cargue los datos antes de entrenar el modelo.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self._data.set,
            self._data.target,
            random_state=self._params.seed,
            test_size=self._params.test_size
        )        
        
        self._data = Data(
            self._data.set,
            self._data.target,
            X_train,
            y_train,
            X_test,
            y_test
        )
        
        self._knn.fit(self._data.train_features, self._data.train_target)

    def evaluate(self):
        if self._data.test_features is None or self._data.test_target is None:
            raise ValueError("Los datos de prueba no están disponibles. Por favor, asegúrese de que el modelo ha sido entrenado correctamente antes de evaluar.")
        
        predicted = self._knn.predict(self._data.test_features)
        expected = self._data.test_target
        errored = [(pred, exp) for (pred, exp) in zip(predicted, expected) if pred != exp]

        accuracy = self._knn.score(
            self._data.test_features, 
            self._data.test_target
        )

        crossval_accuracy = 0.0
        if self._params.k_fold > 0:
            kfold = KFold(n_splits=self._params.k_fold, shuffle=True, random_state=self._params.seed)
            scores = cross_val_score(self._knn, self._data.set, self._data.target, cv=kfold)
            crossval_accuracy = scores.mean()

        cm = confusion_matrix(self._data.test_target, predicted)

        self._result = Result(
            normal_accuracy=accuracy, 
            crossval_accuracy=crossval_accuracy,
            datetime=datetime.now(),
            feature_count=self._data.set.shape[1],
            total_records=len(self._data.set),
            train_records=len(self._data.train_features),
            test_records=len(self._data.test_features), 
            error_count=len(errored), 
            confusion_matrix=cm.tolist()
        )

    def predict(self, features: Any) -> int:
        if features is None:
            raise ValueError("No se han proporcionado características para predecir.")
        if not hasattr(self._knn, "classes_"):
            raise ValueError("El modelo no ha sido entrenado. Entrene el modelo antes de predecir.")

        arr = np.asarray(features)

        # Single 8x8 -> (1,64)
        if arr.ndim == 2 and arr.shape == (8, 8):
            arr = arr.reshape(1, 64)
        # Batch of 8x8 images -> (n,64)
        elif arr.ndim == 3 and arr.shape[1:] == (8, 8):
            arr = arr.reshape(arr.shape[0], 64)
        # Single flattened -> (1,64)
        elif arr.ndim == 1 and arr.size == 64:
            arr = arr.reshape(1, 64)
        # Batch flattened (n,64) -> OK
        elif arr.ndim == 2 and arr.shape[1] == 64:
            pass
        else:
            raise ValueError(f"Unsupported feature shape: {arr.shape}. Expected 8x8 or flattened 64-length vectors.")

        preds = self._knn.predict(arr)
        
        if preds.shape[0] != 1:
            raise ValueError("Expected 64-length 'pixels' list")
            
        return int(preds[0])
