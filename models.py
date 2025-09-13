from typing import Any

from datetime import datetime
from dataclasses import dataclass
from dataclasses import replace
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
    accuracy: float
    datetime: datetime
    feature_count: int
    total_records: int
    train_records: int
    test_records: int

@dataclass(frozen=True)
class HyperParameters:
    seed:int
    test_size:float
    k_neighbors:int

class Model():
    _params: HyperParameters
    _data: Data
    _result: Result
    _knn: KNeighborsClassifier

    def __init__(self):
        self._params = HyperParameters(
            seed=0, 
            test_size=0.3, 
            k_neighbors=3
        )
        self._data = Data(None, None, None, None, None, None)
        self._knn = KNeighborsClassifier(n_neighbors=self._params.k_neighbors)
        self._result = None

    def load(self):
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
        
    def get_result(self) -> Result:
        return self._result

    def train(self):
        if self._data.set is None or self._data.target is None:
            raise ValueError("Los datos no han sido cargados. Por favor, cargue los datos antes de entrenar el modelo.")
        
        x_train, x_test, y_train, y_test = train_test_split(
            self._data.set,
            self._data.target,
            random_state=self._params.seed,
            test_size=self._params.test_size
        )
        self._data = Data(
            self._data.set,
            self._data.target,
            x_train,
            y_train,
            x_test,
            y_test
        )
        self._knn.fit(
            self._data.train_features, 
            self._data.train_target
        )

    def evaluate(self):
        if self._data.test_features is None or self._data.test_target is None:
            raise ValueError("Los datos de prueba no están disponibles. Por favor, asegúrese de que el modelo ha sido entrenado correctamente antes de evaluar.")
        
        accuracy = self._knn.score(
            self._data.test_features, 
            self._data.test_target
        )

        self._result = Result(
            accuracy=accuracy, 
            datetime=datetime.now(),
            feature_count=self._data.set.shape[1],
            total_records=len(self._data.set),
            train_records=len(self._data.train_features),
            test_records=len(self._data.test_features)
        )
