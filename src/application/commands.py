import numpy as np
from dataclasses import dataclass
from flask import jsonify
from infrastructure.repositories import KNNRepository
from domain.digits import KNNModel

@dataclass(frozen=True)
class CommandResult:
    success: bool
    message: str
    data: dict[str, any] = None

@dataclass(frozen=True)
class CommandInput:
    data: dict[str, any] = None

class Command():
    _name: str 
    
    def __init__(self, name:str):
        if name is None or name == "":
            raise ValueError("Name cannot be null or empty")
        self._name = name

    def get_name(self) -> str:
        return self._name
    
    def execute(self, input: CommandInput) -> CommandResult:
        if input is None:
            raise ValueError("Input cannot be null")

class TrainCommand(Command):
    _repository: KNNRepository

    def __init__(self, repository: KNNRepository):
        super().__init__("train_model")
        self._repository = repository

    def execute(self, input: CommandInput) -> CommandResult:
        super().execute(input)
        
        model = KNNModel()
        model.read_data()
        model.set_seed(input.data["seed"])
        model.set_k_neighbors(input.data["k_neighbors"])
        model.set_test_size(input.data["test_size"])
        model.set_k_fold(input.data["k_fold"])

        model.train()
        model.evaluate()
        result = model.get_result()

        self._repository.update(model)

        return CommandResult(success=True, message="Model trained successfully", data=result)
    
class PredictCommand(Command):
    _repository: KNNRepository

    def __init__(self, repository: KNNRepository):
        super().__init__("predict")
        self._repository = repository

    def execute(self, input: CommandInput) -> CommandResult:
        super().execute(input)

        if input.data is None:
            return CommandResult(success=False, message="No input data provided")
        pixels = input.data.get("pixels")
        if pixels is None:
            return CommandResult(success=False, message="No pixels provided for prediction")
        if not isinstance(pixels, list) or len(pixels) != 64:
            return CommandResult(success=False, message="Expected 64-length 'pixels' list")

        features = np.array(pixels, dtype=float).reshape(8, 8)
        if features is None or len(features) == 0:
            return CommandResult(success=False, message="No features provided for prediction")

        model = self._repository.get()

        prediction = model.predict(features)
        return CommandResult(success=True, message="Prediction successful", data={"prediction": prediction})