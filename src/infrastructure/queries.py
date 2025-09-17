
from dataclasses import dataclass

@dataclass(frozen=True)
class QueryResult:
    success: bool
    message: str
    data: dict[str, any] = None

class Query():
    _name: str 

    def __init__(self, name:str):
        if name is None or name == "":
            raise ValueError("Name cannot be null or empty")
        self._name = name

    def execute(self) -> QueryResult:
        pass