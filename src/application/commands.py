
from dataclasses import dataclass


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
        pass