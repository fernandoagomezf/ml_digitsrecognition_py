import os
from flask import Flask
from application.commands import Command
from application.commands import CommandInput
from application.commands import CommandResult
from infrastructure.queries import Query

class Config():
    _basedir = os.path.abspath(os.path.dirname(__file__))
    _parentdir = os.path.abspath(os.path.join(_basedir, os.pardir))
    DEBUG = True

class WebApp():
    _engine: Flask
    _commands: dict[str, Command]
    _queries: dict[str, Query]
    
    def __init__(self, name:str):
        self._engine = Flask(name)
        self._engine.config.from_object(Config)
        self._commands = {}

    def get_engine(self) -> Flask:
        return self._engine    
    
    def start(self):
        self._engine.run(debug=Config.DEBUG)

    def register_command(self, command:Command) -> None:
        if command is None:
            raise ValueError("Command cannot be null")
        self._commands[command.get_name()] = command

    def register_query(self, query:Query) -> None:
        if query is None:
            raise ValueError("Query cannot be null")
        self._queries[query._name] = query

    def command(self, command_name: str, data: dict[str, any] = {}) -> any:
        cmd = self._commands.get(command_name)
        if cmd is None:
            raise ValueError(f"Command '{command_name}' not found")
        input = CommandInput(data=data)
        result = cmd.execute(input)
        
        return result