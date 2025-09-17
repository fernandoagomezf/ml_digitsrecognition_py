import numpy as np
from flask import render_template
from flask import request 
from flask import jsonify
from infrastructure.repositories import KNNRepository
from application.web import WebApp 
from application.commands import PredictCommand
from application.commands import TrainCommand
from application.commands import CommandInput

webapp = WebApp(__name__)

@webapp.get_engine().route("/")
@webapp.get_engine().route("/index")
def index():
    return render_template("index.html")

@webapp.get_engine().route("/about")
def about():
    return render_template("about.html")

@webapp.get_engine().route("/api/train", methods=["POST"])
def train():
    command = TrainCommand(KNNRepository())
    input = CommandInput(data=request.get_json(force=True))
    result = command.execute(input)
    return jsonify(success=result.success, message=result.message, data=result.data)

@webapp.get_engine().route("/api/predict", methods=["POST"])
def predict():    
    input = CommandInput(data=request.get_json(force=True))
    command = PredictCommand(KNNRepository())
    result = command.execute(input)
    return jsonify(success=result.success, message=result.message, data=result.data)

webapp.start()