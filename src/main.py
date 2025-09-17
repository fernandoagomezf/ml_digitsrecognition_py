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
    data = request.get_json(force=True)
    if not data:
        return jsonify(success=False, error="No JSON body"), 400
    pixels = data.get("pixels")
    if pixels is None:
        return jsonify(success=False, error="No pixels provided"), 400    
    if not isinstance(pixels, list) or len(pixels) != 64:
        return jsonify(success=False, error="Expected 64-length 'pixels' list"), 400    

    input = CommandInput(data={
        "features": np.array(pixels, dtype=float).reshape(8, 8)
    })
    command = PredictCommand(KNNRepository())
    result = command.execute(input)
    return jsonify(result)

webapp.start()