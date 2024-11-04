train_data_file = 'train.csv'
test_preds_file = 'predictions.csv'
model_file = 'model.keras'
client_input_file = 'input_data.csv'

from flask import Flask, jsonify, request
from loguru import logger
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image

app = Flask(__name__)

train_status = "not training"
prediction_status = "not predicting"

def _train():
    global train_status
    logger.info("Training started")
    train_status = "training"

    # Train a model and save it
    os.system(f"python3 scripts/train.py --model_file {model_file} --data_file {train_data_file}  --overwrite_model")

    logger.info("Training finished")
    train_status = "not training"

def _predict():
    global prediction_status
    logger.info("Predicting started")
    prediction_status = "predicting"

    # Make predictions with a saved model
    os.system(f"python3 scripts/predict.py --model_file {model_file} --input_file {client_input_file}  --predictions_file {test_preds_file}")

    logger.info("Predicting finished")
    prediction_status = "not predicting"

def _prepare_image(input_data):
    img_array = cv2.imread(input_data, cv2.IMREAD_GRAYSCALE)
    img_pil = Image.fromarray(img_array)
    img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
    img_array = (img_28x28.flatten())
    img_array  = img_array.reshape(-1,1).T
    pd.DataFrame(img_array).to_csv(client_input_file)

@app.route('/')
def index():
    return "Hello World!"

@app.route("/status")
def status():
    return jsonify("train_status", train_status,
                   "prediction_status", prediction_status)

@app.route("/predict")
def predict():
    input_data = request.json.get('input')
    _prepare_image(input_data)

    _predict()

    result = pd.read_csv(test_preds_file)
    result = np.array(result)
    choice = str(result.argmax())
    return jsonify({"prediction": choice})

@app.route("/train")
def train():
    if train_status == 'training':
       return  jsonify({"result": "already training"})

    _train()
    return jsonify({"result": "training finished"})

if __name__ == "__main__":
    app.run(debug=True)