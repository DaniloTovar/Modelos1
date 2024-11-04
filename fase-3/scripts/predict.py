import argparse
from loguru import logger
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', required=True, type=str, help='a csv file with input data (no targets)')
parser.add_argument('--predictions_file', required=True, type=str, help='a csv file where predictions will be saved to')
parser.add_argument('--model_file', required=True, type=str, help='a .keras file with a model already stored (see train.py)')

args = parser.parse_args()

model_file       = args.model_file
input_file        = args.input_file
predictions_file = args.predictions_file

if not os.path.isfile(model_file):
    logger.error(f"model file {model_file} does not exist")
    exit(-1)

if not os.path.isfile(input_file):
    logger.error(f"input file {input_file} does not exist")
    exit(-1)



logger.info("loading input data")
Xts = pd.read_csv(input_file)
Xts = Xts.iloc[:,1:]
Xts = Xts/255.0
Xts = Xts.values.reshape(1,28,28,1)

logger.info("loading model")
m = load_model(model_file)

logger.info("making predictions")
preds = m.predict(Xts)

logger.info(f"saving predictions to {predictions_file}")
pd.DataFrame(preds.reshape(-1, 1), columns=['preds']).to_csv(predictions_file, index=False)