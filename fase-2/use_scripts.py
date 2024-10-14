train_data_file = 'train.csv'
test_input_file = 'test.csv'
test_preds_file = 'test_predictions.csv'
model_file = 'model.keras'

import os

# Train a model and save it
os.system(f"python3 scripts/train.py --model_file {model_file} --data_file {train_data_file}  --overwrite_model")

# Make predictions with a saved model
os.system(f"python3 scripts/predict.py --model_file {model_file} --input_file {test_input_file}  --predictions_file {test_preds_file}")