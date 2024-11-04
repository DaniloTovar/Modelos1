import requests

input_file = 'drawing.jpg'

endpoint = 'http://127.0.0.1:5000/train'
response = requests.get(endpoint)
print("train: ", response.json())

endpoint = 'http://127.0.0.1:5000/predict'
args = {'input': input_file}
response = requests.get(endpoint, json=args)
print("prediction: ", response.json())
