from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('iris_model.pkl', 'rb'))

@app.route('/')
def home():
    return "ML Model is Live! Send a POST request to /predict"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Expecting input like [5.1, 3.5, 1.4, 0.2]
    prediction = model.predict([np.array(data['input'])])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)