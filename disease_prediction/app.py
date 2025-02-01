from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained models
with open("models/heart_disease_model.pkl", "rb") as file:
    heart_model = pickle.load(file)

with open("models/diabetes_model.pkl", "rb") as file:
    diabetes_model = pickle.load(file)

with open("models/liver_model.pkl", "rb") as file:
    liver_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

# Route for heart disease prediction
@app.route('/predict/heart', methods=['POST'])
def predict_heart_disease():
    data = request.get_json(force=True)
    features = np.array([list(data.values())])
    prediction = heart_model.predict(features)
    result = "Disease" if prediction[0] == 1 else "No Disease"
    return jsonify({"prediction": result})

# Route for diabetes prediction
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data = request.get_json(force=True)
    features = np.array([list(data.values())])
    prediction = diabetes_model.predict(features)
    result = "Disease" if prediction[0] == 1 else "No Disease"
    return jsonify({"prediction": result})

# Route for liver disease prediction
@app.route('/predict/liver', methods=['POST'])
def predict_liver_disease():
    data = request.get_json(force=True)
    features = np.array([list(data.values())])
    prediction = liver_model.predict(features)
    result = "Disease" if prediction[0] == 1 else "No Disease"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
