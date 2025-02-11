from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import openai
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    column_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
    features = pd.DataFrame([list(data.values())], columns=column_names)
    prediction = heart_model.predict(features)
    result = "disease" if prediction[0] == 1 else "no-disease"
    return jsonify({"prediction": result})

# Route for diabetes prediction
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data = request.get_json(force=True)
    column_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    features = pd.DataFrame([list(data.values())], columns=column_names)
    prediction = diabetes_model.predict(features)
    result = "disease" if prediction[0] == 1 else "no-disease"
    return jsonify({"prediction": result})

# Route for liver disease prediction
@app.route('/predict/liver', methods=['POST'])
def predict_liver_disease():
    data = request.get_json(force=True)
    column_names=['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']
    features = pd.DataFrame([list(data.values())], columns=column_names)
    prediction = liver_model.predict(features)
    result = "disease" if prediction[0] == 1 else "no-disease"
    return jsonify({"prediction": result})

# Chatbot route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get("message")
    
    if not user_message:
        return jsonify({"reply": "Please ask a health-related question."})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical assistant. Provide health-related advice."},
                {"role": "user", "content": user_message}
            ]
        )
        bot_reply = response["choices"][0]["message"]["content"]
    except Exception as e:
        bot_reply = "Sorry, I am unable to process your request at the moment."

    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
