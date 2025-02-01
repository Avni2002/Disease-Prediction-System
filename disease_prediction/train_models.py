import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the heart disease dataset
df = pd.read_csv("dataset/heart.csv")

# Split features and target
X = df.drop(columns=["target"])  # Features
y = df["target"]  # Target variable (1 = Disease, 0 = No Disease)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
heart_model = RandomForestClassifier(n_estimators=100, random_state=42)
heart_model.fit(X_train, y_train)

# Evaluate model
heart_accuracy = heart_model.score(X_test, y_test)
print(f"Heart Disease Model Accuracy: {heart_accuracy * 100:.2f}%")

# Ensure the 'models' directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# Save the trained model
with open("models/heart_disease_model.pkl", "wb") as model_file:
    pickle.dump(heart_model, model_file)

print("Heart Disease Model Trained and Saved Successfully!")

# Load the diabetes dataset
df_diabetes = pd.read_csv("dataset/diabetes.csv")

# Split features and target
X_diabetes = df_diabetes.drop(columns=["Outcome"])  # Features
y_diabetes = df_diabetes["Outcome"]  # Target variable (1 = Disease, 0 = No Disease)

# Split into training and testing sets
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)

# Train the model
diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
diabetes_model.fit(X_train_d, y_train_d)

# Evaluate model
diabetes_accuracy = diabetes_model.score(X_test_d, y_test_d)
print(f"Diabetes Model Accuracy: {diabetes_accuracy * 100:.2f}%")

# Save the trained model
with open("models/diabetes_model.pkl", "wb") as model_file:
    pickle.dump(diabetes_model, model_file)

print("Diabetes Model Trained and Saved Successfully!")

# Load the liver disease dataset
df_liver = pd.read_csv("dataset/indian_liver_patient.csv")

# Handling missing values (if any)
df_liver = df_liver.dropna()

# Convert categorical features to numerical (e.g., Gender)
df_liver["Gender"] = df_liver["Gender"].map({"Male": 1, "Female": 0})

# Ensure correct column names (modify as per dataset)
X_liver = df_liver.drop(columns=["Dataset"])  # Features
y_liver = df_liver["Dataset"]  # Target variable (1 = Disease, 2 = No Disease)

# Convert target variable to binary (1 = Disease, 0 = No Disease)
y_liver = y_liver.map({1: 1, 2: 0})

# Split into training and testing sets
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_liver, y_liver, test_size=0.2, random_state=42)

# Train the model
liver_model = RandomForestClassifier(n_estimators=100, random_state=42)
liver_model.fit(X_train_l, y_train_l)

# Evaluate model
liver_accuracy = liver_model.score(X_test_l, y_test_l)
print(f"Liver Disease Model Accuracy: {liver_accuracy * 100:.2f}%")

# Save the trained model
with open("models/liver_model.pkl", "wb") as model_file:
    pickle.dump(liver_model, model_file)

print("Liver Disease Model Trained and Saved Successfully!")
