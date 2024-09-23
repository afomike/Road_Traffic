
# import pickle
# import sys

# print(f"Pickle version: {pickle.format_version}")
# print(f"Python version: {sys.version}")

import os
import pickle
from flask import Flask, request, render_template, jsonify
import numpy as np


app = Flask(__name__)

# Load the saved scaler and models
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load each model
with open('model/Naive Bayes_model.pkl', 'rb') as f:
    naive_bayes_model = pickle.load(f)

with open('model/ANN (MLP)_model.pkl', 'rb') as f:
    ann_model = pickle.load(f)

with open('model/Decision Tree_model.pkl', 'rb') as f:
    decision_tree_model = pickle.load(f)

with open('model/SVM_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Dictionary to store models for easy access
models = {
    "Naive Bayes": naive_bayes_model,
    "ANN (MLP)": ann_model,
    "Decision Tree": decision_tree_model,
    "SVM": svm_model
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    data = request.form
    feature_values = [float(data[feature]) for feature in data if feature != 'model']

    # Reshape input and scale it
    input_data = np.array([feature_values])
    input_data_scaled = scaler.transform(input_data)

    # Model selection based on user input
    selected_model = data['model']
    model = models[selected_model]

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Return the prediction to the webpage
    return render_template('index.html', prediction_text=f'Model: {selected_model}, Predicted Class: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
