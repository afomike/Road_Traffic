# PredictFlow — Road Traffic Management System

> Classify road traffic congestion risk in real time using machine learning — powered by four selectable models and a clean web interface.

[![Live App](https://img.shields.io/badge/Live%20App-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://road-traffic.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](./LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Live App](#live-app)
- [Features](#features)
- [Supported Models](#supported-models)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [Usage](#usage)
- [Input Reference](#input-reference)
- [Model Artifacts](#model-artifacts)
- [Deployment](#deployment)
- [License](#license)

---

## Overview

**PredictFlow** is a Flask web application for real-time road traffic condition classification. It accepts ten road-site features through a responsive browser interface, runs them through a user-selected machine learning model, and returns an immediate congestion risk classification.

The system supports four interchangeable classifiers — Naive Bayes, ANN (MLP), Decision Tree, and SVM — all loaded from pre-trained, serialized artifacts. Feature inputs are scaled at inference time using a fitted scaler, ensuring predictions are consistent with the distributions seen during training.

---

## Live App

**[https://road-traffic.onrender.com/](https://road-traffic.onrender.com/)**

> **Note:** The app is hosted on Render's free tier. The first request after a period of inactivity may take 30–60 seconds as the instance cold-starts.

---

## Features

| Feature | Description |
|---|---|
| **Multi-model Selection** | Choose from four trained classifiers per prediction request |
| **Real-time Classification** | Results returned in the browser immediately after form submission |
| **Feature Scaling** | Inputs are scaled automatically at inference using a pre-fitted scaler |
| **Responsive UI** | Clean form interface accessible across desktop and mobile |
| **Extensible Pipeline** | Swap in updated model artifacts without changing application code |

---

## Supported Models

| Model | File |
|---|---|
| Naive Bayes | `Naive Bayes_model.pkl` |
| ANN (MLP) | `ANN (MLP)_model.pkl` |
| Decision Tree | `Decision Tree_model.pkl` |
| SVM | `SVM_model.pkl` |

All four models are loaded at startup and kept in memory for low-latency inference. The user selects the desired model from the web interface before submitting a prediction request.

---

## Architecture

```
User Browser
     │
     ▼
Flask Application (app.py)
     │
     ├── GET  /    →  Render input form (index.html)
     └── POST /    →  Scale inputs → run selected model → return classification
           │
           ▼
     Inference Pipeline
     ├── scaler.pkl              (StandardScaler / MinMaxScaler)
     ├── Naive Bayes_model.pkl
     ├── ANN (MLP)_model.pkl
     ├── Decision Tree_model.pkl
     └── SVM_model.pkl
```

---

## Project Structure

```
predictflow/
│
├── app.py                      # Flask application, routes, and inference logic
├── requirements.txt            # Pinned Python dependencies
├── road_accident.ipynb         # Data analysis and model training notebook
│
├── model/
│   ├── scaler.pkl              # Fitted feature scaler
│   ├── Naive Bayes_model.pkl   # Trained Naive Bayes classifier
│   ├── ANN (MLP)_model.pkl     # Trained MLP neural network
│   ├── Decision Tree_model.pkl # Trained Decision Tree classifier
│   └── SVM_model.pkl           # Trained Support Vector Machine
│
├── templates/
│   └── index.html              # Web UI — input form and prediction display
│
├── static/
│   └── styles.css              # Application styling
│
├── README.md
└── LICENSE
```

---

## Prerequisites

- **Python 3.11 or higher**
- `pip` or a compatible package manager
- All model artifacts present in `model/` (see [Model Artifacts](#model-artifacts))

---

## Installation

**1. Clone the repository.**

```bash
git clone https://github.com/your-username/predictflow.git
cd predictflow
```

**2. Create and activate a virtual environment.**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies.**

```bash
pip install -r requirements.txt
```

**4. Verify all model artifacts are in place.**

```bash
ls model/
# Expected:
# scaler.pkl
# Naive Bayes_model.pkl
# ANN (MLP)_model.pkl
# Decision Tree_model.pkl
# SVM_model.pkl
```

---

## Running Locally

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## Usage

1. Open the application in your browser.
2. Fill in the ten road-site input fields (see [Input Reference](#input-reference) below).
3. Select the machine learning model you want to use for classification.
4. Click **Submit** to receive the traffic condition classification result.

---

## Input Reference

The application accepts the following ten road-site features:

| Field | Full Name | Description |
|---|---|---|
| `AADT` | Annual Average Daily Traffic | Total vehicle count averaged across the year |
| `LOCATION` | Location | Road site location descriptor |
| `COSITE` | Co-site Code | Site grouping or co-location identifier |
| `SECTION_` | Road Section | Identifier for the road section being evaluated |
| `X` | Longitude | Geographic longitude coordinate of the site |
| `Y` | Latitude | Geographic latitude coordinate of the site |
| `TFCTR` | Traffic Flow Factor | Adjustment factor for traffic flow volume |
| `FID` | Feature ID | Unique identifier for the road feature record |
| `DFCTR` | Daily Flow Factor | Daily variation adjustment factor |
| `KFCTR` | K-Factor | Peak-hour to AADT ratio factor |

---

## Model Artifacts

All model files are stored in the `model/` directory and loaded at application startup:

| File | Purpose |
|---|---|
| `scaler.pkl` | Scales raw input features before inference |
| `Naive Bayes_model.pkl` | Probabilistic classifier |
| `ANN (MLP)_model.pkl` | Multi-layer perceptron neural network |
| `Decision Tree_model.pkl` | Rule-based tree classifier |
| `SVM_model.pkl` | Support Vector Machine classifier |

**If you retrain any model**, ensure:
- The replacement `.pkl` file uses the same filename.
- The scaler is retrained on the same feature set and saved as `scaler.pkl`.
- The new artifacts are compatible with the `scikit-learn` version pinned in `requirements.txt`.

---

## Deployment

For production, serve the application with **Gunicorn** (included in `requirements.txt`):

```bash
gunicorn --bind 0.0.0.0:8000 app:app
```

### Deploying to Render

1. Connect your GitHub repository to a new Render **Web Service**.
2. Set the **Build Command** to `pip install -r requirements.txt`.
3. Set the **Start Command** to `gunicorn app:app`.
4. Ensure the `model/` directory and all `.pkl` artifacts are committed to the repository.

---

## License

This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE) for the full terms.
