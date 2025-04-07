# 🚗 A3 Car Price Prediction (MLflow + Flask + Docker)

Made by st125367 - Min Hein Tun

Please kindly see the website at CSIM ML server in the link below......
>>>>>>>
st125367.ml.brain.cs.ait.ac.th
<<<<<<<<<<<<
>>>>>>>

This project is part of the AIT Machine Learning course (Assignment 3). It demonstrates a full machine learning pipeline for car price **classification**, including:

- Data cleaning & preprocessing
- Model training and evaluation
- MLflow experiment tracking
- Dockerized Flask API for prediction
- CI/CD deployment to AIT ML Brain Server

---

## 📦 Features

- ✅ Logistic Regression classifier
- ✅ Custom training with batch & regularization support
- ✅ MLflow tracking & model registry
- ✅ REST API for predictions
- ✅ Dockerized & GitHub CI/CD
- ✅ Deployed with Traefik reverse proxy (HTTPS ready)

---

## 🧠 ML Model

- Input features:
  - Year
  - Max Power
  - Engine CC
  - Transmission (Encoded)
  - Fuel Type (Encoded)
  - Owner (Encoded)

- Output: Car price category (low / medium / high)

---

## 🚀 Deployment

### Docker (manual)

```bash
docker compose pull
docker compose up -d
