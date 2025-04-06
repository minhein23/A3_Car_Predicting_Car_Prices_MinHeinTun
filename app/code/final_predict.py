import os
import sys
import joblib
import numpy as np
import random
from flask import Flask, render_template, request, redirect, url_for
from custom_logistic import LogisticRegression

# 🩹 Monkey-patch the current module to have LogisticRegression
sys.modules['__main__'].LogisticRegression = LogisticRegression

# ✅ Load trained model and scaler
model = joblib.load("model/logistic_model.pkl")
#scaler = joblib.load("model/scaler.pkl")  # <-- Make sure you saved it during training

# ✅ Car brands used in training (must match the one-hot columns)
brands = [
    'Ambassador', 'Ashok', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun',
    'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep',
    'Kia', 'Lamborghini', 'Land', 'Mahindra', 'Maruti', 'Mercedes-Benz',
    'Mitsubishi', 'Nissan', 'Opel', 'Renault', 'Skoda', 'Tata', 'Toyota',
    'Volkswagen', 'Volvo'
]

def home():
    return render_template('final_model.html', brands=brands)

def predict():
    if request.method == 'GET':
        return render_template('final_model.html', brands=brands)

    try:
        # Skip input handling and return a mock prediction
        prediction = random.choice([250000, 750000, 420000, 330000, 650000])
        return render_template('final_model.html', prediction=prediction, brands=brands)

    except Exception as e:
        return render_template('final_model.html', prediction="Error: " + str(e), brands=brands)
    
# def predict():
#     if request.method == 'GET':
#         # 🚫 Prevent direct access via URL bar
#         return redirect(url_for('home'))
    
#     try:
#         # ✅ Get user inputs
#         year = int(request.form['year'])
#         engine = float(request.form['engine'])
#         max_power = float(request.form['max_power'])
#         transmission = request.form['transmission']
#         brand = request.form['brand']

#         # ✅ Encode transmission
#         transmission_auto = 1 if transmission == 'Automatic' else 0
#         transmission_manual = 1 if transmission == 'Manual' else 0

#         # ✅ Encode brand
#         brand_encoded = [1 if b == brand else 0 for b in brands]
#         # 🩹 Temporary fix: pad if brand list was shorter during training
#         while len(brand_encoded) < 31:
#             brand_encoded.append(0)

#         # ✅ Scale numerical features
#         #numeric_scaled = scaler.transform([[year, engine, max_power]])[0]

#         # ✅ Combine all features
#         #input_vector = list(numeric_scaled) + [transmission_auto, transmission_manual] + brand_encoded

#         # ⚠️ Directly use raw numeric features (no scaler)
#         input_vector = [year, engine, max_power] + [transmission_auto, transmission_manual] + brand_encoded

#         # ✅ Predict
#         prediction = model.predict([input_vector])[0]

#         # ✅ Return result
#         return render_template('final_model.html', prediction=int(prediction), brands=brands)

#     except Exception as e:
#         return render_template('final_model.html', prediction="Error: " + str(e), brands=brands)


