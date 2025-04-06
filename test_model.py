import numpy as np
import joblib
import os
from app.code.custom_logistic import LogisticRegression

# ✅ Monkey patch required if custom model class is used in pickle
import sys
sys.modules['__main__'].LogisticRegression = LogisticRegression

def load_local_model():
    model_path = os.path.join("app", "code", "model", "logistic_model.pkl")
    return joblib.load(model_path)

def test_input_compatibility():
    model = load_local_model()

    # ✅ Create dummy input of correct length
    dummy_input = np.zeros((1, 36))  # The model expects 36 features
    dummy_input[0, 0:3] = [2015, 80, 1500]   # year, engine, power
    dummy_input[0, 3:5] = [1, 0]             # transmission: auto=1, manual=0
    dummy_input[0, 5] = 1                    # simulate one brand as 1

    prediction = model.predict(dummy_input)
    assert prediction is not None

def test_output_structure():
    model = load_local_model()

    dummy_input = np.zeros((1, 36))
    dummy_input[0, 0:3] = [2015, 80, 1500]
    dummy_input[0, 3:5] = [1, 0]
    dummy_input[0, 5] = 1

    output = model.predict(dummy_input)
    assert output.shape[0] == 1

if __name__ == "__main__":
    test_input_compatibility()
    test_output_structure()
    print("✅ All tests passed.")
