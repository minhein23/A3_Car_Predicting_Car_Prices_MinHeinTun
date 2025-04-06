import numpy as np
import joblib
import sys
from app.code.custom_logistic import LogisticRegression

# 🩹 Monkey-patch the class so joblib can find it during unpickling
sys.modules['__main__'].LogisticRegression = LogisticRegression

def load_local_model():
    model_path = "app/code/model/logistic_model.pkl"
    return joblib.load(model_path)

def test_input_compatibility():
    model = load_local_model()
    test_input = np.array([[2015, 80, 1500, 1, 0, 0]])
    output = model.predict(test_input)
    assert output is not None

def test_output_structure():
    model = load_local_model()
    test_input = np.array([[2015, 80, 1500, 1, 0, 0]])
    output = model.predict(test_input)
    assert output.shape[0] == 1
