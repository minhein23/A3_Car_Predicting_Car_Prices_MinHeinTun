import numpy as np
import joblib
import os

def load_local_model():
    model = joblib.load("model/logistic_model.pkl")
    return joblib.load(model)

def test_input_compatibility():
    model = load_local_model()
    test_features = np.array([[2015, 80, 1500, 1, 0, 0]])
    prediction = model.predict(test_features)
    assert prediction is not None

def test_output_structure():
    model = load_local_model()
    test_features = np.array([[2015, 80, 1500, 1, 0, 0]])
    result = np.asarray(model.predict(test_features))
    assert result.shape[0] == 1



if __name__ == "__main__":
    test_input_compatibility()
    test_output_structure()
    print("✅ All model shape tests passed.")
