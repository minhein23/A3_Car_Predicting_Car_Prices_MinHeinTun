import numpy as np
import joblib
import os

def load_local_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "app", "model", "logistic_model.pkl")
    return joblib.load(model_path)

def test_input_compatibility():
    model = load_local_model()
    test_features = np.array([[2015, 80, 1500, 1, 0, 0]])
    prediction = model.predict(test_features)
    assert prediction is not None, "Model did not return any prediction"

def test_output_structure():
    model = load_local_model()
    test_features = np.array([[2015, 80, 1500, 1, 0, 0]])
    result = np.asarray(model.predict(test_features))
    assert result.shape[0] == 1, f"Expected output shape (1,), got {result.shape}"

if __name__ == "__main__":
    test_input_compatibility()
    test_output_structure()
    print("✅ All model shape tests passed.")