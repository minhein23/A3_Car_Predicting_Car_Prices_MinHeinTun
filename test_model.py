import numpy as np
import mlflow
import mlflow.pyfunc


def fetch_model():
    """
    Loads the MLflow model from the staging environment.
    """
    tracking_url = "https://mlflow.ml.brain.cs.ait.ac.th"
    mlflow.set_tracking_uri(tracking_url)
    model_path = "models:/st125367-a3-model/Staging"
    return mlflow.pyfunc.load_model(model_path)


def test_input_compatibility():
    """
    Ensures the model accepts the expected input format and returns a result.
    """
    classifier = fetch_model()
    test_features = np.array([[2015, 80, 1500, 1, 0, 0]])
    prediction = classifier.predict(test_features)
    assert prediction is not None, "No output returned from model prediction."


def test_output_structure():
    """
    Confirms the output shape matches expected dimensions (1 prediction).
    """
    classifier = fetch_model()
    test_features = np.array([[2015, 80, 1500, 1, 0, 0]])
    prediction = np.asarray(classifier.predict(test_features))
    assert prediction.shape[0] == 1, f"Expected output length of 1, got {prediction.shape[0]}"


if __name__ == "__main__":
    test_input_compatibility()
    test_output_structure()
    print("✅ All model shape tests passed.")
