import unittest
import numpy as np
import joblib

# ✅ Load the model (adjust the path if needed)
model = joblib.load("app/code/model/logistic_model.pkl")

class TestCarPriceModel(unittest.TestCase):
    def test_input_shape(self):
        """
        Test if the model accepts input of shape (1, 36)
        """
        sample_input = np.random.rand(1, 36)
        try:
            prediction = model.predict(sample_input)
            passed = True
        except Exception as e:
            passed = False
        self.assertTrue(passed, "❌ Model should accept input of shape (1, 36)")

    def test_output_shape(self):
        """
        Test if the model outputs prediction of shape (1,)
        """
        sample_input = np.random.rand(1, 36)
        prediction = model.predict(sample_input)
        self.assertEqual(prediction.shape, (1,), "❌ Prediction output shape should be (1,)")

if __name__ == '__main__':
    unittest.main()
