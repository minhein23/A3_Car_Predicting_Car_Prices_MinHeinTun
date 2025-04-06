# Importing necessary libraries
from flask import Flask, render_template, request, flash  # Flask for web application, render_template for HTML rendering, request for handling form data, flash for messaging
import pickle  # For loading serialized model files
import numpy as np  # For numerical operations
import os
from predict import LinearRegression, NormalPenalty, LassoPenalty, RidgePenalty, ElasticPenalty, Normal, Lasso, Ridge, ElasticNet  # Custom linear regression models
from final_predict import model, brands
from final_predict import predict

# Initialize the Flask application
app = Flask(__name__)

# Set a secret key for Flask to use flash messages (for user notifications)
app.secret_key = 'mht23'


# ✅ Add the route
app.add_url_rule('/predict', view_func=predict, methods=['POST'])

# Importing the old model
# Construct the absolute path to the old model file
filename1 = os.path.join(os.path.dirname(__file__), 'model', 'car_price_old.model')

try:
    # Open and load the old model file
    with open(filename1, 'rb') as file:
        loaded_data1 = pickle.load(file)
    print("Model loaded successfully")
except FileNotFoundError:
    print(f"Error: File not found at {filename1}")
except pickle.UnpicklingError:
    print(f"Error: File is corrupted or not a valid pickle file at {filename1}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Separating the values in the old model file into variables for easy access
model_old = loaded_data1['model']  # Trained model
scaler_old = loaded_data1['scaler']  # Scaler used for preprocessing
name_map_old = loaded_data1['name_map']  # Mapping for brand names
engine_default_old = loaded_data1['engine_default']  # Default engine value
mileage_default_old = loaded_data1['mileage_default']  # Default mileage value

# Importing the new model
# Construct the absolute path to the new model file
filename2 = os.path.join(os.path.dirname(__file__), 'model', 'car_price_new.model')

try:
    # Open and load the new model file
    with open(filename2, 'rb') as file:
        loaded_data2 = pickle.load(file)
    print("New model loaded successfully")
except FileNotFoundError:
    print(f"Error: File not found at {filename2}")
except pickle.UnpicklingError:
    print(f"Error: File is corrupted or not a valid pickle file at {filename2}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Separating the values in the new model file into variables for easy access
model_new = loaded_data2['model']  # Trained model
scaler_new = loaded_data2['scaler']  # Scaler used for preprocessing
name_map_new = loaded_data2['name_map']  # Mapping for brand names
engine_default_new = loaded_data2['engine_default']  # Default engine value
mileage_default_new = loaded_data2['mileage_default']  # Default mileage value

# Route for the home page
@app.route('/')
def index():
    """
    Renders the home page of the web application.
    """
    return render_template('index.html')

# Route for the old model prediction page
@app.route('/old_model')
def predict_old():
    """
    Renders the prediction page for the old model.
    """
    return render_template('old_model.html')

# Route to process data for the old model (not directly accessed by users)
@app.route('/process-data_old', methods=['POST'])
def process_data_old():
    """
    Processes user input from the old model prediction form and returns the predicted car price.
    """
    if request.method == 'POST':
        # Get user input from the form
        brand_name = request.form.get('name')
        name = name_map_old.get(brand_name, '32')  # Map brand name to a numerical value
        engine = request.form.get('engine', engine_default_old)  # Get engine value or use default
        mileage = request.form.get('mileage', mileage_default_old)  # Get mileage value or use default

        # Convert engine and mileage to float (use defaults if empty)
        engine = float(engine) if engine else engine_default_old
        mileage = float(mileage) if mileage else mileage_default_old

        # Predict the car price using the old model
        result = str(int(prediction_old(name, engine, mileage)[0]))

        return result

# Function to predict car price using the old model
def prediction_old(name, engine, mileage):
    """
    Predicts the car price using the old model.
    
    Args:
        name (int): Numerical representation of the car brand.
        engine (float): Engine size of the car.
        mileage (float): Mileage of the car.
    
    Returns:
        float: Predicted car price.
    """
    # Prepare the input data as a numpy array
    sample = np.array([[name, engine, mileage]])

    # Scale the input data using the trained scaler
    sample_scaled = scaler_old.transform(sample)

    # Predict the car price using the trained model and apply exponential transformation
    result = np.exp(model_old.predict(sample_scaled))

    return result

# Route for the new model prediction page
@app.route('/new_model')
def predict_new():
    """
    Renders the prediction page for the new model and displays a flash message.
    """
    flash('New model, same vibe—trained fresh to predict your ride’s price better. 🚗💸✨', 'success')
    return render_template('new_model.html')

# Route to process data for the new model (not directly accessed by users)
@app.route('/process-data_new', methods=['POST'])
def process_data_new():
    """
    Processes user input from the new model prediction form and returns the predicted car price.
    """
    if request.method == 'POST':
        # Get user input from the form
        brand_name = request.form.get('name')
        name = name_map_new.get(brand_name, '32')  # Map brand name to a numerical value
        engine = request.form.get('engine', engine_default_new)  # Get engine value or use default
        mileage = request.form.get('mileage', mileage_default_new)  # Get mileage value or use default

        # Convert engine and mileage to float (use defaults if empty)
        engine = float(engine) if engine else engine_default_new
        mileage = float(mileage) if mileage else mileage_default_new

        # Predict the car price using the new model
        result = str(int(prediction_new(name, engine, mileage)[0]))

        return result

# Function to predict car price using the new model
def prediction_new(name, engine, mileage):
    """
    Predicts the car price using the new model.
    
    Args:
        name (int): Numerical representation of the car brand.
        engine (float): Engine size of the car.
        mileage (float): Mileage of the car.
    
    Returns:
        float: Predicted car price.
    """
    # Prepare the input data as a numpy array
    sample = np.array([[name, engine, mileage]])

    # Scale the input data using the trained scaler and add intercepts
    sample_scaled = scaler_new.transform(sample)
    intercept = np.ones((sample_scaled.shape[0], 1))  # Add a column of ones for the intercept term
    sample_scaled = np.concatenate((intercept, sample_scaled), axis=1)

    # Predict the car price using the trained model and apply exponential transformation
    result = np.exp(model_new.predict(sample_scaled))

    return result

@app.route('/final_model')
def predict_final():
    """
    Renders the prediction page for the final classification model.
    """
    return render_template('final_model.html', brands=brands)


@app.route('/process-data_final', methods=['POST'])
def process_data_final():
    """
    Processes user input for the final model and returns the predicted price class.
    """
    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            engine = float(request.form['engine'])
            max_power = float(request.form['max_power'])
            transmission = request.form['transmission']
            brand = request.form['brand']

            # Transmission encoding
            transmission_auto = 1 if transmission == 'Automatic' else 0
            transmission_manual = 1 if transmission == 'Manual' else 0

            # Brand encoding
            brand_encoded = [1 if b == brand else 0 for b in brands]

            # Scale numerical features
            #numeric_scaled = scaler.transform([[year, engine, max_power]])[0]

            # Raw numerical features (no scaling for now)
            numeric_scaled = [year, engine, max_power]

            # Full input vector
            input_vector = list(numeric_scaled) + [transmission_auto, transmission_manual] + brand_encoded

            prediction = model.predict([input_vector])[0]

            return render_template('final_model.html', prediction=int(prediction), brands=brands)

        except Exception as e:
            return render_template('final_model.html', prediction=f"Error: {e}", brands=brands)



if __name__ == '__main__':
    app.run(debug=True)


# Define the port number for the Flask application
port_number = 8000

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)