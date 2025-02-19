import os
import logging
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import jsonschema  # For input validation
from flask_cors import CORS

# Configure logging (good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load model and scaler (using environment variables - important for deployment)
model_path = os.environ.get("MODEL_PATH", "best_model.pkl")  # Default if env var not set
scaler_path = os.environ.get("SCALER_PATH", "scaler.pkl")

try:
    best_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logging.info(f"Model loaded from: {model_path}")
    logging.info(f"Scaler loaded from: {scaler_path}")
except Exception as e:
    logging.error(f"Error loading model/scaler: {e}")
    exit(1)  # Exit if loading fails

# Input validation schema (essential!)
schema = {
    "type": "object",
    "properties": {
        "Day of Week": {"type": "integer"},
        "Month": {"type": "integer"},
        "Holiday Indicator": {"type": "integer"},
        "Advertising Spend": {"type": "integer"},
        "Discount (%)": {"type": "integer"},
        "Stock Levels": {"type": "integer"},
        "Economic Indicator": {"type": "number"},  # Or "integer" if it's always an integer
        "Product Category_Furniture": {"type": "integer"},
        "Product Category_Groceries": {"type": "integer"},
        "Region_North": {"type": "integer"},
        "Region_South": {"type": "integer"},
        "Weather Condition_Rainy": {"type": "integer"},
        "Weather Condition_Sunny": {"type": "integer"},
    },
    "required": [  # List all required fields
        "Day of Week", "Month", "Holiday Indicator", "Advertising Spend",
        "Discount (%)", "Stock Levels", "Economic Indicator",
        "Product Category_Furniture", "Product Category_Groceries",
        "Region_North", "Region_South", "Weather Condition_Rainy",
        "Weather Condition_Sunny"
    ],
}


@app.route('/predict', methods=['POST'])
def predict_sales():
    data = request.get_json()

    # Input validation
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        logging.error(f"Invalid input: {e}")
        return jsonify({"error": "Invalid input", "details": e.message}), 400  # Bad Request

    try:
        day_data = pd.DataFrame([data])  # Data is already a dictionary
        day_data_scaled = scaler.transform(day_data)
        predicted_sales = best_model.predict(day_data_scaled)[0]
        return jsonify({"predicted_sales": predicted_sales}), 200  # 200 OK

    except Exception as e:  # Catch other potential errors
        logging.exception("Error during prediction:")  # Log the full traceback
        return jsonify({"error": "Prediction error"}), 500  # Internal Server Error


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Get port from environment variable
    app.run(debug=True, host='0.0.0.0', port=port)  # host='0.0.0.0' is important for Render