import os
import logging
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and scaler
model_path = os.environ.get("MODEL_PATH", "best_model.pkl")
scaler_path = os.environ.get("SCALER_PATH", "scaler.pkl")

try:
    best_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logging.info(f"Model loaded from: {model_path}")
    logging.info(f"Scaler loaded from: {scaler_path}")
except Exception as e:
    logging.error(f"Error loading model/scaler: {e}")
    raise RuntimeError("Failed to load model or scaler.")

# Define input schema using Pydantic
class PredictionInput(BaseModel):
    Day_of_Week: int
    Month: int
    Holiday_Indicator: int
    Advertising_Spend: int
    Discount: int
    Stock_Levels: int
    Economic_Indicator: float
    Product_Category_Furniture: int
    Product_Category_Groceries: int
    Region_North: int
    Region_South: int
    Weather_Condition_Rainy: int
    Weather_Condition_Sunny: int


# Define the /predict endpoint
@app.post("/predict")
def predict_sales(input_data: PredictionInput):
    try:
        # Convert input data to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])

        # Scale the data
        input_scaled = scaler.transform(input_df)

        # Predict sales
        predicted_sales = best_model.predict(input_scaled)[0]

        # Return the prediction
        return {"predicted_sales": predicted_sales}

    except Exception as e:
        logging.exception("Error during prediction:")
        raise HTTPException(status_code=500, detail="Prediction error")


# Test endpoint
@app.get("/")
def root():
    return {"message": "FastAPI app is running!"}
