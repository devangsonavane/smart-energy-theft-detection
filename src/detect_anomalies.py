import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

def load_model(model_path):
    """Load the trained model."""
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

def preprocess_input(data, scaler):
    """Preprocess a single data point or batch of data points."""
    # Convert to DataFrame for consistency
    if isinstance(data, list):
        # Ensure the correct order of features (use integers as feature names)
        data = pd.DataFrame([data], columns=[0, 1, 2, 3, 4])  # Use integer columns
    elif isinstance(data, dict):
        raise ValueError("Real-time input must be a list, not a dictionary.")

    # Normalize features
    data_scaled = scaler.transform(data)
    return data_scaled

def real_time_detection(model, scaler):
    """Simulate real-time anomaly detection."""
    print("\nReal-Time Anomaly Detection Started...\n")
    print("Provide new data points as a list of values in the format:")
    print("[Electricity_Consumed, Temperature, Humidity, Wind_Speed, Avg_Past_Consumption]")
    print("Type 'exit' to end the simulation.\n")

    while True:
        user_input = input("Enter data point: ")
        if user_input.lower() == 'exit':
            print("Exiting Real-Time Anomaly Detection...")
            break

        try:
            # Parse input
            data_point = eval(user_input)
            if not isinstance(data_point, list) or len(data_point) != 5:
                raise ValueError("Invalid format. Please provide 5 numeric values.")

            # Preprocess input
            data_scaled = preprocess_input(data_point, scaler)

            # Predict anomaly
            prediction = model.predict(data_scaled)
            prediction_proba = model.predict_proba(data_scaled)[:, 1]

            # Display result
            result = "Abnormal" if prediction[0] == 1 else "Normal"
            print(f"Prediction: {result} (Anomaly Probability: {prediction_proba[0]:.2f})\n")
        except Exception as e:
            print(f"Error: {e}. Please try again.\n")

if __name__ == "__main__":
    # Load the trained model and scaler
    model_path = "E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/models/random_forest_model.pkl"
    model = load_model(model_path)

    # Load the scaler (re-create it since we normalized in preprocessing)
    scaler = MinMaxScaler()
    scaler.fit(pd.read_csv("E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/data/X_train.csv", header=None))  # No column names used

    # Start real-time anomaly detection
    real_time_detection(model, scaler)