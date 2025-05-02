import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def load_test_data():
    """Load the test dataset."""
    X_test = pd.read_csv("E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/data/X_test.csv")
    y_test = pd.read_csv("E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/data/y_test.csv").squeeze()  # Convert to Series
    return X_test, y_test

def load_model(model_path):
    """Load the trained model."""
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using precision, recall, F1-score, and ROC-AUC."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

if __name__ == "__main__":
    # Load the test data
    X_test, y_test = load_test_data()

    # Load the trained model
    model_path = "E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/models/random_forest_model.pkl"
    model = load_model(model_path)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)