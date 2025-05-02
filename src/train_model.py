import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def load_data():
    """Load preprocessed training and testing data."""
    X_train = pd.read_csv("E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/data/X_train.csv")
    X_test = pd.read_csv("E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/data/X_test.csv")
    y_train = pd.read_csv("E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/data/y_train.csv").squeeze()  # Convert to Series
    y_test = pd.read_csv("E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/data/y_test.csv").squeeze()    # Convert to Series
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """Train a Random Forest Classifier."""
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

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

def save_model(model, filepath):
    """Save the trained model to a file."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train model
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate model
    evaluate_model(rf_model, X_test, y_test)

    # Save model
    save_model(rf_model, "E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/models/random_forest_model.pkl")