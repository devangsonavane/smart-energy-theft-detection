import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """Preprocess the dataset."""
    # Drop missing values
    data = data.dropna()

    # Convert timestamp to datetime
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])

    # Encode labels (Normal -> 0, Abnormal -> 1)
    data['Anomaly_Label'] = data['Anomaly_Label'].apply(lambda x: 1 if x == 'Abnormal' else 0)

    # Extract features and labels
    X = data[['Electricity_Consumed', 'Temperature', 'Humidity', 'Wind_Speed', 'Avg_Past_Consumption']]
    y = data['Anomaly_Label']

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y):
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load the dataset
    filepath = "E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/data/smart_meter_data.csv"
    data = load_data(filepath)

    # Preprocess the data
    X, y = preprocess_data(data)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save preprocessed data for further use
    pd.DataFrame(X_train).to_csv("E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/data/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/data/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/data/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("E:/Programs/SEM 6/AI/1/smart-energy-theft-detection/data/y_test.csv", index=False)