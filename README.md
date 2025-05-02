# Smart Energy Theft Detection System

A Python-based implementation of an unsupervised anomaly detection system for detecting potential energy theft in smart meter data.

## Overview

This system detects irregular electricity usage patterns (Non-Technical Losses or theft) using unsupervised anomaly detection techniques. Rather than classifying usage as "theft" or "normal" with labeled data, the system learns what "normal" consumption looks like and then flags deviations.

The architecture is modular and implements the full pipeline from data ingestion and preprocessing through model training and visualization.

## Features

- **Data Pipeline**: Handles data ingestion, cleaning, resampling, and normalization of smart meter data.
- **Unsupervised Anomaly Detection**: Implements two anomaly detection models:
  - **Isolation Forest**: A tree-based ensemble model that isolates anomalies.
  - **Autoencoder**: A neural network-based model that learns to reconstruct normal patterns.
- **Evaluation**: Comprehensive evaluation metrics including ROC-AUC, Precision-Recall curves, and F1-score when labels are available.
- **Visualization**: Graphical visualization of consumption patterns, anomaly scores, and detected anomalies.
- **Analysis Tools**: Pattern mining and feature importance analysis to understand energy theft behaviors.
- **Interactive Dashboard**: Optional interactive dashboard for exploring results (requires Dash and Plotly).

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- seaborn

Optional:
- dash
- plotly

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-energy-theft-detection.git
cd smart-energy-theft-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from smart_energy_theft_detection import SmartEnergyTheftDetection

# Initialize the system (default: Isolation Forest)
system = SmartEnergyTheftDetection(use_autoencoder=False)

# Load and process data
df = system.load_data('smart_meter_data.csv')
X, y = system.preprocess_data(df)  # y is optional if labels exist

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
system.train_model(X_train, y_train)

# Detect anomalies
anomaly_scores, anomalies, threshold = system.detect_anomalies(X_test)

# Evaluate and visualize
if y_test is not None:
    metrics = system.evaluate_model(anomaly_scores, y_test)
system.visualize_anomalies(test_df, anomaly_scores, anomalies)
```

### Command Line Usage

```bash
# Run with default settings (Isolation Forest)
python run_detection_system.py --data smart_meter_data.csv

# Run with Autoencoder model
python run_detection_system.py --data smart_meter_data.csv --model autoencoder

# Generate visualizations
python run_detection_system.py --data smart_meter_data.csv --visualize

# Launch interactive dashboard
python run_detection_system.py --data smart_meter_data.csv --dashboard

# Run analysis toolkit
python energy_theft_analysis.py --data smart_meter_data.csv --analysis full
```

## File Structure

- `smart_energy_theft_detection.py`: Main implementation of the detection system.
- `run_detection_system.py`: Command-line interface for running the system.
- `energy_theft_analysis.py`: Toolkit for analyzing energy theft patterns.
- `smart_energy_theft_notebook.ipynb`: Jupyter notebook with examples and visualizations.
- `requirements.txt`: Required Python packages.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Architecture based on current best practices in anomaly detection.
- Utilizes publicly available smart meter datasets for development and evaluation.