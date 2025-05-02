# ⚡ Smart Energy Theft Detection using Machine Learning

This project leverages machine learning to detect electricity theft using smart meter data. It involves preprocessing datasets, training a Random Forest model, evaluating its performance, and detecting anomalies. A simple Streamlit UI allows users to interact with the system for theft prediction.

---

## 📁 Project Structure


SMART-ENERGY-THEFT-DETECTION/
│
├── data/                      # Contains input CSV datasets
│   ├── smart\_meter\_data.csv
│   ├── X\_train.csv, y\_train.csv
│   ├── X\_test.csv, y\_test.csv
│
├── models/                   # Trained model file
│   └── random\_forest\_model.pkl
│
├── notebooks/                # Jupyter Notebook for EDA
│   └── eda.ipynb
│
├── src/                      # Source code files
│   ├── detect\_anomalies.py       # Anomaly detection logic
│   ├── evaluate\_model.py         # Model performance metrics
│   ├── preprocess.py             # Data cleaning & feature engineering
│   ├── train\_model.py            # Model training script
│   ├── tempCodeRunnerFile.py     # (Ignore: VSCode temp file)
│
├── steamlit.py               # Streamlit UI for user interaction
│
├── Architecture.md           # System architecture overview
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies


---

## ⚙️ How It Works

1. **Data Preprocessing** (`preprocess.py`)
   - Cleans the raw smart meter data
   - Handles missing values and feature scaling
   - Splits data into train/test sets

2. **Model Training** (`train_model.py`)
   - Trains a `RandomForestClassifier` on labeled data
   - Saves the trained model to `models/random_forest_model.pkl`

3. **Model Evaluation** (`evaluate_model.py`)
   - Calculates accuracy, precision, recall, F1-score
   - Confusion matrix and other classification metrics

4. **Anomaly Detection** (`detect_anomalies.py`)
   - Uses trained model to detect energy theft
   - Returns whether a user is “Normal” or “Suspected Theft”

5. **Visualization and Insights** (`notebooks/eda.ipynb`)
   - Exploratory data analysis with graphs and plots

6. **Web Interface** (`steamlit.py`)
   - Upload data and get real-time predictions
   - Easy interface to test the system interactively

---

## 🚀 Getting Started

### 1. Clone the Repository

git clone https://github.com/yourusername/smart-energy-theft-detection.git
cd smart-energy-theft-detection

### 2. Set Up a Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

### 3. Install Requirements

pip install -r requirements.txt

### 4. Run Streamlit App

streamlit run steamlit.py

---

## 📊 Sample Output (Streamlit UI)

* Upload smart meter data (CSV)
* System displays prediction (Normal / Theft)
* Confidence score for each sample
* Visual insights (if included in the Streamlit UI)

---

## 📌 Dependencies

Main packages used:

* `pandas`, `numpy`
* `scikit-learn`
* `matplotlib`, `seaborn`
* `streamlit`
* `joblib`

---

## 📈 Model Used

* **Random Forest Classifier**

  * Handles high-dimensional data
  * Robust to outliers
  * Provides feature importance

---

## 📎 Future Improvements

* Add deep learning-based anomaly detection
* Integrate real-time data ingestion
* Store flagged users in a database
* Add admin dashboard with alerts

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

Devang Sonavane
Feel free to connect!
