import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="Smart Energy Theft Detection System",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 0px;
    }
    .card {
        border-radius: 5px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .normal-tag {
        background-color: #66BB6A;
        color: white;
        padding: 4px 8px;
        border-radius: 3px;
        font-weight: bold;
    }
    .abnormal-tag {
        background-color: #EF5350;
        color: white;
        padding: 4px 8px;
        border-radius: 3px;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        border: none;
    }
    hr {
        margin-top: 30px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_model(model_path):
    """Load the trained model."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler(scaler_data_path):
    """Load and prepare the scaler."""
    try:
        scaler = MinMaxScaler()
        scaler.fit(pd.read_csv(scaler_data_path, header=None))
        return scaler
    except Exception as e:
        st.error(f"Error preparing scaler: {e}")
        return None

def preprocess_input(data, scaler):
    """Preprocess a single data point."""
    # Convert to DataFrame with correct column names
    data_df = pd.DataFrame([data], columns=[0, 1, 2, 3, 4])
    # Normalize features
    data_scaled = scaler.transform(data_df)
    return data_scaled

def detect_anomaly(data, model, scaler):
    """Predict if the data point represents an anomaly."""
    # Preprocess input
    data_scaled = preprocess_input(data, scaler)
    
    # Make prediction
    prediction = model.predict(data_scaled)
    prediction_proba = model.predict_proba(data_scaled)[:, 1]
    
    result = {
        "prediction": "Abnormal" if prediction[0] == 1 else "Normal",
        "probability": float(prediction_proba[0]),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return result

def load_and_preprocess_original_data(filepath, limit=1000):
    """
    Load the original dataset and prepare it for visualization.
    
    Args:
        filepath: Path to the original CSV file
        limit: Maximum number of rows to load (for performance)
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        # Load the dataset
        data = pd.read_csv(filepath)
        
        # Limit the number of rows for performance
        if len(data) > limit:
            data = data.sample(limit, random_state=42)
            
        # Basic preprocessing similar to your preprocess.py
        data = data.dropna()
        
        # If the Timestamp column exists, convert it to datetime
        if 'Timestamp' in data.columns:
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            
        # If the Anomaly_Label column exists, encode it
        if 'Anomaly_Label' in data.columns:
            data['Anomaly_Label'] = data['Anomaly_Label'].apply(lambda x: 1 if x == 'Abnormal' else 0)
            
        return data
    except Exception as e:
        st.error(f"Error loading original dataset: {e}")
        return None

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'history' not in st.session_state:
        st.session_state.history = []

# Initialize session state
initialize_session_state()

# App header
st.markdown('<p class="main-header">Smart Energy Theft Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time monitoring and anomaly detection</p>', unsafe_allow_html=True)

# Load model and scaler - Set paths according to your environment
MODEL_PATH = os.environ.get('MODEL_PATH', './models/random_forest_model.pkl')
SCALER_DATA_PATH = os.environ.get('SCALER_DATA_PATH', './data/X_train.csv')

model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_DATA_PATH)

# Main layout with tabs
tab1, tab2, tab3 = st.tabs(["Dashboard", "History", "Dataset"])

with tab1:
    # Create a two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input Parameters")
        
        # Input form
        with st.form(key='input_form'):
            electricity = st.number_input("Electricity Consumed (kWh)", min_value=0.0, value=None, format="%.2f", placeholder="e.g. 42.5")
            temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=None, format="%.1f", placeholder="e.g. 22.8")
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=None, format="%.1f", placeholder="e.g. 65")
            wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, value=None, format="%.1f", placeholder="e.g. 5.2")
            avg_consumption = st.number_input("Avg Past Consumption (kWh)", min_value=0.0, value=None, format="%.2f", placeholder="e.g. 38.1")
            
            submit_button = st.form_submit_button(label='Detect Anomaly')
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Results card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Detection Results")
        
        if submit_button and model and scaler:
            if electricity is not None and temperature is not None and humidity is not None and wind_speed is not None and avg_consumption is not None:
                # Process inputs
                input_data = [electricity, temperature, humidity, wind_speed, avg_consumption]
                
                # Get prediction
                result = detect_anomaly(input_data, model, scaler)
                
                # Add to history
                history_entry = {
                    "timestamp": result["timestamp"],
                    "electricity": electricity,
                    "temperature": temperature,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "avg_consumption": avg_consumption,
                    "prediction": result["prediction"],
                    "probability": result["probability"]
                }
                st.session_state.history.insert(0, history_entry)
                
                # Display result
                if result["prediction"] == "Normal":
                    st.markdown(f'<p><span class="normal-tag">{result["prediction"]}</span></p>', unsafe_allow_html=True)
                    message = "The consumption pattern appears normal and within expected parameters."
                else:
                    st.markdown(f'<p><span class="abnormal-tag">{result["prediction"]}</span></p>', unsafe_allow_html=True)
                    message = "The consumption pattern shows anomalies that may indicate energy theft or system malfunction."
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result["probability"] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Anomaly Probability (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#EF5350" if result["prediction"] == "Abnormal" else "#66BB6A"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(message)
            else:
                st.warning("Please fill in all fields.")
        else:
            st.info("Submit data to see detection results.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization section - FIXED VERSION
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Consumption Trends")
        
        if len(st.session_state.history) > 0:
            # Convert history to DataFrame and ensure numeric values
            chart_data = pd.DataFrame(st.session_state.history[:10][::-1])  # Get last 10 entries in correct order
            
            # Ensure numeric conversion
            for col in ['electricity', 'temperature', 'humidity', 'wind_speed', 'avg_consumption']:
                if col in chart_data.columns:
                    chart_data[col] = pd.to_numeric(chart_data[col], errors='coerce')
            
            # Convert probability to numeric and ensure it's between 0-1
            if 'probability' in chart_data.columns:
                chart_data['probability'] = pd.to_numeric(chart_data['probability'], errors='coerce')
            
            # Debug info
            st.write(f"Data points: {len(chart_data)}")
            
            # Use Plotly for a more reliable chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add traces
            fig.add_trace(
                go.Scatter(
                    x=chart_data['timestamp'], 
                    y=chart_data['electricity'],
                    name="Electricity (kWh)",
                    line=dict(color="#1E88E5", width=3)
                ),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(
                    x=chart_data['timestamp'], 
                    y=chart_data['avg_consumption'],
                    name="Avg. Consumption (kWh)",
                    line=dict(color="#8B5CF6", width=3, dash='dash')
                ),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(
                    x=chart_data['timestamp'], 
                    y=chart_data['probability'],
                    name="Anomaly Probability",
                    line=dict(color="#EF5350", width=3)
                ),
                secondary_y=True,
            )
            
            # Set titles
            fig.update_layout(
                title_text="Energy Consumption and Anomaly Detection Trends",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Energy (kWh)", secondary_y=False)
            fig.update_yaxes(title_text="Anomaly Probability", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a fallback simple chart in case the above doesn't work
            if st.checkbox("Show alternative visualization"):
                st.line_chart(
                    chart_data,
                    x='timestamp',
                    y=['electricity', 'avg_consumption']
                )
        else:
            st.info("No data points submitted yet. Detection history will appear here.")
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Detection History")
    
    if len(st.session_state.history) > 0:
        # Convert history to DataFrame for display
        history_df = pd.DataFrame(st.session_state.history)
        
        # Format the probability column
        history_df['probability'] = history_df['probability'].apply(lambda x: f"{x*100:.1f}%")
        
        # Apply color formatting to prediction column
        def highlight_prediction(row):
            prediction = row['prediction']
            if prediction == 'Abnormal':
                return ['background-color: #FFEBEE' if col == 'prediction' else '' for col in row.index]
            elif prediction == 'Normal':
                return ['background-color: #E8F5E9' if col == 'prediction' else '' for col in row.index]
            else:
                return ['' for _ in row.index]
        
        # Display the styled DataFrame
        st.dataframe(
            history_df.style.apply(highlight_prediction, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        # Add export option
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Export to CSV",
            data=csv,
            file_name="energy_theft_detection_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No detection history available.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Additional section for analytics if needed
with tab1:
    if len(st.session_state.history) > 1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Analysis & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare pie chart data
            history_df = pd.DataFrame(st.session_state.history)
            prediction_counts = history_df['prediction'].value_counts().reset_index()
            prediction_counts.columns = ['Prediction', 'Count']
            
            fig = px.pie(
                prediction_counts, 
                values='Count', 
                names='Prediction',
                color='Prediction',
                color_discrete_map={'Normal': '#66BB6A', 'Abnormal': '#EF5350'},
                title='Prediction Distribution'
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot of electricity vs avg consumption
            fig = px.scatter(
                history_df, 
                x='electricity', 
                y='avg_consumption',
                color='prediction',
                color_discrete_map={'Normal': '#66BB6A', 'Abnormal': '#EF5350'},
                size=[0.5] * len(history_df),  # Adjust point size
                hover_data=['timestamp', 'probability'],
                title='Electricity Consumed vs Average Consumption'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Dataset visualization tab
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Original Dataset Visualization")
    
    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload original dataset (CSV)", type='csv')
    original_data_path = os.environ.get('ORIGINAL_DATA_PATH', './data/smart_meter_data.csv')
    
    # Radio button to choose data source
    data_source = st.radio(
        "Select data source",
        ["Use default path", "Upload file"],
        horizontal=True
    )
    
    filepath = original_data_path if data_source == "Use default path" else uploaded_file
    
    if filepath:
        # Load the dataset
        with st.spinner("Loading dataset..."):
            data = load_and_preprocess_original_data(filepath)
            
        if data is not None:
            # Display dataset info
            st.write(f"Dataset shape: {data.shape}")
            
            # Create tabs for different visualizations
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Data Preview", "Distribution", "Correlation", "Time Series"])
            
            with viz_tab1:
                st.subheader("Data Preview")
                st.dataframe(data.head(100), use_container_width=True)
                
                # Display column statistics
                st.subheader("Column Statistics")
                st.dataframe(data.describe().T, use_container_width=True)
            
            with viz_tab2:
                st.subheader("Feature Distributions")
                
                # Select a feature to visualize
                features = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                selected_feature = st.selectbox("Select feature to visualize", features)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = px.histogram(
                        data, 
                        x=selected_feature, 
                        color='Anomaly_Label' if 'Anomaly_Label' in data.columns else None,
                        color_discrete_map={0: '#66BB6A', 1: '#EF5350'},
                        marginal="box",
                        title=f"Distribution of {selected_feature}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig = px.box(
                        data, 
                        y=selected_feature, 
                        color='Anomaly_Label' if 'Anomaly_Label' in data.columns else None,
                        color_discrete_map={0: '#66BB6A', 1: '#EF5350'},
                        title=f"Box Plot of {selected_feature}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab3:
                st.subheader("Feature Correlation")
                
                # Calculate correlation matrix
                corr_features = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                corr_matrix = data[corr_features].corr()
                
                # Plot heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot for selected features
                st.subheader("Feature Relationships")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox("X-axis feature", corr_features, index=0)
                with col2:
                    y_feature = st.selectbox("Y-axis feature", corr_features, index=min(1, len(corr_features)-1))
                
                fig = px.scatter(
                    data,
                    x=x_feature,
                    y=y_feature,
                    color='Anomaly_Label' if 'Anomaly_Label' in data.columns else None,
                    color_discrete_map={0: '#66BB6A', 1: '#EF5350'},
                    opacity=0.7,
                    title=f"{x_feature} vs {y_feature}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab4:
                if 'Timestamp' in data.columns:
                    st.subheader("Time Series Analysis")
                    
                    # Aggregate data by time periods
                    time_features = [col for col in corr_features if col != 'Anomaly_Label']
                    selected_time_feature = st.selectbox("Select feature", time_features)
                    
                    # Group by time period
                    time_period = st.selectbox(
                        "Group by time period",
                        ["Day", "Week", "Month"]
                    )
                    
                    if time_period == "Day":
                        data['period'] = data['Timestamp'].dt.date
                    elif time_period == "Week":
                        data['period'] = data['Timestamp'].dt.isocalendar().week
                    else:
                        data['period'] = data['Timestamp'].dt.month
                    
                    # Group data
                    grouped_data = data.groupby(['period', 'Anomaly_Label' if 'Anomaly_Label' in data.columns else None])[selected_time_feature].mean().reset_index()
                    
                    # Plot time series
                    fig = px.line(
                        grouped_data,
                        x='period',
                        y=selected_time_feature,
                        color='Anomaly_Label' if 'Anomaly_Label' in data.columns else None,
                        color_discrete_map={0: '#66BB6A', 1: '#EF5350'},
                        markers=True,
                        title=f"{selected_time_feature} Over Time by {time_period}"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Anomaly counts over time
                    if 'Anomaly_Label' in data.columns:
                        anomaly_counts = data.groupby('period')['Anomaly_Label'].mean().reset_index()
                        anomaly_counts['Anomaly_Rate'] = anomaly_counts['Anomaly_Label'] * 100
                        
                        fig = px.bar(
                            anomaly_counts,
                            x='period',
                            y='Anomaly_Rate',
                            title=f"Anomaly Rate (%) by {time_period}",
                            color_discrete_sequence=['#EF5350']
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No timestamp column found in the dataset for time series analysis.")
        else:
            st.error("Failed to load dataset. Please check the file path or try uploading a different file.")
    else:
        st.info("Please upload your dataset or use the default path to visualize the original data.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #f5f5f5; border-radius: 5px;">
    <p style="color: #666; font-size: 0.8rem;">
        Smart Energy Theft Detection System © 2025
    </p>
</div>
""", unsafe_allow_html=True)