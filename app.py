from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json
import plotly
import plotly.express as px

app = Flask(__name__)

# Load the model
model_info = joblib.load('models/random_forest_model.pkl')
model = model_info['model']
feature_names = model_info['feature_names']

def predict_spoilage(temperature, ph_level, light_intensity, hours_data):
    """
    Predict water content over time and estimate spoilage
    """
    predictions = []
    for hour in hours_data:
        input_data = pd.DataFrame({
            'Temperature (°C)': [temperature],
            'pH Level': [ph_level],
            'Light Intensity': [light_intensity],
            'Hour': [hour]
        })
        pred = model.predict(input_data)[0]
        predictions.append(pred)
    
    # Define spoilage threshold (this is an example threshold)
    spoilage_threshold = 0.15
    
    # Find when water content exceeds threshold
    for i, pred in enumerate(predictions):
        if pred > spoilage_threshold:
            return i // 24  # Convert hours to days
    
    return 7  # Maximum prediction window

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    temperature = float(data['temperature'])
    ph_level = float(data['ph_level'])
    light_intensity = float(data['light_intensity'])
    
    # Generate predictions for next 7 days (168 hours)
    hours = list(range(24))
    days_until_spoilage = predict_spoilage(temperature, ph_level, light_intensity, hours)
    
    # Generate time series data for plotting
    time_series_data = []
    for hour in range(168):  # 7 days * 24 hours
        input_data = pd.DataFrame({
            'Temperature (°C)': [temperature],
            'pH Level': [ph_level],
            'Light Intensity': [light_intensity],
            'Hour': [hour % 24]
        })
        prediction = model.predict(input_data)[0]
        time_series_data.append({
            'hour': hour,
            'water_content': prediction
        })
    
    # Create plot
    df = pd.DataFrame(time_series_data)
    fig = px.line(df, x='hour', y='water_content', 
                  title='Predicted Water Content Over Time',
                  labels={'hour': 'Hours', 'water_content': 'Water Content'})
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'days_until_spoilage': days_until_spoilage,
        'plot_data': graphJSON
    })

if __name__ == '__main__':
    app.run(debug=True)