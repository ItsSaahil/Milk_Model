from flask import Flask, render_template, request, jsonify, send_file
import os
import logging
from io import BytesIO
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json
import plotly
import plotly.express as px

app = Flask(__name__)

# logging
logging.basicConfig(level=logging.INFO)

# Load the model (try a couple of locations and fail gracefully)
model = None
feature_names = []
possible_paths = [
    os.path.join('models', 'random_forest_model.pkl'),
    'random_forest_model.pkl',
]
for p in possible_paths:
    if os.path.exists(p):
        try:
            logging.info(f"Loading model from %s", p)
            model_info = joblib.load(p)
            model = model_info.get('model')
            feature_names = model_info.get('feature_names', [])
            break
        except Exception as e:
            logging.exception('Failed to load model from %s', p)

if model is None:
    logging.error('No model could be loaded. Prediction endpoint will return an error until a model is available.')

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

def generate_recommendations(temperature, ph_level, light_intensity, days_until_spoilage):
    """
    Generate recommendations and preventions based on current conditions
    """
    recommendations = []
    preventions = []
    risk_score = 0
    risk_level = "Low"
    
    # Temperature analysis
    if temperature < 4:
        preventions.append("Temperature is too cold. Avoid freezing milk to preserve nutritional content.")
        risk_score += 1
    elif temperature >= 4 and temperature <= 7:
        recommendations.append("✓ Optimal temperature maintained (4-7°C). Keep refrigeration consistent.")
    elif temperature > 7 and temperature <= 15:
        recommendations.append("⚠ Temperature slightly elevated. Gradually increase refrigeration to 4-7°C.")
        preventions.append("Avoid temperature fluctuations which accelerate spoilage.")
        risk_score += 2
    else:  # temperature > 15
        preventions.append("🔴 Temperature critically high! Refrigerate immediately to <7°C.")
        risk_score += 5
    
    # pH Level analysis
    if ph_level < 6.5:
        preventions.append("pH is acidic. Monitor for early signs of spoilage or fermentation.")
        risk_score += 2
    elif ph_level >= 6.5 and ph_level <= 7.0:
        recommendations.append("✓ pH level is optimal (6.5-7.0). Maintain current storage conditions.")
    elif ph_level > 7.0 and ph_level <= 7.8:
        recommendations.append("✓ pH slightly alkaline but acceptable. Good shelf life expected.")
    else:  # pH > 7.8
        preventions.append("🔴 pH is too alkaline. This indicates potential contamination or degradation.")
        risk_score += 3
    
    # Light Intensity analysis
    if light_intensity < 100:
        recommendations.append("✓ Low light exposure. Excellent—light degrades milk quality.")
    elif light_intensity >= 100 and light_intensity <= 200:
        recommendations.append("⚠ Moderate light exposure. Store in a shaded area when possible.")
        preventions.append("Use opaque containers to minimize light exposure.")
    elif light_intensity > 200 and light_intensity <= 500:
        preventions.append("⚠ High light exposure detected. Store in dark conditions to prevent UV-driven degradation.")
        risk_score += 2
    else:  # light > 500
        preventions.append("🔴 Very high light intensity! Move to a dark storage area immediately.")
        risk_score += 4
    
    # Determine risk level based on risk_score (NOT just days_until_spoilage)
    if risk_score >= 9:
        risk_level = "Critical"
        preventions.append("🔴 CRITICAL: Immediate action required to prevent spoilage.")
    elif risk_score >= 7:
        risk_level = "High"
        preventions.append("⚠ HIGH RISK: Multiple factors indicate accelerated spoilage potential.")
    elif risk_score >= 4:
        risk_level = "Medium"
        preventions.append("⚠ MEDIUM RISK: Some storage conditions need improvement.")
    else:
        risk_level = "Low"
        recommendations.append("✓ Good conditions. Current storage environment is suitable.")
    
    # Add shelf life info
    if days_until_spoilage <= 1:
        recommendations.append(f"🔴 Consume immediately or discard within 1 day to avoid health risks.")
    elif days_until_spoilage <= 2:
        recommendations.append(f"⚠ Use within next {days_until_spoilage} days.")
    elif days_until_spoilage <= 4:
        recommendations.append(f"✓ Expected shelf life: approximately {days_until_spoilage} days.")
    else:
        recommendations.append(f"✓ Expected shelf life: approximately {days_until_spoilage} days under current conditions.")
    
    return {
        'recommendations': recommendations,
        'preventions': preventions,
        'risk_score': min(risk_score, 10),  # Cap at 10
        'risk_level': risk_level
    }

def generate_optimal_conditions():
    """
    Generate optimal storage conditions guideline
    """
    return {
        'optimal_temperature': '4-7°C',
        'optimal_ph': '6.5-7.0',
        'optimal_light': '<100 lux',
        'storage_duration': '7-10 days in optimal conditions',
        'container_type': 'Opaque, food-grade containers',
        'storage_location': 'Back of refrigerator (coldest area)'
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if model is None:
            return jsonify({'error': 'Model not loaded on server.'}), 500

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

        # Generate recommendations and insights
        recommendations_data = generate_recommendations(temperature, ph_level, light_intensity, days_until_spoilage)
        optimal_conditions = generate_optimal_conditions()

        return jsonify({
            'days_until_spoilage': days_until_spoilage,
            'plot_data': graphJSON,
            'current_conditions': {
                'temperature': temperature,
                'ph_level': ph_level,
                'light_intensity': light_intensity
            },
            'recommendations': recommendations_data['recommendations'],
            'preventions': recommendations_data['preventions'],
            'risk_score': recommendations_data['risk_score'],
            'risk_level': recommendations_data['risk_level'],
            'optimal_conditions': optimal_conditions,
            'time_series_data': time_series_data
        })
    except Exception as e:
        logging.exception('Error in /predict')
        return jsonify({'error': str(e)}), 500

@app.route('/download-enriched', methods=['POST'])
def download_enriched():
    """
    Download enriched CSV with recommendations and insights
    """
    try:
        data = request.get_json()
        temperature = float(data['temperature'])
        ph_level = float(data['ph_level'])
        light_intensity = float(data['light_intensity'])
        days_until_spoilage = int(data['days_until_spoilage'])
        
        # Load original CSV
        df_original = pd.read_csv('dummy_data_300.csv')
        
        # Add computed columns
        recommendations_data = generate_recommendations(temperature, ph_level, light_intensity, days_until_spoilage)
        optimal_conditions = generate_optimal_conditions()
        
        # Create enriched dataframe with metadata
        enriched_rows = []
        for idx, row in df_original.iterrows():
            enriched_row = row.to_dict()
            enriched_row['Risk_Level'] = recommendations_data['risk_level']
            enriched_row['Risk_Score'] = recommendations_data['risk_score']
            enriched_row['Days_Until_Spoilage'] = days_until_spoilage
            enriched_row['Current_Temp_C'] = temperature
            enriched_row['Current_pH'] = ph_level
            enriched_row['Current_Light_Lux'] = light_intensity
            enriched_rows.append(enriched_row)
        
        df_enriched = pd.DataFrame(enriched_rows)
        
        # Convert to CSV
        csv_buffer = df_enriched.to_csv(index=False)
        
        # Add metadata as comments at the top
        metadata = f"""# Milk Quality Analysis - Enriched Report
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Current Conditions: Temp={temperature}°C, pH={ph_level}, Light={light_intensity} lux
# Predicted Shelf Life: {days_until_spoilage} days
# Risk Level: {recommendations_data['risk_level']}
# Risk Score: {recommendations_data['risk_score']}/10
#
# Recommendations:
"""
        for rec in recommendations_data['recommendations']:
            metadata += f"# - {rec}\n"
        metadata += "#\n# Prevention Tips:\n"
        for prev in recommendations_data['preventions']:
            metadata += f"# - {prev}\n"
        metadata += "#\n# Optimal Conditions:\n"
        for key, val in optimal_conditions.items():
            metadata += f"# {key}: {val}\n"
        metadata += "#\n"
        
        full_csv = metadata + csv_buffer
        
        # Create a BytesIO object for download
        output = BytesIO()
        output.write(full_csv.encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'milk_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        logging.exception('Error in /download-enriched')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # In development you may want the reloader, but excessive reloads (especially
    # triggered by unrelated file changes) can cause connection resets in the
    # browser. Set use_reloader=False while debugging this issue to avoid that.
    app.run(debug=True, use_reloader=False)