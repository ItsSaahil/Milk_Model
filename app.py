from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS          # <-- NEW
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
CORS(app)                            # <-- ALLOW ALL ORIGINS

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------------------
# Load model (graceful fallback)
# ----------------------------------------------------------------------
model = None
feature_names = []
possible_paths = [
    os.path.join('models', 'random_forest_model.pkl'),
    'random_forest_model.pkl',
]
for p in possible_paths:
    if os.path.exists(p):
        try:
            logging.info("Loading model from %s", p)
            model_info = joblib.load(p)
            model = model_info.get('model')
            feature_names = model_info.get('feature_names', [])
            break
        except Exception as e:
            logging.exception('Failed to load model from %s', p)

if model is None:
    logging.error('No model could be loaded. Prediction endpoint will return an error until a model is available.')

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def predict_spoilage(temperature, ph_level, light_intensity, hours_data):
    """Predict water content over time and estimate spoilage based on conditions and critical factors."""
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

    # Calculate risk score for critical factor assessment
    risk_score = 0
    
    # Temperature risk
    if temperature < 4:
        risk_score += 1
    elif 7 < temperature <= 15:
        risk_score += 2
    elif temperature > 15:
        risk_score += 5

    # pH risk
    if ph_level < 6.5:
        risk_score += 2
    elif ph_level > 7.8:
        risk_score += 3

    # Light intensity risk
    if 100 <= light_intensity <= 200:
        risk_score += 0  # No additional risk
    elif 200 < light_intensity <= 500:
        risk_score += 2
    elif light_intensity > 500:
        risk_score += 4

    # If critical conditions (risk_score >= 7), mark as expired immediately
    if risk_score >= 7:
        return 0  # Expired immediately
    
    # Adjust spoilage threshold based on risk level
    base_threshold = 0.15
    if risk_score >= 5:
        spoilage_threshold = base_threshold * 0.8  # More sensitive threshold for high risk
    elif risk_score >= 3:
        spoilage_threshold = base_threshold * 0.9  # Slightly more sensitive
    else:
        spoilage_threshold = base_threshold
    
    # Check predictions against threshold
    for i, pred in enumerate(predictions):
        if pred > spoilage_threshold:
            days = i // 24
            # Apply risk-based reduction to predicted days
            if risk_score >= 5:
                days = max(1, days // 2)  # Halve the days for high risk, minimum 1 day
            elif risk_score >= 3:
                days = max(1, int(days * 0.75))  # Reduce by 25% for medium risk
            return days
    
    # Base maximum days adjusted by risk
    max_days = 7
    if risk_score >= 5:
        max_days = 3  # High risk conditions
    elif risk_score >= 3:
        max_days = 5  # Medium risk conditions
        
    return max_days


def generate_recommendations(temperature, ph_level, light_intensity, days_until_spoilage):
    """Generate recommendations, preventions and a risk score."""
    recommendations = []
    preventions = []
    risk_score = 0
    risk_level = "Low"

    # Temperature
    if temperature < 4:
        preventions.append("Temperature is too cold. Avoid freezing milk to preserve nutritional content.")
        risk_score += 1
    elif 4 <= temperature <= 7:
        recommendations.append("✓ Optimal temperature maintained (4-7°C). Keep refrigeration consistent.")
    elif 7 < temperature <= 15:
        recommendations.append("⚠ Temperature slightly elevated. Gradually increase refrigeration to 4-7°C.")
        preventions.append("Avoid temperature fluctuations which accelerate spoilage.")
        risk_score += 2
    else:
        preventions.append("🔴 Temperature critically high! Refrigerate immediately to <7°C.")
        risk_score += 5

    # pH
    if ph_level < 6.5:
        preventions.append("pH is acidic. Monitor for early signs of spoilage or fermentation.")
        risk_score += 2
    elif 6.5 <= ph_level <= 7.0:
        recommendations.append("✓ pH level is optimal (6.5-7.0). Maintain current storage conditions.")
    elif 7.0 < ph_level <= 7.8:
        recommendations.append("✓ pH slightly alkaline but acceptable. Good shelf life expected.")
    else:
        preventions.append("🔴 pH is too alkaline. This indicates potential contamination or degradation.")
        risk_score += 3

    # Light
    if light_intensity < 100:
        recommendations.append("✓ Low light exposure. Excellent—light degrades milk quality.")
    elif 100 <= light_intensity <= 200:
        recommendations.append("⚠ Moderate light exposure. Store in a shaded area when possible.")
        preventions.append("Use opaque containers to minimize light exposure.")
    elif 200 < light_intensity <= 500:
        preventions.append("⚠ High light exposure detected. Store in dark conditions to prevent UV-driven degradation.")
        risk_score += 2
    else:
        preventions.append("🔴 Very high light intensity! Move to a dark storage area immediately.")
        risk_score += 4

    # Risk level
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

    # Shelf-life message
    if days_until_spoilage == 0:
        recommendations.append(f"🔴 EXPIRED: Milk has spoiled under current critical conditions. Discard immediately for safety!")
    elif days_until_spoilage <= 1:
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
        'risk_score': min(risk_score, 10),
        'risk_level': risk_level
    }


def generate_optimal_conditions():
    return {
        'optimal_temperature': '4-7°C',
        'optimal_ph': '6.5-7.0',
        'optimal_light': '<100 lux',
        'storage_duration': '7-10 days in optimal conditions',
        'container_type': 'Opaque, food-grade containers',
        'storage_location': 'Back of refrigerator (coldest area)'
    }

# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
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

        # 1-day quick check (used only for spoilage day calculation)
        days_until_spoilage = predict_spoilage(temperature, ph_level, light_intensity, list(range(24)))

        # Full time series for the plot - adjust based on spoilage prediction
        max_hours = 168  # 7 days * 24h
        if days_until_spoilage == 0:
            max_hours = 24  # Show only 1 day if expired
        elif days_until_spoilage <= 3:
            max_hours = days_until_spoilage * 24 + 24  # Show predicted days + 1 extra day
        
        time_series_data = []
        for hour in range(max_hours):
            input_data = pd.DataFrame({
                'Temperature (°C)': [temperature],
                'pH Level': [ph_level],
                'Light Intensity': [light_intensity],
                'Hour': [hour % 24]
            })
            pred = model.predict(input_data)[0]
            time_series_data.append({'hour': hour, 'water_content': pred})

        df = pd.DataFrame(time_series_data)
        title = 'Predicted Water Content Over Time'
        if days_until_spoilage == 0:
            title += ' (EXPIRED - Critical Conditions)'
        elif days_until_spoilage <= 2:
            title += f' (High Risk - {days_until_spoilage} days remaining)'
        
        fig = px.line(df, x='hour', y='water_content',
                      title=title,
                      labels={'hour': 'Hours', 'water_content': 'Water Content'})
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        rec_data = generate_recommendations(temperature, ph_level, light_intensity, days_until_spoilage)
        optimal = generate_optimal_conditions()

        return jsonify({
            'days_until_spoilage': days_until_spoilage,
            'plot_data': graphJSON,
            'current_conditions': {
                'temperature': temperature,
                'ph_level': ph_level,
                'light_intensity': light_intensity
            },
            'recommendations': rec_data['recommendations'],
            'preventions': rec_data['preventions'],
            'risk_score': rec_data['risk_score'],
            'risk_level': rec_data['risk_level'],
            'optimal_conditions': optimal,
            'time_series_data': time_series_data
        })
    except Exception as e:
        logging.exception('Error in /predict')
        return jsonify({'error': str(e)}), 500


@app.route('/download-enriched', methods=['POST'])
def download_enriched():
    """Return a CSV enriched with the current analysis."""
    try:
        data = request.get_json()
        temperature = float(data['temperature'])
        ph_level = float(data['ph_level'])
        light_intensity = float(data['light_intensity'])
        days_until_spoilage = int(data['days_until_spoilage'])

        # Load dummy CSV (make sure the file exists in the project root)
        df_original = pd.read_csv('dummy_data_300.csv')

        rec_data = generate_recommendations(temperature, ph_level, light_intensity, days_until_spoilage)
        optimal = generate_optimal_conditions()

        enriched_rows = []
        for _, row in df_original.iterrows():
            r = row.to_dict()
            r['Risk_Level'] = rec_data['risk_level']
            r['Risk_Score'] = rec_data['risk_score']
            r['Days_Until_Spoilage'] = days_until_spoilage
            r['Current_Temp_C'] = temperature
            r['Current_pH'] = ph_level
            r['Current_Light_Lux'] = light_intensity
            enriched_rows.append(r)

        df_enriched = pd.DataFrame(enriched_rows)
        csv_buffer = df_enriched.to_csv(index=False)

        # ---- METADATA BLOCK ----
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata = f"""# Milk Quality Analysis - Enriched Report
# Generated: {now}
# Current Conditions: Temp={temperature}°C, pH={ph_level}, Light={light_intensity} lux
# Predicted Shelf Life: {days_until_spoilage} days
# Risk Level: {rec_data['risk_level']}
# Risk Score: {rec_data['risk_score']}/10
#
# Recommendations:
"""
        for rec in rec_data['recommendations']:
            metadata += f"# - {rec}\n"
        metadata += "#\n# Prevention Tips:\n"
        for prev in rec_data['preventions']:
            metadata += f"# - {prev}\n"
        metadata += "#\n# Optimal Conditions:\n"
        for k, v in optimal.items():
            metadata += f"# {k}: {v}\n"
        metadata += "#\n"

        full_csv = metadata + csv_buffer

        output = BytesIO()
        output.write(full_csv.encode('utf-8'))
        output.seek(0)

        filename = f"milk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename          # <-- FIXED TYPO
        )
    except Exception as e:
        logging.exception('Error in /download-enriched')
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # use_reloader=False avoids double-loading the model in dev
    app.run(debug=True, use_reloader=False)