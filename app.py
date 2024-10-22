from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import joblib  # Add this import

app = Flask(__name__)

# Load the model and scaler
model = joblib.load(open('lgb_model.pkl', 'rb'))  # Using joblib instead of pickle
scaler = joblib.load(open('scaler.pkl', 'rb'))  # Using joblib for the scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve the values from the form
        PM25 = float(request.form['PM2.5'])
        PM10 = float(request.form['PM10'])
        NO = float(request.form['NO'])
        NO2 = float(request.form['NO2'])
        NOx = float(request.form['NOx'])
        CO = float(request.form['CO'])
        SO2 = float(request.form['SO2'])

        # Prepare the input for the model
        input_data = [PM25, PM10, NO, NO2, NOx, CO, SO2]
        df = pd.DataFrame([input_data])

        # Scale the input data
        scaled_input = pd.DataFrame(scaler.transform(df), columns=df.columns)

        # Make the prediction using the model on scaled data
        prediction = model.predict(scaled_input)[0]
        
        # Result analysis based on the prediction
        if prediction <= 50:
            aqi_category = "Good"
            aqi_description = "Air quality is considered satisfactory, and air pollution poses little or no risk."
        elif prediction <= 100:
            aqi_category = "Satisfactory"
            aqi_description = "Air quality is acceptable, but there may be minor concerns for some people who are highly sensitive to air pollution."
        elif prediction <= 200:
            aqi_category = "Moderate"
            aqi_description = "Moderate air quality may pose a health concern for people sensitive to air pollution."
        elif prediction <= 300:
            aqi_category = "Poor"
            aqi_description = "Poor air quality is detrimental to health, especially for sensitive groups."
        elif prediction <= 400:
            aqi_category = "Very Poor"
            aqi_description = "Very poor air quality poses severe health risks."
        else:
            aqi_category = "Severe"
            aqi_description = "Severe air quality can lead to serious health effects for everyone."

        # Return the result to the template
        return render_template('index.html', prediction=prediction, aqi_category=aqi_category, aqi_description=aqi_description)

if __name__ == "__main__":
    app.run(debug=True)
