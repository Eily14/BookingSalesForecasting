import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, json, render_template
from datetime import datetime, date, timedelta
import calendar
import holidays

app = Flask(__name__)

# Load the XGBoost model
model_path = 'booksalesforecasting-model.pkl'
salesxgbmodel = joblib.load(model_path)

# define API endpoint
@app.route('/sales', methods=['POST'])
def predict():

    # get user input and append to the DataFrame
    data = request.get_json()
    month_name = data['month']
    year = data['year']

    # Convert month name to month number
    month_number = list(calendar.month_name).index(month_name.capitalize())

    # Create the 'test' DataFrame
    test = pd.DataFrame(columns=['Date', 'Year', 'Month', 'Holidays', 'M1', 'M2'])

    # Add the user input values to the DataFrame
    test.loc[0] = [pd.Timestamp(year=int(year), month=month_number, day=1), int(year), int(month_number), 0, 0.0, 0.0]

    # Get Philippines holidays
    ph_holidays = holidays.country_holidays('PH')

    # Create 'Holidays' column and check if the date is a holiday
    test['Holidays'] = [1 if x in ph_holidays else 0 for x in test['Date']]

    # Convert 'Holidays' column to integer type (optional if already 0/1)
    test['Holidays'] = test['Holidays'].astype(int)

    # Cyclical features
    test['M1'] = np.sin(test['Month'] * (2 * np.pi / 12))
    test['M2'] = np.cos(test['Month'] * (2 * np.pi / 12))

    # Drop 'Date' column as it is no longer needed
    test.drop('Date', axis=1, inplace=True)

    # Make predictions using the model
    predictions = salesxgbmodel.predict(test)

    # Add 'Prediction' column to the DataFrame
    test['Prediction'] = predictions
    test['Prediction'] = test['Prediction'].round().astype(int)

    output = test['Prediction'].tolist()

    return jsonify({'Prediction': output})

if __name__ == '__main__':
    app.run(debug=True)
