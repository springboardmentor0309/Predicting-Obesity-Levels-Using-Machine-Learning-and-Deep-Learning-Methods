from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'C:\Users\AdithiyanPV\OneDrive\Desktop\WEB APP INFOSYS SPRINGBAORD\finallll.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict-form')
def predict_form():
    return render_template('predict.html')

@app.route('/health-tips')
def health_tips():
    return render_template('health_tips.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        gender = int(request.form['Gender'])  # Male: 0, Female: 1
        age = float(request.form['Age'])
        height = float(request.form['Height'])
        weight = float(request.form['Weight'])
        family_history = 1 if request.form['family_history_with_overweight'] == 'yes' else 0
        favc = 1 if request.form['FAVC'] == 'yes' else 0
        fcvc = int(request.form['FCVC'])
        ncp = int(request.form['NCP'])
        caec = ['Sometimes', 'Frequently', 'no', 'Always'].index(request.form['CAEC'])
        smoke = 1 if request.form['SMOKE'] == 'yes' else 0
        ch2o = float(request.form['CH2O'])
        scc = 1 if request.form['SCC'] == 'yes' else 0
        faf = int(request.form['FAF'])
        tue = int(request.form['TUE'])
        calc = ['Sometimes', 'Frequently', 'no'].index(request.form['CALC'])
        mtrans = request.form['MTRANS']

        # Map mtrans to numerical values
        mtrans_map = {'Walking': 0, 'Car': 1, 'Public': 2, 'Bike': 3}
        mtrans_encoded = mtrans_map.get(mtrans, -1)

        # Prepare the input data
        input_data = np.array([gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans_encoded, 0]).reshape(1, -1)

        # Get the prediction from the model
        prediction = model.predict(input_data)
        
        # Interpret the prediction
        result = 'Overweight' if prediction[0] == 1 else 'Normal Weight'

        return render_template('predict.html', result=result)

    except Exception as e:
        return render_template('predict.html', error=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)