from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import pandas as pd
app = Flask(__name__)

# Load pre-trained models and scalers
model_4_category = joblib.load('model_4_category.pkl')
model_7_category = joblib.load('model_7_category.pkl')
scaler_4_category = joblib.load('scaler_4_category.pkl')
scaler_7_category = joblib.load('scaler_7_category.pkl')
# Define the columns that were used during training
required_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 
                   'Age', 'Height', 'Weight', 'BMI', 'CH2O', 'FAF', 'FCVC', 'NCP', 'TUE']

# Mapping of text inputs to numerical values
text_to_number_map = {
    'Gender': {'Male': 0, 'Female': 1},
    'family_history_with_overweight': {'No': 0, 'Yes': 1},
    'FAVC': {'No': 0, 'Yes': 1},
    'CAEC': {'No': 0, 'Some': 1, 'Full': 2},
    'SMOKE': {'No': 0, 'Yes': 1},
    'SCC':{'No':0,'Yes':1},
    'CALC': {'None': 0, 'Some': 1, 'Full': 2},
    'MTRANS': {'Motorbike': 0, 'Bike': 1, 'Public': 2, 'Walking': 3}
}

def preprocess_input(data, model_type):
    # Map text inputs to numerical values
    for key in text_to_number_map:
        if key in data:
            data[key] = text_to_number_map[key].get(data[key], data[key])

    # Calculate BMI if not provided
    if 'BMI' not in data or pd.isnull(data['BMI']):
        data['BMI'] = data['Weight'] / (data['Height'] ** 2)

    # Create DataFrame and ensure it matches training columns
    input_data = pd.DataFrame([data])
    input_data = input_data.reindex(columns=required_columns).fillna(0)

    # Reorder and drop any irrelevant columns
    if model_type == '4_category':
        input_data = input_data[scaler_4_category.feature_names_in_]
        input_scaled = scaler_4_category.transform(input_data)
    else:
        input_data = input_data[scaler_7_category.feature_names_in_]
        input_scaled = scaler_7_category.transform(input_data)

    return input_scaled


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    model_type = request.form['model_type']
    if model_type == '4_category':
        return redirect(url_for('form_4_category'))
    else:
        return redirect(url_for('form_7_category'))

@app.route('/form_4_category')
def form_4_category():
    return render_template('form_4_category.html')

@app.route('/form_7_category')
def form_7_category():
    return render_template('form_7_category.html')

@app.route('/predict_4_category', methods=['POST'])
def predict_4_category():
    # Extract form data and preprocess for prediction
    input_data = {
        'Gender': request.form['Gender'],
        'family_history_with_overweight': request.form['family_history_with_overweight'],
        'FAVC': request.form['FAVC'],
        'CAEC': request.form['CAEC'],
        'SMOKE': request.form['SMOKE'],
        'SCC': request.form['SCC'], 
        'CALC': request.form['CALC'],
        'MTRANS': request.form['MTRANS'],
        'Age': float(request.form['Age']),
        'Height': float(request.form['Height']),
        'Weight': float(request.form['Weight']),
        'CH2O': float(request.form['CH2O']),
        'FAF': float(request.form['FAF']),
        'FCVC': float(request.form['FCVC']),
        'NCP': float(request.form['NCP']),
        'TUE': float(request.form['TUE'])
    }
    input_scaled = preprocess_input(input_data, '4_category')
    prediction = model_4_category.predict(input_scaled)

    # Define the obesity levels and suggestions
    prediction_label = ['Underweight', 'Normal', 'Obesity', 'Overweight'][prediction[0]]
    
    # Suggestions based on prediction
    suggestions = {
        'Underweight': {
            'message': "It's important to maintain a balanced diet and avoid excessive weight loss. Consult a nutritionist to ensure you're getting enough nutrients.",
            'emoji': "🍏💪"
        },
        'Normal': {
            'message': "Great! Keep maintaining a healthy lifestyle by eating well, exercising regularly, and getting enough rest.",
            'emoji': "🏃‍♀️🥗😊"
        },
        'Obesity': {
            'message': "It's important to consult a healthcare professional to manage your weight. Regular exercise and a balanced diet can help reduce health risks.",
            'emoji': "⚖️💪🥦"
        },
        'Overweight': {
            'message': "Consider working on a plan to lose weight healthily. Regular exercise, a balanced diet, and staying hydrated are key factors.",
            'emoji': "🏋️‍♂️🍏💧"
        }
    }

    # Get the suggestion based on prediction
    suggestion = suggestions.get(prediction_label, {'message': '', 'emoji': ''})
    
    # Pass prediction and suggestion separately to the template
    return render_template('result_4_category.html', 
                           prediction_label=prediction_label,
                           suggestion_message=suggestion['message'],
                           suggestion_emoji=suggestion['emoji'])


@app.route('/predict_7_category', methods=['POST'])
def predict_7_category():
    # Extract form data and preprocess for prediction
    input_data = {
        'Gender': request.form['Gender'],
        'family_history_with_overweight': request.form['family_history_with_overweight'],
        'FAVC': request.form['FAVC'],
        'CAEC': request.form['CAEC'],
        'SMOKE': request.form['SMOKE'],
        'SCC': request.form['SCC'], 
        'CALC': request.form['CALC'],
        'MTRANS': request.form['MTRANS'],
        'Age': float(request.form['Age']),
        'Height': float(request.form['Height']),
        'Weight': float(request.form['Weight']),
        'CH2O': float(request.form['CH2O']),
        'FAF': float(request.form['FAF']),
        'FCVC': float(request.form['FCVC']),
        'NCP': float(request.form['NCP']),
        'TUE': float(request.form['TUE'])
    }
    input_scaled = preprocess_input(input_data, '7_category')
    prediction = model_7_category.predict(input_scaled)

    # Define the obesity levels and suggestions
    prediction_label = ['Insufficient Weight', 'Normal Weight', 'Obesity Type_I', 'Obesity Type II', 
                        'Obesity Type III', 'Overweight Level I', 'Overweight Level II'][prediction[0]]
    
    # Suggestions based on prediction
    suggestions = {
        'Insufficient Weight': {
            'message': "It's important to eat a nutrient-dense diet. Consider speaking to a nutritionist for advice on how to gain weight safely.",
            'emoji': "🍽️💪"
        },
        'Normal Weight': {
            'message': "Keep up the good work! A balanced diet and regular physical activity will help maintain your current weight.",
            'emoji': "🏃‍♀️🥗😊"
        },
        'Obesity Type_I': {
            'message': "Consult a healthcare provider for a weight management plan. A healthy diet and regular exercise can help.",
            'emoji': "⚖️🍏🥦"
        },
        'Obesity Type II': {
            'message': "It's important to consult a healthcare professional. Regular exercise, along with a controlled diet, can help improve health.",
            'emoji': "⚖️💪🥗"
        },
        'Obesity Type III': {
            'message': "You should work closely with a healthcare provider for a tailored weight loss plan. A significant lifestyle change is essential.",
            'emoji': "⚠️🏋️‍♂️🍏"
        },
        'Overweight Level I': {
            'message': "Focus on healthy eating and increasing physical activity. This can help prevent further weight gain.",
            'emoji': "🏃‍♀️🍎💧"
        },
        'Overweight Level II': {
            'message': "Consider working on a weight loss plan that includes diet changes and increased physical activity.",
            'emoji': "⚖️🥦🏋️‍♀️"
        }
    }

    # Get the suggestion based on prediction
    suggestion = suggestions.get(prediction_label, {'message': '', 'emoji': ''})
    
    # Pass prediction and suggestion separately to the template
    return render_template('result_7_category.html', 
                           prediction_label=prediction_label,
                           suggestion_message=suggestion['message'],
                           suggestion_emoji=suggestion['emoji'])


if __name__ == "__main__":
    app.run(debug=True)
