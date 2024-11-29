from flask import Flask, request, render_template
import pickle
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

app = Flask(__name__)

# Load your trained scaler and model
with open("balancenewscaler.pkl", "rb") as f:
    scaler = pickle.load(f)

transfer_tabnet_model = TabNetClassifier()
transfer_tabnet_model.load_model("balancetabnet_model1.pt.zip")

# Categorical feature mappings
gender_map = {"Male": 0, "Female": 1}
blood_pressure_map = {"Normal": 0, "Elevated": 1, "Hypertension Stage 1": 2, "Hypertension Stage 2": 3}
cholesterol_levels_map = {"Normal": 0, "Borderline": 1, "High": 2}
diet_type_map = {"Non-Vegetarian": 0, "Vegetarian": 1}
smoking_habits_map = {"Non-smoker": 0, "Occasional Smoker": 1, "Regular Smoker": 2}
alcohol_consumption_map = {"Non-drinker": 0, "Drinks Occasionally": 1}
family_history_obesity_map = {"No": 0, "Yes": 1}
education_level_map = {"Primary": 0, "Secondary": 1, "Higher Education": 2}
income_level_map = {"Low": 0, "Middle": 1, "High": 2}
geographical_region_map = {"Urban": 0, "Suburban": 1, "Rural": 2}



@app.route('/')
def home():
    return render_template('index.html')
    

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Extract form data and map categorical features to integers
        try:
            age = int(request.form['age'])
            gender = gender_map[request.form['gender']]
            height_cm = float(request.form['height_cm'])
            weight_kg = float(request.form['weight_kg'])
            bmi = float(request.form['bmi'])
            physical_activity_level = int(request.form['physical_activity_level'])
            
            # Map categorical features to integers using predefined mappings
            diet_type = diet_type_map[request.form['diet_type']]
            smoking_habits = smoking_habits_map[request.form['smoking_habits']]
            alcohol_consumption = alcohol_consumption_map[request.form['alcohol_consumption']]
            family_history_obesity = family_history_obesity_map[request.form['family_history_obesity']]
            blood_pressure = blood_pressure_map[request.form['blood_pressure']]
            cholesterol_levels = cholesterol_levels_map[request.form['cholesterol_levels']]
            education_level = education_level_map[request.form['education_level']]
            income_level = income_level_map[request.form['income_level']]
            geographical_region = geographical_region_map[request.form['geographical_region']]

            # Create the input data array as expected by the model
            input_data = np.array([[age, gender, height_cm, weight_kg, bmi,
                                    physical_activity_level, diet_type, smoking_habits,
                                    alcohol_consumption, family_history_obesity, blood_pressure,
                                    cholesterol_levels, education_level, income_level, geographical_region]])

            # Apply the scaler transformation to numerical features (age, gender, height, weight, bmi)
            input_data[:, :4] = scaler.transform(input_data[:, :4])

            # Make prediction using the trained TabNet model
            prediction = transfer_tabnet_model.predict(input_data)

            # Map the predicted integer to obesity status labels
            obesity_labels = ["Underweight", "Normal weight", "Overweight", "Obese"]
            obesity_status = obesity_labels[int(prediction[0])]

            # Render the result on result.html
            return render_template('result.html', obesity_status=obesity_status)

        except KeyError as e:
            return f"Error: Missing form field {e}", 400

    else:  # If request.method == 'GET'
        # Render the form page (predict.html)
        return render_template('predict.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/About')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)


