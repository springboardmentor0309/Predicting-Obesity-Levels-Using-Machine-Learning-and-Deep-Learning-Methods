# pages/predict.py

import streamlit as st
import numpy as np
import pickle
from pytorch_tabnet.tab_model import TabNetClassifier
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain

# Set the page configuration
st.set_page_config(layout="centered", page_title="🏥 Obesity Predictor", page_icon="🧬")

# Load scaler and TabNet model
with open("models/balancenewscaler.pkl", "rb") as f:
    scaler = pickle.load(f)

transfer_tabnet_model = TabNetClassifier()
transfer_tabnet_model.load_model("models/balancetabnet_model1.pt.zip")

# LangChain setup for health suggestions
llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", api_key="AIzaSyBPqdPNvmnCJjcPrHvuFCClatxGfMjAST8")
prompt_template = PromptTemplate(
    input_variables=["obesity_status"],
    template="""
    ## Health and Wellness Guidance - Safety and Positivity First
    You are a helpful, respectful AI designed to provide health and wellness suggestions based on a person’s obesity status.
    ### Guidelines for Providing Advice:
    - **Positivity:** Frame all recommendations in an encouraging tone.
    - **Safety First:** Avoid advice that could harm or endanger users in any way.
    - **Individual Respect:** Refrain from generalized or critical language. Tailor suggestions to support a healthy and balanced lifestyle.

    ### Obesity Status Classification:
    - **Underweight**: Recommendations for balanced calorie intake and strength-building activities.
    - **Normal Weight**: Maintenance suggestions for a balanced lifestyle.
    - **Overweight**: Suggestions focused on balanced nutrition and moderate physical activities.
    - **Obese**: Tips for sustainable lifestyle changes and potential medical consultation.

    Obesity Status: {obesity_status}
    Health Suggestions:
   """
)

chain = LLMChain(prompt=prompt_template, llm=llm)

# Define the function for prediction
def predict_obesity_status(age, gender, height_cm, weight_kg, bmi, physical_activity_level, diet_type,
                           smoking_habits, alcohol_consumption, family_history_obesity, blood_pressure,
                           cholesterol_levels, education_level, income_level, geographical_region):
    input_data = np.array([[age, gender, height_cm, weight_kg, bmi,
                            physical_activity_level, diet_type, smoking_habits,
                            alcohol_consumption, family_history_obesity, blood_pressure,
                            cholesterol_levels, education_level, income_level, geographical_region]])
    input_data[:, :4] = scaler.transform(input_data[:, :4])
    prediction = transfer_tabnet_model.predict(input_data)
    obesity_labels = ["Underweight", "Normal weight", "Overweight", "Obese"]
    return obesity_labels[int(prediction[0])]

# Display Predict Page
def display_predict_page():
    #st.title("📊 Obesity Prediction")

    # Two-column layout for inputs
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Personal Information")
        age = st.number_input("🔢 Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("⚥ Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        height_cm = st.number_input("📏 Height (cm)", min_value=100, max_value=250, value=175)
        weight_kg = st.number_input("⚖️ Weight (kg)", min_value=30, max_value=200, value=70)
        bmi = st.number_input("📊 BMI", min_value=10.0, max_value=50.0, value=22.86)

    with col2:
        st.subheader("💼 Lifestyle Details")
        physical_activity_level = st.selectbox("🏃 Physical Activity Level", options=[1, 2, 3, 4],  format_func=lambda x: ["Very Low", "Low", "Moderate", "High"][x-1])
        diet_type = st.selectbox("🥗 Diet Type", options=[0, 1, 2], format_func=lambda x: ["Non-Vegetarian", "Vegetarian", "Vegan"][x])
        smoking_habits = st.selectbox("🚬 Smoking Habits", options=[ 0, 1, 2], format_func=lambda x: ["Non-Smoker", "Occasional Smoker", "Regular Smoker"][x])
        alcohol_consumption = st.selectbox("🍷 Alcohol Consumption", options=[0, 1, 2], format_func=lambda x: ["None", "Moderate", "Heavy"][x])
        family_history_obesity = st.selectbox("👨‍👩‍👧 Family History of Obesity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        blood_pressure = st.selectbox("💉 Blood Pressure", options=[0, 1, 2, 3], format_func=lambda x: ["Normal", "Prehypertension", "Hypertension Stage 1", "Hypertension Stage 2"][x])
        cholesterol_levels = st.selectbox("🧬 Cholesterol Levels", options=[1, 2, 3], format_func=lambda x: ["Normal", "Borderline High", "High"][x-1])
        education_level = st.selectbox("🎓 Education Level", options=[1, 2, 3, 4, 5], format_func=lambda x: ["Primary", "Secondary", "High School", "Undergraduate", "Graduate"][x-1])
        income_level = st.selectbox("💵 Income Level", options=[1, 2, 3], format_func=lambda x: ["Low", "Middle", "High"][x-1])
        geographical_region = st.selectbox("🌍 Geographical Region", options=[1, 2, 3], format_func=lambda x: ["Urban", "Suburban", "Rural"][x-1])

    # Predict button
    if st.button("🚀 Predict"):
        # Get prediction
        obesity_status = predict_obesity_status(age, gender, height_cm, weight_kg, bmi,
                                                physical_activity_level, diet_type, smoking_habits,
                                                alcohol_consumption, family_history_obesity,
                                                blood_pressure, cholesterol_levels, education_level,
                                                income_level, geographical_region)

        # Display results below the inputs
        st.subheader("Prediction Result")
        st.write(f"**Obesity Status:** {obesity_status}")

        # Generate health suggestions using LangChain
        response = chain.invoke({"obesity_status": obesity_status})
        health_suggestions = response.get("text", "No suggestions available.")

        # Display personalized suggestions below the prediction result
        st.subheader("Personalized Health Suggestions")
        st.markdown(f"<div style='background-color:#E8F6EF;padding:15px;border-radius:10px;'>{health_suggestions}</div>", unsafe_allow_html=True)

# Display Predict Page
display_predict_page()
