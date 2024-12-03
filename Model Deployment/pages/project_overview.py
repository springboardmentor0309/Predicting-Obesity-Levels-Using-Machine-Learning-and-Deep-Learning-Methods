# pages/project_overview.py

import streamlit as st
from PIL import Image

def display_project_overview():
    st.markdown("<h2 style='text-align: center;'>📊 Project Overview 📑</h2>", unsafe_allow_html=True)

    st.subheader("Project Goals 🎯")
    st.write("""
    Our primary objective is to predict obesity levels based on individual characteristics and lifestyle habits. We aim to 
    provide users with personalized health suggestions to foster healthier lifestyle choices.
    """)

    st.markdown("---")

    st.subheader("Modeling Approach 🚀")
    st.write("""
    - **Data Collection**: We sourced extensive healthcare data relevant to obesity factors such as diet, activity level, and 
      family history.
    - **Feature Engineering**: Key features were selected to improve model accuracy, including **BMI**, **Diet Type**, and 
      **Age**, **Gender**,**Family History Obesity Status**,**Physical Activity**.
    - **Machine Learning Model**: We used a **TabNet Classifier** model to handle tabular data efficiently. Our approach 
      emphasizes accuracy and interpretability.
    """)

    st.markdown("---")

    st.subheader("Model Performance 📈")
    st.write("""
    - **Accuracy**: 98.5% on test data.
    - **Evaluation Metrics**: We assessed the model using accuracy, precision, recall, and F1-score.
    """)

    st.markdown("---")

    st.subheader("Feature Importance 🌟")
    st.write("""
    - **BMI**: Most impactful feature in determining obesity status.
    - **Weight & Height**: Combined with BMI, these features contribute significantly.
    - **Physical Activity**: Essential for understanding lifestyle impacts on obesity.
    """)

    st.image("images/project tabnet feature imp.png", use_column_width=True, caption="Feature Importance Analysis")

    st.markdown("---")

    st.subheader("Our Solution 🌐")
    st.write("""
    By combining data-driven insights with healthcare expertise, our application empowers users to make informed health 
    decisions. We envision expanding this tool to offer additional health assessments in the future.
    """)
