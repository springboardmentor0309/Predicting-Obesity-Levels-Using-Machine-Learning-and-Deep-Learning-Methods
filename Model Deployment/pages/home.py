# pages/home.py

import streamlit as st
from PIL import Image

def display_home():
    #st.markdown("<h1 style='text-align: center;'>🏥 Welcome to the AI-Powered Obesity Predictor!</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align: center; font-size: 1.2em;'>This app helps predict obesity status based on personalized inputs 
        and provides health suggestions tailored to support your well-being.</p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Features
    st.subheader("🚀 Features")
    st.write("""
    - **Predict Obesity Levels:** Get instant predictions based on your profile.
    - **Personalized Health Advice:** Receive tailored health suggestions for lifestyle improvement.
    - **Detailed Insights:** Learn more about the model’s accuracy and the importance of different features in predictions.
    """)

    st.image("images/home Healthcare.png", use_column_width=True, caption="Empowering health through data-driven insights")

    st.markdown("---")
    st.subheader("How to Get Started:")
    st.write("1. Go to the **Predict** page and fill in your details.")
    st.write("2. Receive personalized obesity predictions and health advice.")
    st.write("3. Explore the **Project Overview** for detailed insights into our approach and results.")
