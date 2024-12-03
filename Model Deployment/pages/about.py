# pages/about.py

import streamlit as st
from PIL import Image

def display_about():
    st.markdown("<h2 style='text-align: center;'>🩺 About Us 👨🏼‍💻</h2>", unsafe_allow_html=True)
    st.write(
        """
        This application was developed to provide accurate obesity predictions using machine learning. Our team has a 
        strong background in data science, healthcare, and software engineering, committed to creating a meaningful 
        impact on public health.
        
        **Mission:** Empower individuals to take proactive steps for better health through personalized insights.
        
        **Vision:** Use AI-driven insights to foster a healthier society.
        """
    )

    #st.subheader("The Team 👥")
    #st.write("""
    #- **AI Experts**: Our AI specialists developed and fine-tuned the prediction model for accuracy and efficiency.
    #- **Data Scientists**: They explored various data sets, optimized features, and ensured data quality.
    #- **Healthcare Consultants**: Provided input on healthcare-related features and tailored health suggestions.
    #""")

    st.image("images/working to healthier.webp", use_column_width=True, caption="Our team dedicated to building a healthier tomorrow")
