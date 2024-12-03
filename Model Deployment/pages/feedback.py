# pages/feedback.py

import streamlit as st
from PIL import Image

def display_feedback():
    st.markdown("<h2 style='text-align: center;'>Feedback 📬</h2>", unsafe_allow_html=True)
    st.write("We would love to hear your thoughts on the application. Please fill out the form below.")

    # Feedback form
    with st.form("feedback_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        feedback = st.text_area("Your Feedback")
        rating = st.slider("Rate Us", min_value=1, max_value=5, step=1)

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success("Thank you for your feedback!")
