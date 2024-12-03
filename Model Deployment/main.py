import streamlit as st
from pages.home import display_home
from pages.predict import display_predict_page
from pages.about import display_about
from pages.feedback import display_feedback
from pages.project_overview import display_project_overview
from PIL import Image
from pages import predict

# Set the page configuration
st.set_page_config(page_title="🩺 Obesity Predictor 🏥", page_icon="🧬", layout="centered")

# Set header background image and title
def set_header_background(image_path):
    image = Image.open(image_path)
    st.image(image, use_column_width=True)
    st.markdown("<h3 style='text-align: center; color: #004466;'>🧑🏼‍⚕️ Obesity Predictor with Personalized Suggestions 🧬</h3>", unsafe_allow_html=True)

# Set Header Background Image
header_image_path = "images/home page people.png"
set_header_background(header_image_path)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Predict", "About", "Feedback", "Project Overview"))

# Render pages based on selection
if page == "Home":
    display_home()
elif page == "Predict":
    display_predict_page()
elif page == "About":
    display_about()
elif page == "Feedback":
    display_feedback()
elif page == "Project Overview":
    display_project_overview()
