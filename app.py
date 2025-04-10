import streamlit as st
from config import setup_page
from utils.helper_functions import get_image_download_link
from pages import introduction, data_exploration, model_comparison, prediction, explainability

# Set up page config and styling
setup_page()

# App title and description
st.markdown('<div class="main-header">Financial Exclusion Prediction with Explainable AI</div>', unsafe_allow_html=True)
st.markdown("""
This application demonstrates how machine learning models can predict financial exclusion 
based on demographic and behavioral features. It also provides explainability through LIME (Local Interpretable 
Model-agnostic Explanations) to help understand why predictions are made.
""")

# Sidebar navigation
st.sidebar.title("💡 Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", 
    ["Introduction", "Data Exploration", "Model Comparison", "Interactive Prediction", "Model Explainability"])

# Route to the appropriate page
if app_mode == "Introduction":
    introduction.show()
elif app_mode == "Data Exploration":
    data_exploration.show()
elif app_mode == "Model Comparison":
    model_comparison.show()
elif app_mode == "Interactive Prediction":
    prediction.show()
elif app_mode == "Model Explainability":
    explainability.show()

# Footer
st.markdown('<div class="footer">Financial Exclusion Prediction App with Explainable AI</div>', unsafe_allow_html=True)