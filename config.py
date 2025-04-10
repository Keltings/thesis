import streamlit as st

def setup_page():
    """Configure page settings and apply custom CSS"""
    # Set page config
    st.set_page_config(
        page_title="Financial Exclusion Prediction",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E88E5;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #6c428d;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }
        .section-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #212121;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .highlight {
            background-color: #FFB6C1;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid #1E88E5;
        }
        .metric-card {
            background-color: #FFB6C1;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .footer {
            font-size: 0.8rem;
            color: #757575;
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e0e0e0;
        }
    </style>
    """, unsafe_allow_html=True)

# Constants
DATA_PATH = {
    '2016': 'data/selected_2016_data.csv',
    '2021': 'data/selected_2021_data.csv'
}

# Model metrics data
MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]

# Scenario 1 (No Weights)
F1_S1_DEMO = [0.0222, 0.0057, 0.0000, 0.0000]
F1_S1_FULL = [0.7084, 0.6273, 0.0000, 0.6524]
AUC_S1_DEMO = [0.8371, 0.7837, 0.8025, 0.8419]
AUC_S1_FULL = [0.9779, 0.9445, 0.9687, 0.9713]

# Scenario 2 (Weighted)
F1_S2_DEMO = [0.0351, 0.1954, 0.1987, 0.1121]
F1_S2_FULL = [0.7168, 0.7205, 0.6829, 0.6648]
AUC_S2_DEMO = [0.8249, 0.7316, 0.7789, 0.8512]
AUC_S2_FULL = [0.9675, 0.9638, 0.9736, 0.9725]

# Scenario 3 (SMOTE)
F1_S3_DEMO = [0.3364, 0.3112, 0.3167, 0.3917]
F1_S3_FULL = [0.6903, 0.7202, 0.6770, 0.6760]
AUC_S3_DEMO = [0.8446, 0.7318, 0.7681, 0.8417]
AUC_S3_FULL = [0.9764, 0.9625, 0.9740, 0.9739]

# Feature groups
DEMOGRAPHIC_FEATURES = ["age", "gender", "education_level", "residence_type", 
                      "marital_status", "relationship_to_hh", "region", "population_weight"]