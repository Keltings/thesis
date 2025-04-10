import streamlit as st

def show():
    """Display the Introduction page"""
    st.markdown('<div class="sub-header">Understanding Financial Exclusion</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="highlight">
        <b>Financial exclusion</b> refers to a situation where individuals lack access to appropriate financial 
        products and services. This app implements models to predict whether someone is likely to be 
        financially excluded based on:
        
        1. <b>Demographic features</b>: age, gender, education level, residence type, etc.
        2. <b>Behavioral features</b>: mobile money usage, savings accounts, loans, insurance, etc.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Approach and Methodology</div>', unsafe_allow_html=True)
        st.markdown("""
        Our approach tests three scenarios:
        1. **Models without sample weights** - Raw predictions on demographic and behavioral data
        2. **Models with sample weights** - Population-weighted predictions for demographic representation
        3. **Models with SMOTE** - Using synthetic minority oversampling to address class imbalance
        
        For each scenario, we compare different classifiers and feature sets to determine the optimal 
        approach for predicting financial exclusion.
        """)
        
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*E_Vmhpd-VDX-7NA91g1U1Q.png", 
                caption="LIME Explaining Predictions")
        
        st.markdown('<div class="section-header">Key Features of this App</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Data exploration** of demographic and behavioral features
        - **Model comparison** across different classifiers and scenarios
        - **Interactive prediction** to test model on custom inputs
        - **Explainability** through LIME to understand model decisions
        """)
    
    st.markdown('<div class="sub-header">How to Use This App</div>', unsafe_allow_html=True)
    st.markdown("""
    1. Start with the **Data Exploration** section to understand the dataset features
    2. Visit the **Model Comparison** section to see how different models perform
    3. Try making predictions with the **Interactive Prediction** tool
    4. Understand model decisions in the **Model Explainability** section
    """)