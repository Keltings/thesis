import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.helper_functions import get_image_download_link
from models.model_functions import get_lime_explanation

def show():
    """Display the Model Explainability page"""
    st.markdown('<div class="sub-header">Model Explainability with LIME</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    Explainable AI is crucial for understanding model predictions, especially in sensitive domains like financial inclusion.
    LIME (Local Interpretable Model-agnostic Explanations) helps explain individual predictions by approximating the complex model
    with a simpler, interpretable one around the prediction instance.
    </div>
    """, unsafe_allow_html=True)
    
    # Example selection
    st.markdown('<div class="section-header">Explore Example Cases</div>', unsafe_allow_html=True)
    
    example_choice = st.selectbox(
        "Select an example case:",
        ["Example 1: Rural woman with mobile money but no banking access",
         "Example 2: Urban young professional with multiple financial services",
         "Example 3: Elderly rural resident with limited financial services",
         "Example 4: Middle-aged urban resident with mixed financial access"]
    )
    
    # Model selection
    model_for_explanation = st.selectbox(
        "Select model to explain:",
        ["Decision Tree (Demographics + Behavior, SMOTE)", 
         "Logistic Regression (Demographics + Behavior, Weighted)"]
    )
    
    # Load example data
    example_data = load_example_data(example_choice)
    
    # Display example data
    display_example_data(example_data)
    
    # Generate LIME explanation
    st.markdown('<div class="section-header">LIME Explanation</div>', unsafe_allow_html=True)
    
    # Get LIME explanation
    lime_result = get_lime_explanation(example_choice, model_for_explanation)
    
    # Create prediction header
    display_prediction_header(lime_result)
    
    # Generate LIME visualization
    fig = create_lime_visualization(lime_result["feature_importances"])
    st.pyplot(fig)
    
    # Download LIME visualization
    st.markdown(get_image_download_link(fig, filename=f"lime_explanation_{example_choice.split(':')[0]}.png", 
                                     text="Download LIME visualization"), unsafe_allow_html=True)
    
    # Interpretation of LIME results
    st.markdown('<div class="section-header">Interpretation</div>', unsafe_allow_html=True)
    
    # Display LIME interpretation guide
    display_lime_interpretation_guide()
    
    # Display example-specific interpretation
    display_example_interpretation(example_choice)
    
    # Add educational explanation about LIME
    with st.expander("Learn More About LIME"):
        display_lime_educational_content()

def load_example_data(example_choice):
    """Load example data based on selection"""
    if "Example 1" in example_choice:
        return {
            "age": 42,
            "gender": "female",
            "education_level": "primary",
            "residence_type": "rural",
            "marital_status": "married",
            "relationship_to_hh": "spouse",
            "region": "eastern",
            "mobile_money_registered": 1,
            "bank_account_current": 0,
            "bank_account_savings": 0,
            "savings_mobile_banking": 1,
            "loan_mobile_banking": 1,
            "insurance_nhif": 0,
            "pension_nssf": 0
        }
    elif "Example 2" in example_choice:
        return {
            "age": 28,
            "gender": "male",
            "education_level": "university",
            "residence_type": "urban",
            "marital_status": "single",
            "relationship_to_hh": "head",
            "region": "nairobi",
            "mobile_money_registered": 1,
            "bank_account_current": 1,
            "bank_account_savings": 1,
            "savings_mobile_banking": 1,
            "loan_mobile_banking": 0,
            "insurance_nhif": 1,
            "pension_nssf": 1
        }
    elif "Example 3" in example_choice:
        return {
            "age": 67,
            "gender": "male",
            "education_level": "no_formal_education",
            "residence_type": "rural",
            "marital_status": "married",
            "relationship_to_hh": "head",
            "region": "western",
            "mobile_money_registered": 0,
            "bank_account_current": 0,
            "bank_account_savings": 0,
            "savings_mobile_banking": 0,
            "loan_mobile_banking": 0,
            "insurance_nhif": 0,
            "pension_nssf": 0
        }
    else:  # Example 4
        return {
            "age": 45,
            "gender": "female",
            "education_level": "secondary",
            "residence_type": "urban",
            "marital_status": "divorced",
            "relationship_to_hh": "head",
            "region": "central",
            "mobile_money_registered": 1,
            "bank_account_current": 1,
            "bank_account_savings": 0,
            "savings_mobile_banking": 1,
            "loan_mobile_banking": 0,
            "insurance_nhif": 1,
            "pension_nssf": 0
        }

def display_example_data(example_data):
    """Display example data in a tabular format"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Demographic Information**")
        demo_df = pd.DataFrame({
            "Feature": ["Age", "Gender", "Education Level", "Residence Type", 
                      "Marital Status", "Household Relation", "Region"],
            "Value": [example_data["age"], example_data["gender"], example_data["education_level"],
                    example_data["residence_type"], example_data["marital_status"], 
                    example_data["relationship_to_hh"], example_data["region"]]
        })
        st.table(demo_df)
    
    with col2:
        st.markdown("**Financial Behavior**")
        behavior_df = pd.DataFrame({
            "Feature": ["Mobile Money Account", "Current Bank Account", "Savings Bank Account", 
                      "Mobile Banking Savings", "Mobile Banking Loan", "NHIF Insurance", "NSSF Pension"],
            "Value": ["Yes" if example_data["mobile_money_registered"] == 1 else "No",
                    "Yes" if example_data["bank_account_current"] == 1 else "No",
                    "Yes" if example_data["bank_account_savings"] == 1 else "No",
                    "Yes" if example_data["savings_mobile_banking"] == 1 else "No",
                    "Yes" if example_data["loan_mobile_banking"] == 1 else "No",
                    "Yes" if example_data["insurance_nhif"] == 1 else "No",
                    "Yes" if example_data["pension_nssf"] == 1 else "No"]
        })
        st.table(behavior_df)

def display_prediction_header(lime_result):
    """Display the prediction header with result and probability"""
    prediction = lime_result["prediction"]
    prediction_proba = lime_result["probability"]
    prediction_color = "#D32F2F" if prediction == "Financially Excluded" else "#388E3C"
    
    st.markdown(f"""
    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="margin-bottom: 10px;">Prediction: <span style="color: {prediction_color};">{prediction}</span></h3>
        <p style="font-size: 16px;">Probability: {prediction_proba:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

def  create_lime_visualization(feature_importances):
    """Create the LIME-style horizontal bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort features by absolute weight
    sorted_features = sorted(feature_importances, key=lambda x: abs(x["weight"]), reverse=True)
    
    # Extract feature names and weights
    features = [item["feature"] for item in sorted_features]
    weights = [item["weight"] for item in sorted_features]
    
    # Create colors based on weight sign (green for negative, red for positive)
    colors = ["#D32F2F" if w > 0 else "#388E3C" for w in weights]
    
    # Create horizontal bar chart
    bars = ax.barh(features, weights, color=colors)
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels
    ax.set_xlabel('Feature Weight')
    ax.set_title('LIME Feature Importance')
    
    # Add weight annotations
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.06
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
              f'{width:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def display_lime_interpretation_guide():
    """Display the guide for interpreting LIME results"""
    st.markdown("""
    <div class="highlight">
    <b>How to read the LIME explanation:</b><br>
    Each horizontal bar shows how a feature contributes to the prediction. 
    <span style="color: #D32F2F;">Red bars</span> push the prediction toward financial exclusion, 
    while <span style="color: #388E3C;">green bars</span> push toward financial inclusion.
    The length of each bar indicates how strongly that feature influences the prediction.
    </div>
    """, unsafe_allow_html=True)

def display_example_interpretation(example_choice):
    """Display interpretation specific to the selected example"""
    if "Example 1" in example_choice:
        st.markdown("""
        **Interpretation for Rural Woman with Mobile Money:**
        
        This individual is predicted as financially excluded (71% probability) despite having mobile money access.
        Key factors pushing toward exclusion are:
        - Lack of traditional banking accounts
        - Rural location
        - Primary education level
        
        The mobile money access and mobile banking loan slightly reduce the exclusion probability,
        but not enough to change the overall prediction. This illustrates how mobile money alone
        is not always sufficient to achieve financial inclusion.
        """)
    elif "Example 2" in example_choice:
        st.markdown("""
        **Interpretation for Urban Young Professional:**
        
        This individual is strongly predicted as financially included (88% confidence).
        The model identifies multiple factors contributing to inclusion:
        - Having both current and savings bank accounts
        - University education
        - Urban residence
        - Formal insurance and pension
        
        The only slight factor pushing toward exclusion is younger age, but this is
        heavily outweighed by the positive financial access indicators.
        """)
    elif "Example 3" in example_choice:
        st.markdown("""
        **Interpretation for Elderly Rural Resident:**
        
        This individual has a very high probability (95%) of being financially excluded.
        The model identifies a complete lack of financial services as the primary driver:
        - No banking, mobile money, insurance, or pension
        - No formal education
        - Rural location
        - Elderly age
        
        This example highlights how multiple disadvantages can compound to create severe
        financial exclusion that requires targeted interventions across multiple dimensions.
        """)
    else:  # Example 4
        st.markdown("""
        **Interpretation for Middle-aged Urban Resident:**
        
        This individual is predicted as financially included, but with a lower confidence (62%).
        The mixed financial access creates some interesting tensions in the model:
        
        Inclusion factors:
        - Having a current bank account
        - Urban location
        - Secondary education
        - Mobile money access
        - Health insurance
        
        Exclusion factors:
        - No savings account
        - No pension
        
        This example shows how financial inclusion exists on a spectrum, and how different
        financial services address different aspects of inclusion.
        """)

def display_lime_educational_content():
    """Display educational content about LIME"""
    st.markdown("""
    ### How LIME Works
    
    LIME (Local Interpretable Model-agnostic Explanations) helps explain individual predictions 
    from any machine learning model by approximating it locally with an interpretable model.
    
    The algorithm:
    1. Takes a prediction and the original instance
    2. Generates perturbed samples around that instance
    3. Gets predictions from the complex model for these samples
    4. Trains a simple model (like linear regression) on this dataset
    5. Explains the original prediction using the weights from the simple model
    
    ### Benefits of LIME for Financial Exclusion Models
    
    - **Transparency**: Helps regulators and users understand why a prediction was made
    - **Fairness**: Identifies if models are using problematic features or creating discrimination
    - **Trust**: Builds confidence in model predictions by showing the reasoning
    - **Improvement**: Highlights areas where models might be failing or using incorrect patterns
    
    ### Limitations
    
    - LIME provides local explanations only (specific to individual predictions)
    - Different runs can produce slightly different explanations
    - Feature interactions may not be fully captured
    - The quality of explanation depends on how well the local model approximates the original model
    """)