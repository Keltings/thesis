import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from models.model_functions import make_prediction

def show():
    """Display the Interactive Prediction page"""
    st.markdown('<div class="sub-header">Interactive Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This section allows you to input demographic and behavioral information to predict 
    whether someone is likely to be financially excluded. You can try different scenarios 
    to see how various factors influence the prediction.
    """)
    
    # Model choice
    model_choice = st.selectbox(
        "Choose prediction model:",
        ["Decision Tree (Demographics + Behavior, SMOTE)", 
         "Logistic Regression (Demographics + Behavior, Weighted)",
         "Gradient Boosting (Demographics Only, SMOTE)"]
    )
    
    # Design the input form
    st.markdown('<div class="section-header">Input Information</div>', unsafe_allow_html=True)
    
    # Get user inputs
    input_data = get_user_inputs(model_choice)
    
    # Make prediction button
    if st.button("Predict Financial Exclusion Status"):
        # Make prediction
        prediction_results = make_prediction(input_data, model_choice)
        
        # Display prediction
        display_prediction_results(prediction_results)
        
        # Display key factors
        display_influential_factors(prediction_results["factors"])

def get_user_inputs(model_choice):
    """Collect user inputs for prediction"""
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        st.markdown("**Demographic Information**")
        
        input_data["age"] = st.slider("Age", 18, 100, 35)
        input_data["gender"] = st.selectbox("Gender", ["male", "female"])
        input_data["education_level"] = st.selectbox(
            "Education Level", 
            ["no_formal_education", "primary", "secondary", "university"]
        )
        input_data["residence_type"] = st.selectbox("Residence Type", ["urban", "rural"])
        input_data["marital_status"] = st.selectbox(
            "Marital Status", 
            ["single", "married", "divorced", "widowed", "separated"]
        )
        input_data["relationship_to_hh"] = st.selectbox(
            "Relationship to Household Head", 
            ["head", "spouse", "son_daughter", "parent", "other_relative", "not_related"]
        )
        input_data["region"] = st.selectbox(
            "Region", 
            ["nairobi", "central", "coast", "eastern", "north_eastern", "nyanza", "rift_valley", "western"]
        )
    
    with col2:
        st.markdown("**Behavioral Information**")
        
        # Only include behavioral fields if not using demographics-only model
        if "Demographics Only" not in model_choice:
            input_data["mobile_money"] = st.checkbox("Has Mobile Money Account", value=True)
            input_data["bank_account"] = st.checkbox("Has Bank Account", value=False)
            input_data["savings_account"] = st.checkbox("Has Savings Account", value=False)
            input_data["loan"] = st.checkbox("Has Any Loan", value=False)
            input_data["insurance"] = st.checkbox("Has Any Insurance", value=False)
            input_data["pension"] = st.checkbox("Has Pension", value=False)
            
            # Additional behavioral details if needed
            st.markdown("**Financial Details (Optional)**")
            input_data["has_debit_card"] = st.checkbox("Has Debit Card", value=False)
            input_data["has_credit_card"] = st.checkbox("Has Credit Card", value=False)
            input_data["savings_microfinance"] = st.checkbox("Saves with Microfinance", value=False)
            input_data["savings_sacco"] = st.checkbox("Saves with SACCO", value=False)
            input_data["savings_group"] = st.checkbox("Saves with Group/Chama", value=False)
        else:
            st.info("Behavioral features not included in demographics-only model.")
    
    return input_data

def display_prediction_results(results):
    """Display prediction results with visualization"""
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Prediction result
        prediction = results["prediction"]
        exclusion_probability = results["probability"]
        prediction_color = "#D32F2F" if prediction == "Financially Excluded" else "#388E3C"
        
        st.markdown(f"""
        <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="margin-bottom: 10px;">Prediction Result</h3>
            <p style="font-size: 24px; font-weight: bold; color: {prediction_color};">{prediction}</p>
            <p style="font-size: 18px;">Probability: {exclusion_probability:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Probability gauge
        fig = create_gauge_chart(exclusion_probability)
        st.pyplot(fig)

def create_gauge_chart(probability):
    """Create a gauge chart visualization for the prediction probability"""
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#5a6159')
    
    # Create gauge chart
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(0, 1)
    
    # Draw gauge background
    ax.barh(0, 1, left=0, height=0.3, color='#FFFFcc')
    
    # Draw gauge value
    ax.barh(0, probability, left=0, height=0.3, color=cmap(norm(probability)))
    
    # Add marker for threshold
    ax.axvline(x=0.5, color='#757575', linestyle='--', alpha=0.7)
    ax.text(0.5, -0.5, 'Threshold', ha='center', va='center', color='#757575', fontsize=10)
    
    # Add labels
    ax.text(0.0, -0.2, 'Included', ha='left', va='center', fontsize=10, color='#388E3C')
    ax.text(1.0, -0.2, 'Excluded', ha='right', va='center', fontsize=10, color='#D32F2F')
    
    # Value marker
    ax.text(probability, 0.5, f'{probability:.2f}', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="circle,pad=0.3", fc='white', ec=cmap(norm(probability))))
    
    # Remove axes and set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1)
    ax.axis('off')
    
    return fig

def display_influential_factors(factors):
    """Display the factors that influenced the prediction"""
    st.markdown('<div class="section-header">Key Factors Influencing Prediction</div>', unsafe_allow_html=True)
    
    # Display factors
    for factor in factors:
        if factor["direction"] == "positive":
            direction_color = "#D32F2F"  # Red for factors increasing exclusion
            arrow = "↑"
        else:
            direction_color = "#388E3C"  # Green for factors decreasing exclusion
            arrow = "↓"
            
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px; background-color: #f8f8f8; padding: 10px; border-radius: 5px;">
            <div style="flex: 4;">
                <span style="font-weight: bold;">{factor["feature"]}</span>
            </div>
            <div style="flex: 1; text-align: right;">
                <span style="color: {direction_color}; font-weight: bold; font-size: 18px;">{arrow} {factor["impact"]:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Explanation
    st.markdown("""
    <div class="highlight" style="margin-top: 20px;">
    <b>Explanation:</b><br>
    The factors above show the features that most influenced this prediction. 
    <span style="color: #D32F2F;">Red arrows (↑)</span> indicate factors that increase the likelihood of financial exclusion, 
    while <span style="color: #388E3C;">green arrows (↓)</span> indicate factors that decrease it.
    </div>
    """, unsafe_allow_html=True)