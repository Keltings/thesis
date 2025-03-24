import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lime.lime_tabular

# Load necessary files (Encoder, Columns, Model)
encoder = joblib.load("onehot_encoder.pkl")
model = joblib.load("best_financial_exclusion_model.pkl")
columns_full_features = joblib.load("columns_full_features.pkl")
print("Columns loaded from columns_full_features.pkl:", columns_full_features)

# Define categorical features and behaviorals explicitly
categorical_features = ["gender", "education_level", "residence_type",
                        "marital_status", "region"]

behavioral_features = [col for col in columns_full_features if col not in categorical_features + ['age_group', 'population_weight']]


# Load your encoder (assuming you've saved it)
encoder = joblib.load("onehot_encoder.pkl")

# Categorical columns used during encoding
categorical_features = ['gender', 'education_level', 'residence_type', 
                        'marital_status','region']

# All other columns expected by the model
numeric_features = ['age',
    'mobile_money_access', 'formal_bank_user','saves_money','has_taken_loan','has_insurance','has_pension']

# Full preprocessing function
def preprocess_user_input(user_input):
    # Initialize all columns to default values (np.nan)
    full_input = {col: [np.nan] for col in numeric_features + categorical_features}

    # Update with user-provided values
    for key, value in user_input.items():
        full_input[key] = [value]

    # Create DataFrame
    input_df = pd.DataFrame(full_input)

    # Encode categorical features
    encoded_cats = encoder.transform(input_df[categorical_features])
    encoded_cat_names = encoder.get_feature_names_out(categorical_features)

    encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cat_names)

    # Drop original categorical columns
    input_df.drop(columns=categorical_features, inplace=True)

    # Combine numeric and encoded categorical data
    final_input = pd.concat([encoded_df,input_df.reset_index(drop=True)], axis=1)
    print(final_input)
    # Fill missing numeric data with median or a default value (e.g., zero)
    final_input.fillna(final_input.median(), inplace=True)

    return final_input[columns_full_features]

# Lime Explainer
def lime_explain(input_data):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.zeros((1, len(columns_full_features))),
        feature_names=columns_full_features,
        class_names=["Included", "Excluded"],
        mode="classification"
    )
    explanation = explainer.explain_instance(
        input_data.iloc[0].values, 
        model.predict_proba
    )
    return explanation

# Streamlit interface
st.title("🌍 Financial Exclusion Prediction (Kenya)")

st.sidebar.header("User Input")

# Collect demographic info
user_demo_input = {
    "age": st.sidebar.selectbox("Age", [18,19,20,21]),

    "gender": st.sidebar.selectbox("Gender", ["male", "female"]),
    "education_level": st.sidebar.selectbox("Education Level", ["primary", "secondary", "tertiary", "none"]),
    "residence_type": st.sidebar.selectbox("Residence Type", ["urban", "rural"]),
    "marital_status": st.sidebar.selectbox("Marital Status", ["married", "single", "divorced"]),
    #"relationship_to_hh": st.sidebar.selectbox("Relationship to Household", ["head", "spouse", "child", "other"]),
    "region": st.sidebar.selectbox("Region", ["coast", "central", "western", "nairobi", "nyanza", "rift valley", "north eastern"]),
    #"population_weight": st.sidebar.number_input("Population Weight", 0.1, 10.0, 1.0)
}

# Collect behavioral info (simplified)
st.sidebar.header("Behavioral Inputs")
user_behavioral_input = {

    "mobile_money_access": st.sidebar.checkbox("Registered Mobile Money"),
    "formal_bank_user": st.sidebar.checkbox("Savings Bank Account"),
    "saves_money": st.sidebar.checkbox("Saves"),

    "has_taken_loan": st.sidebar.checkbox("Loan via Mobile Banking"),
    "has_insurance": st.sidebar.checkbox("NHIF Insurance"),
    "has_pension": st.sidebar.checkbox("NSSF Pension"),
}

user_input = {**user_demo_input, **user_behavioral_input}

# Preprocess inputs
final_input = preprocess_user_input(user_input)

# Prediction
pred_prob = model.predict_proba(final_input)[0][1]
prediction = "🔴 Excluded (Financially)" if pred_prob >= 0.5 else "🟢 Included (Financially)"

st.header("📈 Prediction Result")
st.subheader(prediction)
st.write(f"### Probability of Exclusion: {pred_prob:.2%}")

# Lime Explanation
st.header("🔍 LIME Explanation")
explanation = lime_explain(final_input)
st.components.v1.html(explanation.as_html(), height=700)

