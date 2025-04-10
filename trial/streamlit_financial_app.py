import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load trained models and selector
model = joblib.load('voting_ensemble_model.pkl')
selector = joblib.load('feature_selector.pkl')
selected_features = joblib.load('selected_features.pkl')

# Define user input fields
user_inputs = {
    'age': (18, 100),
    'population_weight': (0.1, 3.0),
    'gender': ['Male', 'Female'],
    'education_level': ['None', 'Primary', 'Secondary', 'Tertiary'],
    'residence_type': ['Urban', 'Rural'],
    'marital_status': ['Single', 'Married', 'Divorced/Separated', 'Widowed'],
    'relationship_to_hh': ['Head', 'Spouse', 'Son/Daughter', 'Other relative'],
    'region': ['Nairobi', 'Central', 'Coast', 'Eastern', 'North Eastern', 'Nyanza', 'Rift Valley', 'Western', 'South Rift'],
    'mobile_money_registered': ['Yes', 'No'],
    'bank_account_everyday': ['Yes', 'No'],
    'savings_mobile_banking': ['Yes', 'No'],
    'savings_secret_place': ['Yes', 'No'],
    'loan_mobile_banking': ['Yes', 'No'],
    'loan_sacco': ['Yes', 'No'],
    'loan_group_chama': ['Yes', 'No'],
    'loan_family_friend': ['Yes', 'No'],
    'loan_goods_credit': ['Yes', 'No'],
    'loan_hire_purchase': ['Yes', 'No'],
    'insurance_nhif': ['Yes', 'No'],
    'debit_card': ['Yes', 'No'],
    'pension_nssf': ['Yes', 'No']
}

# Feature groups
feature_groups = {
    'formal_savings': ['bank_account_everyday'],
    'informal_savings': ['savings_secret_place'],
    'digital_financial': ['mobile_money_registered', 'savings_mobile_banking', 'loan_mobile_banking'],
    'formal_credit': ['loan_sacco'],
    'informal_credit': ['loan_group_chama', 'loan_family_friend', 'loan_goods_credit', 'loan_hire_purchase'],
    'insurance': ['insurance_nhif'],
    'pension': ['pension_nssf']
}

# Preprocessing and feature engineering
def preprocess_input(user_input):
    df = pd.DataFrame([user_input])

    # Binary encode yes/no
    for col in df.columns:
        if df[col].iloc[0] in ['Yes', 'No']:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Compute scores
    for group, features in feature_groups.items():
        df[f'{group}_score'] = df[features].sum(axis=1)

    df['formal_financial_score'] = df[[f for f in (
        feature_groups['formal_savings'] + feature_groups['formal_credit'] + feature_groups['insurance'] + feature_groups['pension']
    ) if f in df.columns]].sum(axis=1)

    df['informal_financial_score'] = df[[f for f in feature_groups['informal_credit'] + feature_groups['informal_savings'] if f in df.columns]].sum(axis=1)
    df['digital_financial_score'] = df[[f for f in feature_groups['digital_financial'] if f in df.columns]].sum(axis=1)

    df['financial_engagement_score'] = (
        1.5 * df['formal_financial_score'] +
        1.0 * df['informal_financial_score'] +
        2.0 * df['digital_financial_score']
    )

    df['product_category_diversity'] = df[[f"{g}_score" for g in feature_groups]].gt(0).sum(axis=1)
    df['risk_management_score'] = df[[f for f in feature_groups['insurance'] + feature_groups['pension'] if f in df.columns]].sum(axis=1)

    # Formal vs informal
    df['formal_informal_ratio'] = np.where(
        df['informal_financial_score'] == 0,
        np.where(df['formal_financial_score'] > 0, 'Formal_Only', 'None'),
        np.where(df['formal_financial_score'] == 0, 'Informal_Only', 'Mixed')
    )

    df['credit_to_savings_ratio'] = np.where(
        (df['formal_savings_score'] + df['informal_savings_score']) == 0,
        np.where((df['formal_credit_score'] + df['informal_credit_score']) > 0, 'Credit_Only', 'None'),
        np.where((df['formal_credit_score'] + df['informal_credit_score']) == 0, 'Savings_Only', 'Mixed')
    )

    # Encode categoricals
    categorical = ['gender', 'education_level', 'residence_type', 'marital_status', 'relationship_to_hh', 'region', 'formal_informal_ratio', 'credit_to_savings_ratio']
    df_encoded = pd.get_dummies(df, columns=categorical, drop_first=False)

    # Fill any missing expected features with 0
    for col in selected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[selected_features]
    df_final = df_encoded[selected_features]
    return df_final

# Streamlit UI
st.title("Kenya Financial Exclusion Predictor")
st.markdown("Enter household and financial details to assess financial exclusion risk.")

user_input = {}
for feature, options in user_inputs.items():
    if isinstance(options, tuple):
        user_input[feature] = st.slider(f"{feature.replace('_', ' ').capitalize()}", *options)
    else:
        user_input[feature] = st.selectbox(f"{feature.replace('_', ' ').capitalize()}", options)

if st.button("Predict Exclusion"):
    X_input = preprocess_input(user_input)
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

    if prediction == 1:
        st.error(f"Prediction: Financially Excluded \n Risk Score = {probability:.2%}")
    else:
        st.success(f"Prediction: Financially Included \n Confidence = {1 - probability:.2%}")