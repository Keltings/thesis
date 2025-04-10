import numpy as np

def make_prediction(input_data, model_choice):
    """
    Simulate prediction based on input data and model choice
    
    In a real implementation, this would:
    1. Load the trained model
    2. Process input features (one-hot encoding, etc.)
    3. Make prediction
    4. Generate explanations
    
    For this demo, we simulate predictions based on the input features
    """
    # Simplified prediction logic for demo purposes
    exclusion_probability = 0.0
    
    # Base probability from demographic factors
    if input_data["age"] < 25:
        exclusion_probability += 0.2
    if input_data["education_level"] in ["no_formal_education", "primary"]:
        exclusion_probability += 0.3
    if input_data["residence_type"] == "rural":
        exclusion_probability += 0.15
    if input_data["region"] in ["north_eastern", "coast", "eastern"]:
        exclusion_probability += 0.1
        
    # If using behavioral model, adjust probability
    if "Demographics Only" not in model_choice:
        if not input_data.get("mobile_money", False):
            exclusion_probability += 0.25
        if not input_data.get("bank_account", False):
            exclusion_probability += 0.15
        if not input_data.get("savings_account", False):
            exclusion_probability += 0.1
        if not input_data.get("insurance", False):
            exclusion_probability += 0.1
        if not input_data.get("pension", False):
            exclusion_probability += 0.05
    
    # Cap probability between 0 and 1
    exclusion_probability = min(max(exclusion_probability, 0.0), 0.98)
    
    # Add model-specific adjustments
    if "Decision Tree" in model_choice:
        # Decision trees tend to be more decisive
        if exclusion_probability > 0.5:
            exclusion_probability += 0.1
        else:
            exclusion_probability -= 0.1
    elif "Logistic Regression" in model_choice:
        # Make probability distribution more spread
        exclusion_probability = (exclusion_probability - 0.5) * 1.2 + 0.5
    
    # Final capping
    exclusion_probability = min(max(exclusion_probability, 0.01), 0.99)
    
    # Generate influential factors
    factors = generate_influential_factors(input_data, model_choice)
    
    return {
        "probability": exclusion_probability,
        "prediction": "Financially Excluded" if exclusion_probability > 0.5 else "Financially Included",
        "factors": factors
    }

def generate_influential_factors(input_data, model_choice):
    """Generate a list of factors that influenced the prediction"""
    factors = []
    
    # Add demographic factors
    if input_data["age"] < 25:
        factors.append({"feature": "Young Age (< 25)", "impact": 0.2, "direction": "positive"})
    if input_data["education_level"] in ["no_formal_education", "primary"]:
        factors.append({"feature": f"Lower Education ({input_data['education_level']})", "impact": 0.3, "direction": "positive"})
    if input_data["residence_type"] == "rural":
        factors.append({"feature": "Rural Residence", "impact": 0.15, "direction": "positive"})
    
    # Add behavioral factors if applicable
    if "Demographics Only" not in model_choice:
        if not input_data.get("mobile_money", False):
            factors.append({"feature": "No Mobile Money Account", "impact": 0.25, "direction": "positive"})
        if not input_data.get("bank_account", False):
            factors.append({"feature": "No Bank Account", "impact": 0.15, "direction": "positive"})
        if not input_data.get("savings_account", False):
            factors.append({"feature": "No Savings Account", "impact": 0.1, "direction": "positive"})
        if input_data.get("mobile_money", False):
            factors.append({"feature": "Has Mobile Money Account", "impact": 0.25, "direction": "negative"})
        if input_data.get("bank_account", False):
            factors.append({"feature": "Has Bank Account", "impact": 0.15, "direction": "negative"})
    
    # Sort by impact and take top 5
    factors = sorted(factors, key=lambda x: x["impact"], reverse=True)[:5]
    
    return factors

def get_lime_explanation(example_choice, model_choice):
    """
    Get LIME explanation for a specific example and model
    
    In a real implementation, this would:
    1. Load the trained model
    2. Process the example data
    3. Generate LIME explanation
    
    For this demo, we return pre-defined explanations
    """
    if "Example 1" in example_choice:
        return {
            "prediction": "Financially Excluded",
            "probability": 0.71,
            "feature_importances": [
                {"feature": "No Banking Account", "weight": 0.35, "contribution": 0.25},
                {"feature": "Rural Location", "weight": 0.20, "contribution": 0.14},
                {"feature": "Primary Education", "weight": 0.18, "contribution": 0.13},
                {"feature": "Has Mobile Money", "weight": -0.15, "contribution": -0.11},
                {"feature": "Mobile Banking Loan", "weight": -0.12, "contribution": -0.09},
                {"feature": "Age (42)", "weight": 0.08, "contribution": 0.06},
                {"feature": "Female", "weight": 0.07, "contribution": 0.05}
            ]
        }
    elif "Example 2" in example_choice:
        return {
            "prediction": "Financially Included",
            "probability": 0.12,
            "feature_importances": [
                {"feature": "Has Bank Accounts", "weight": -0.30, "contribution": -0.22},
                {"feature": "University Education", "weight": -0.25, "contribution": -0.18},
                {"feature": "Urban Location", "weight": -0.18, "contribution": -0.13},
                {"feature": "Has NHIF Insurance", "weight": -0.15, "contribution": -0.11},
                {"feature": "Has NSSF Pension", "weight": -0.12, "contribution": -0.09},
                {"feature": "Young Age (28)", "weight": 0.10, "contribution": 0.07},
                {"feature": "Nairobi Region", "weight": -0.08, "contribution": -0.06}
            ]
        }
    elif "Example 3" in example_choice:
        return {
            "prediction": "Financially Excluded",
            "probability": 0.95,
            "feature_importances": [
                {"feature": "No Financial Services", "weight": 0.40, "contribution": 0.38},
                {"feature": "No Formal Education", "weight": 0.25, "contribution": 0.24},
                {"feature": "Rural Location", "weight": 0.20, "contribution": 0.19},
                {"feature": "Elderly Age (67)", "weight": 0.15, "contribution": 0.14},
                {"feature": "No Mobile Money", "weight": 0.15, "contribution": 0.14},
                {"feature": "Western Region", "weight": 0.10, "contribution": 0.095},
                {"feature": "No Insurance", "weight": 0.10, "contribution": 0.095}
            ]
        }
    else:  # Example 4
        return {
            "prediction": "Financially Included",
            "probability": 0.38,
            "feature_importances": [
                {"feature": "Has Current Bank Account", "weight": -0.25, "contribution": -0.18},
                {"feature": "Urban Location", "weight": -0.20, "contribution": -0.14},
                {"feature": "Secondary Education", "weight": -0.18, "contribution": -0.13},
                {"feature": "Has Mobile Money", "weight": -0.15, "contribution": -0.11},
                {"feature": "Has NHIF Insurance", "weight": -0.12, "contribution": -0.09},
                {"feature": "No Savings Account", "weight": 0.15, "contribution": 0.11},
                {"feature": "No Pension", "weight": 0.12, "contribution": 0.09}
            ]
        }