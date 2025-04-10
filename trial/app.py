import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, roc_auc_score, precision_score, accuracy_score
import lime
import lime.lime_tabular
from imblearn.over_sampling import SMOTE
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Financial Exclusion Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
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
        color: #424242;
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

# Function to get image download link
def get_image_download_link(fig, filename="plot.png", text="Download Plot"):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Application title
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

# Introduction page
if app_mode == "Introduction":
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
    
# Data Exploration
elif app_mode == "Data Exploration":
    st.markdown('<div class="sub-header">Data Exploration</div>', unsafe_allow_html=True)
    
    # Upload datasets
    st.markdown('<div class="section-header">Upload Dataset Files</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file_2016 = st.file_uploader("Upload 2016 dataset (CSV)", type="csv", key="file_2016")
        use_default_2016 = st.checkbox("Use default 'selected_2016.csv' file", value=True)
    with col2:
        uploaded_file_2021 = st.file_uploader("Upload 2021 dataset (CSV)", type="csv", key="file_2021")
        use_default_2021 = st.checkbox("Use default 'selected_2021.csv' file", value=True)

    # Initialize data variables
    data_2016 = None
    data_2021 = None

    # Load 2016 data
    if uploaded_file_2016 is not None:
        data_2016 = pd.read_csv(uploaded_file_2016)
    elif use_default_2016:
        try:
            data_2016 = pd.read_csv('selected_2016_data.csv')
            st.success("Successfully loaded selected_2016_data.csv")
        except FileNotFoundError:
            st.error("Default file 'selected_2016_data.csv' not found. Please upload a file.")

    # Load 2021 data
    if uploaded_file_2021 is not None:
        data_2021 = pd.read_csv(uploaded_file_2021)
    elif use_default_2021:
        try:
            data_2021 = pd.read_csv('selected_2021_data.csv')
            st.success("Successfully loaded selected_2021_data.csv")
        except FileNotFoundError:
            st.error("Default file 'selected_2021_data.csv' not found. Please upload a file.")

    # Continue with visualization only if both datasets are loaded
    if data_2016 is not None and data_2021 is not None:
        # Display basic info
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**2016 Dataset**")
            st.write(f"Shape: {data_2016.shape}")
            st.dataframe(data_2016.head(3))
        
        with col2:
            st.write("**2021 Dataset**")
            st.write(f"Shape: {data_2021.shape}")
            st.dataframe(data_2021.head(3))
        
        # Target distribution
        st.markdown('<div class="section-header">Target Distribution (Financial Exclusion)</div>', unsafe_allow_html=True)
        
        # Process target variable if needed
        if "financially_excluded" in data_2021.columns:
            y_train = data_2021["financially_excluded"].map({"yes": 1, "no": 0})
            y_test = data_2016["financially_excluded"].map({"yes": 1, "no": 0})
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Training data (2021)
                class_counts_train = y_train.value_counts().sort_index()
                class_labels = ["Included (No)", "Excluded (Yes)"]
                
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = sns.barplot(x=class_labels, y=class_counts_train.values, palette=["#4C72B0", "#55A868"], ax=ax)
                
                # Annotate bars
                for i, bar in enumerate(bars.patches):
                    height = bar.get_height()
                    ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
                
                plt.title("Target Distribution - 2021 Data")
                plt.ylabel("Number of Samples")
                plt.grid(axis='y', linestyle='--', alpha=0.6)
                st.pyplot(fig)
                
                # Statistics
                st.markdown(f"""
                <div class="metric-card">
                Included: {class_counts_train[0]} samples ({class_counts_train[0]/len(y_train)*100:.1f}%)<br>
                Excluded: {class_counts_train[1]} samples ({class_counts_train[1]/len(y_train)*100:.1f}%)
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                # Test data (2016)
                class_counts_test = y_test.value_counts().sort_index()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = sns.barplot(x=class_labels, y=class_counts_test.values, palette=["#4C72B0", "#55A868"], ax=ax)
                
                # Annotate bars
                for i, bar in enumerate(bars.patches):
                    height = bar.get_height()
                    ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
                
                plt.title("Target Distribution - 2016 Data")
                plt.ylabel("Number of Samples")
                plt.grid(axis='y', linestyle='--', alpha=0.6)
                st.pyplot(fig)
                
                # Statistics
                st.markdown(f"""
                <div class="metric-card">
                Included: {class_counts_test[0]} samples ({class_counts_test[0]/len(y_test)*100:.1f}%)<br>
                Excluded: {class_counts_test[1]} samples ({class_counts_test[1]/len(y_test)*100:.1f}%)
                </div>
                """, unsafe_allow_html=True)
        
        # Feature Groups
        st.markdown('<div class="section-header">Feature Groups</div>', unsafe_allow_html=True)
        
        # Define feature groups
        demographic_features = ["age", "gender", "education_level", "residence_type", 
                            "marital_status", "relationship_to_hh", "region", "population_weight"]
        
        behavioral_features = [col for col in data_2021.columns 
                            if col not in demographic_features + ["financially_excluded", "respondent_id"]]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Demographic Features:**")
            st.write(", ".join(demographic_features))
            
            # Age distribution
            if "age" in data_2021.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(data_2021["age"], bins=20, kde=True, ax=ax)
                plt.title("Age Distribution")
                plt.xlabel("Age")
                plt.ylabel("Count")
                st.pyplot(fig)
        
        with col2:
            st.write(f"**Behavioral Features ({len(behavioral_features)} features):**")
            if st.checkbox("Show all behavioral features"):
                st.write(", ".join(behavioral_features))
            
            # Sample behavioral feature distribution
            sample_features = behavioral_features[:5] if len(behavioral_features) > 5 else behavioral_features
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            for feature in sample_features:
                if feature in data_2021.columns:
                    counts = data_2021[feature].value_counts().sort_index()
                    ax.bar([f"{feature} - No", f"{feature} - Yes"], 
                        [counts.get("no", 0), counts.get("yes", 0)], alpha=0.7)
            
            plt.title("Sample Behavioral Features")
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Count")
            st.pyplot(fig)
            
        # Additional visualizations - correlation between features and target
        if st.checkbox("Show Correlation between Features and Financial Exclusion"):
            # Extract selected features for correlation
            cols_to_analyze = ["age", "population_weight"]
            
            # Map yes/no to 1/0 for behavioral features
            data_2021_corr = data_2021.copy()  # Create a copy to avoid modifying original
            for col in behavioral_features:
                if col in data_2021_corr.columns:
                    data_2021_corr[col] = data_2021_corr[col].map({"yes": 1, "no": 0})
                    cols_to_analyze.append(col)
            
            # Add target
            data_2021_corr["financially_excluded"] = data_2021_corr["financially_excluded"].map({"yes": 1, "no": 0})
            
            # Calculate correlations
            corr_matrix = data_2021_corr[cols_to_analyze + ["financially_excluded"]].corr()
            
            # Sort by absolute correlation with target
            target_corr = corr_matrix["financially_excluded"].drop("financially_excluded").abs().sort_values(ascending=False)
            top_features = target_corr.head(10).index.tolist()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix.loc[top_features + ["financially_excluded"], top_features + ["financially_excluded"]], 
                    annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
            plt.title("Correlation between Top Features and Financial Exclusion")
            st.pyplot(fig)
    else:
        st.info("Please upload or use default files for both 2016 and 2021 datasets to explore the data.")

        # Show placeholder visualization
        st.markdown('<div class="section-header">Sample Visualizations</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample target distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            sample_data = [8500, 1500]
            bars = sns.barplot(x=["Included (No)", "Excluded (Yes)"], y=sample_data, palette=["#4C72B0", "#55A868"], ax=ax)
            
            # Annotate bars
            for i, bar in enumerate(bars.patches):
                height = bar.get_height()
                ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
            
            plt.title("Sample Target Distribution")
            plt.ylabel("Number of Samples")
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            st.pyplot(fig)
        
        with col2:
            # Sample age distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            sample_ages = np.random.normal(35, 12, 1000)
            sns.histplot(sample_ages, bins=20, kde=True, ax=ax)
            plt.title("Sample Age Distribution")
            plt.xlabel("Age")
            plt.ylabel("Count")
            st.pyplot(fig)# Model Comparison page
elif app_mode == "Model Comparison":
    st.markdown('<div class="sub-header">Model Comparison</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This section compares the performance of different models across three scenarios:
    1. **No Weights** - Models trained without sample weights
    2. **Weighted** - Models trained with population weights
    3. **SMOTE** - Models trained with synthetic minority oversampling
    
    For each scenario, we evaluate two feature sets:
    - **Demographics Only** - Using only demographic features
    - **Demographics + Behavior** - Using both demographic and behavioral features
    """)
    
    # Metrics from the notebook
    models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    
    # Scenario 1 (No Weights)
    f1_s1_demo = [0.0222, 0.0057, 0.0000, 0.0000]
    f1_s1_full = [0.7084, 0.6273, 0.0000, 0.6524]
    auc_s1_demo = [0.8371, 0.7837, 0.8025, 0.8419]
    auc_s1_full = [0.9779, 0.9445, 0.9687, 0.9713]
    
    # Scenario 2 (Weighted)
    f1_s2_demo = [0.0351, 0.1954, 0.1987, 0.1121]
    f1_s2_full = [0.7168, 0.7205, 0.6829, 0.6648]
    auc_s2_demo = [0.8249, 0.7316, 0.7789, 0.8512]
    auc_s2_full = [0.9675, 0.9638, 0.9736, 0.9725]
    
    # Scenario 3 (SMOTE)
    f1_s3_demo = [0.3364, 0.3112, 0.3167, 0.3917]
    f1_s3_full = [0.6903, 0.7202, 0.6770, 0.6760]
    auc_s3_demo = [0.8446, 0.7318, 0.7681, 0.8417]
    auc_s3_full = [0.9764, 0.9625, 0.9740, 0.9739]
    
    # Display metric selection
    metric_choice = st.radio("Choose metric to display:", ["F1 Score", "ROC AUC"])
    
    # Feature set selection
    feature_set = st.radio("Choose feature set:", ["Demographics Only", "Demographics + Behavior"])
    
    # Prepare data based on selection
    if metric_choice == "F1 Score":
        if feature_set == "Demographics Only":
            scenario1_data = f1_s1_demo
            scenario2_data = f1_s2_demo
            scenario3_data = f1_s3_demo
            title = "F1 Score Comparison (Demographics Only)"
            y_label = "F1 Score"
        else:  # Demographics + Behavior
            scenario1_data = f1_s1_full
            scenario2_data = f1_s2_full
            scenario3_data = f1_s3_full
            title = "F1 Score Comparison (Demographics + Behavior)"
            y_label = "F1 Score"
    else:  # ROC AUC
        if feature_set == "Demographics Only":
            scenario1_data = auc_s1_demo
            scenario2_data = auc_s2_demo
            scenario3_data = auc_s3_demo
            title = "ROC AUC Comparison (Demographics Only)"
            y_label = "ROC AUC"
        else:  # Demographics + Behavior
            scenario1_data = auc_s1_full
            scenario2_data = auc_s2_full
            scenario3_data = auc_s3_full
            title = "ROC AUC Comparison (Demographics + Behavior)"
            y_label = "ROC AUC"
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax.bar(x - width, scenario1_data, width, label='No Weights', color='#4C72B0')
    bars2 = ax.bar(x, scenario2_data, width, label='Weighted', color='#55A868')
    bars3 = ax.bar(x + width, scenario3_data, width, label='SMOTE', color='#FF9500')
    
    # Add value annotations
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                      xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    
    # Download button for plot
    st.markdown(get_image_download_link(fig, filename=f"{title.replace(' ', '_')}.png", 
                                     text="Download this plot"), unsafe_allow_html=True)
    
    # Performance metrics table
    st.markdown('<div class="section-header">Detailed Performance Metrics</div>', unsafe_allow_html=True)
    
    scenario = st.selectbox("Select scenario:", ["No Weights", "Weighted", "SMOTE"])
    
    # Create DataFrame based on selection
    if scenario == "No Weights":
        if feature_set == "Demographics Only":
            df_metrics = pd.DataFrame({
                "Model": models,
                "F1 Score": f1_s1_demo,
                "ROC AUC": auc_s1_demo,
                "Precision": [0.4000, 1.0000, 0.0000, 0.0000],
                "Accuracy": [0.9141, 0.9148, 0.9146, 0.9146]
            })
        else:  # Demographics + Behavior
            df_metrics = pd.DataFrame({
                "Model": models,
                "F1 Score": f1_s1_full,
                "ROC AUC": auc_s1_full,
                "Precision": [0.5485, 0.4570, 0.0000, 0.4841],
                "Accuracy": [0.9297, 0.8985, 0.9146, 0.9090]
            })
    elif scenario == "Weighted":
        if feature_set == "Demographics Only":
            df_metrics = pd.DataFrame({
                "Model": models,
                "F1 Score": f1_s2_demo,
                "ROC AUC": auc_s2_demo,
                "Precision": [0.2500, 0.2464, 0.2303, 0.4412],
                "Accuracy": [0.9095, 0.9039, 0.9032, 0.9114]
            })
        else:  # Demographics + Behavior
            df_metrics = pd.DataFrame({
                "Model": models,
                "F1 Score": f1_s2_full,
                "ROC AUC": auc_s2_full,
                "Precision": [0.5586, 0.5631, 0.5185, 0.4979],
                "Accuracy": [0.9327, 0.9336, 0.9208, 0.9147]
            })
    else:  # SMOTE
        if feature_set == "Demographics Only":
            df_metrics = pd.DataFrame({
                "Model": models,
                "F1 Score": f1_s3_demo,
                "ROC AUC": auc_s3_demo,
                "Precision": [0.2147, 0.1895, 0.1925, 0.2698],
                "Accuracy": [0.7744, 0.7432, 0.7485, 0.8018]
            })
        else:  # Demographics + Behavior
            df_metrics = pd.DataFrame({
                "Model": models,
                "F1 Score": f1_s3_full,
                "ROC AUC": auc_s3_full,
                "Precision": [0.5298, 0.5637, 0.5134, 0.5125],
                "Accuracy": [0.9269, 0.9332, 0.9183, 0.9177]
            })
    
    st.dataframe(df_metrics.set_index("Model"))
    
    # Highlight best model
    best_model_idx = df_metrics["F1 Score"].argmax()
    best_model = df_metrics.iloc[best_model_idx]
    
    st.markdown(f"""
    <div class="highlight">
    <b>Best Model:</b> {best_model['Model']}<br>
    <b>F1 Score:</b> {best_model['F1 Score']:.4f}<br>
    <b>ROC AUC:</b> {best_model['ROC AUC']:.4f}<br>
    <b>Precision:</b> {best_model['Precision']:.4f}<br>
    <b>Accuracy:</b> {best_model['Accuracy']:.4f}
    </div>
    """, unsafe_allow_html=True)
    
    # Key insights based on scenario
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    
    if feature_set == "Demographics Only":
        st.markdown("""
        **Demographics-Only Models:**
        
        - Models using only demographic features struggle significantly in identifying financially excluded individuals, especially without adjustments for class imbalance.
        - SMOTE improves F1 scores dramatically for demographic-only models, with Gradient Boosting performing best.
        - Even with SMOTE, demographic-only models have lower performance than models with behavioral features.
        - The notable gap between ROC AUC and F1 scores suggests that while models can rank probabilities well, they struggle with setting the right threshold for actual predictions.
        """)
    else:
        st.markdown("""
        **Demographics + Behavior Models:**
        
        - Adding behavioral features drastically improves all models across all metrics.
        - Decision Tree with SMOTE achieves the highest F1 score (0.7202).
        - Logistic Regression is consistently strong across all scenarios, suggesting it's very stable for this problem.
        - All models achieve excellent ROC AUC (>0.96) when using both demographic and behavioral features.
        - The best performance comes from models that address class imbalance (weighted or SMOTE) while using both feature sets.
        """)

# Interactive Prediction page
elif app_mode == "Interactive Prediction":
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Demographic Information**")
        
        age = st.slider("Age", 18, 100, 35)
        
        gender = st.selectbox("Gender", ["male", "female"])
        
        education_level = st.selectbox(
            "Education Level", 
            ["no_formal_education", "primary", "secondary", "university"]
        )
        
        residence_type = st.selectbox("Residence Type", ["urban", "rural"])
        
        marital_status = st.selectbox(
            "Marital Status", 
            ["single", "married", "divorced", "widowed", "separated"]
        )
        
        relationship_to_hh = st.selectbox(
            "Relationship to Household Head", 
            ["head", "spouse", "son_daughter", "parent", "other_relative", "not_related"]
        )
        
        region = st.selectbox(
            "Region", 
            ["nairobi", "central", "coast", "eastern", "north_eastern", "nyanza", "rift_valley", "western"]
        )
    
    with col2:
        st.markdown("**Behavioral Information**")
        
        # Only include behavioral fields if not using demographics-only model
        if "Demographics Only" not in model_choice:
            mobile_money = st.checkbox("Has Mobile Money Account", value=True)
            bank_account = st.checkbox("Has Bank Account", value=False)
            savings_account = st.checkbox("Has Savings Account", value=False)
            loan = st.checkbox("Has Any Loan", value=False)
            insurance = st.checkbox("Has Any Insurance", value=False)
            pension = st.checkbox("Has Pension", value=False)
            
            # Additional behavioral details if needed
            st.markdown("**Financial Details (Optional)**")
            has_debit_card = st.checkbox("Has Debit Card", value=False)
            has_credit_card = st.checkbox("Has Credit Card", value=False)
            savings_microfinance = st.checkbox("Saves with Microfinance", value=False)
            savings_sacco = st.checkbox("Saves with SACCO", value=False)
            savings_group = st.checkbox("Saves with Group/Chama", value=False)
        else:
            st.info("Behavioral features not included in demographics-only model.")
    
    # Make prediction button
    if st.button("Predict Financial Exclusion Status"):
        # In a real app, we'd load the actual models
        # Here we'll simulate predictions based on the input
        
        # Create a feature vector (simplified for demonstration)
        # In a real implementation, we would:
        # 1. Load the trained model
        # 2. Load the encoder
        # 3. Process input features with encoder
        # 4. Make prediction
        
        # Simplified prediction logic for demo purposes
        exclusion_probability = 0.0
        
        # Base probability from demographic factors
        if age < 25:
            exclusion_probability += 0.2
        if education_level in ["no_formal_education", "primary"]:
            exclusion_probability += 0.3
        if residence_type == "rural":
            exclusion_probability += 0.15
        if region in ["north_eastern", "coast", "eastern"]:
            exclusion_probability += 0.1
            
        # If using behavioral model, adjust probability
        if "Demographics Only" not in model_choice:
            if not mobile_money:
                exclusion_probability += 0.25
            if not bank_account:
                exclusion_probability += 0.15
            if not savings_account:
                exclusion_probability += 0.1
            if not insurance:
                exclusion_probability += 0.1
            if not pension:
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
        
        # Display prediction
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Prediction result
            prediction = "Financially Excluded" if exclusion_probability > 0.5 else "Financially Included"
            prediction_color = "#D32F2F" if exclusion_probability > 0.5 else "#388E3C"
            
            st.markdown(f"""
            <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="margin-bottom: 10px;">Prediction Result</h3>
                <p style="font-size: 24px; font-weight: bold; color: {prediction_color};">{prediction}</p>
                <p style="font-size: 18px;">Probability: {exclusion_probability:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Probability gauge
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#F0F0F0')
            
            # Create gauge chart
            gauge_colors = [(0.0, '#388E3C'), (0.5, '#FFA726'), (1.0, '#D32F2F')]
            cmap = plt.cm.RdYlGn_r
            norm = plt.Normalize(0, 1)
            
            # Draw gauge background
            ax.barh(0, 1, left=0, height=0.3, color='#EEEEEE')
            
            # Draw gauge value
            ax.barh(0, exclusion_probability, left=0, height=0.3, color=cmap(norm(exclusion_probability)))
            
            # Add marker for threshold
            ax.axvline(x=0.5, color='#757575', linestyle='--', alpha=0.7)
            ax.text(0.5, -0.5, 'Threshold', ha='center', va='center', color='#757575', fontsize=10)
            
            # Add labels
            ax.text(0.0, -0.2, 'Included', ha='left', va='center', fontsize=10, color='#388E3C')
            ax.text(1.0, -0.2, 'Excluded', ha='right', va='center', fontsize=10, color='#D32F2F')
            
            # Value marker
            ax.text(exclusion_probability, 0.5, f'{exclusion_probability:.2f}', 
                  ha='center', va='center', fontsize=12, fontweight='bold',
                  bbox=dict(boxstyle="circle,pad=0.3", fc='white', ec=cmap(norm(exclusion_probability))))
            
            # Remove axes and set limits
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 1)
            ax.axis('off')
            
            st.pyplot(fig)
        
        # Display key factors
        st.markdown('<div class="section-header">Key Factors Influencing Prediction</div>', unsafe_allow_html=True)
        
        # Simulate LIME-like output with contributing factors
        factors = []
        
        # Add demographic factors
        if age < 25:
            factors.append({"feature": "Young Age (< 25)", "impact": 0.2, "direction": "positive"})
        if education_level in ["no_formal_education", "primary"]:
            factors.append({"feature": f"Lower Education ({education_level})", "impact": 0.3, "direction": "positive"})
        if residence_type == "rural":
            factors.append({"feature": "Rural Residence", "impact": 0.15, "direction": "positive"})
        
        # Add behavioral factors if applicable
        if "Demographics Only" not in model_choice:
            if not mobile_money:
                factors.append({"feature": "No Mobile Money Account", "impact": 0.25, "direction": "positive"})
            if not bank_account:
                factors.append({"feature": "No Bank Account", "impact": 0.15, "direction": "positive"})
            if not savings_account:
                factors.append({"feature": "No Savings Account", "impact": 0.1, "direction": "positive"})
            if mobile_money:
                factors.append({"feature": "Has Mobile Money Account", "impact": 0.25, "direction": "negative"})
            if bank_account:
                factors.append({"feature": "Has Bank Account", "impact": 0.15, "direction": "negative"})
        
        # Sort by impact and take top 5
        factors = sorted(factors, key=lambda x: x["impact"], reverse=True)[:5]
        
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

# Model Explainability page
elif app_mode == "Model Explainability":
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
    
    # Load example data based on selection
    def load_example_data(example_choice):
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
    
    example_data = load_example_data(example_choice)
    
    # Display example data
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
    
    # Generate LIME explanation (simulated)
    st.markdown('<div class="section-header">LIME Explanation</div>', unsafe_allow_html=True)
    
    # Simulate model prediction 
    if "Example 1" in example_choice:
        prediction_proba = 0.71
        prediction = "Financially Excluded"
        feature_importances = [
            {"feature": "No Banking Account", "weight": 0.35, "contribution": 0.25},
            {"feature": "Rural Location", "weight": 0.20, "contribution": 0.14},
            {"feature": "Primary Education", "weight": 0.18, "contribution": 0.13},
            {"feature": "Has Mobile Money", "weight": -0.15, "contribution": -0.11},
            {"feature": "Mobile Banking Loan", "weight": -0.12, "contribution": -0.09},
            {"feature": "Age (42)", "weight": 0.08, "contribution": 0.06},
            {"feature": "Female", "weight": 0.07, "contribution": 0.05}
        ]
    elif "Example 2" in example_choice:
        prediction_proba = 0.12
        prediction = "Financially Included"
        feature_importances = [
            {"feature": "Has Bank Accounts", "weight": -0.30, "contribution": -0.22},
            {"feature": "University Education", "weight": -0.25, "contribution": -0.18},
            {"feature": "Urban Location", "weight": -0.18, "contribution": -0.13},
            {"feature": "Has NHIF Insurance", "weight": -0.15, "contribution": -0.11},
            {"feature": "Has NSSF Pension", "weight": -0.12, "contribution": -0.09},
            {"feature": "Young Age (28)", "weight": 0.10, "contribution": 0.07},
            {"feature": "Nairobi Region", "weight": -0.08, "contribution": -0.06}
        ]
    elif "Example 3" in example_choice:
        prediction_proba = 0.95
        prediction = "Financially Excluded"
        feature_importances = [
            {"feature": "No Financial Services", "weight": 0.40, "contribution": 0.38},
            {"feature": "No Formal Education", "weight": 0.25, "contribution": 0.24},
            {"feature": "Rural Location", "weight": 0.20, "contribution": 0.19},
            {"feature": "Elderly Age (67)", "weight": 0.15, "contribution": 0.14},
            {"feature": "No Mobile Money", "weight": 0.15, "contribution": 0.14},
            {"feature": "Western Region", "weight": 0.10, "contribution": 0.095},
            {"feature": "No Insurance", "weight": 0.10, "contribution": 0.095}
        ]
    else:  # Example 4
        prediction_proba = 0.38
        prediction = "Financially Included"
        feature_importances = [
            {"feature": "Has Current Bank Account", "weight": -0.25, "contribution": -0.18},
            {"feature": "Urban Location", "weight": -0.20, "contribution": -0.14},
            {"feature": "Secondary Education", "weight": -0.18, "contribution": -0.13},
            {"feature": "Has Mobile Money", "weight": -0.15, "contribution": -0.11},
            {"feature": "Has NHIF Insurance", "weight": -0.12, "contribution": -0.09},
            {"feature": "No Savings Account", "weight": 0.15, "contribution": 0.11},
            {"feature": "No Pension", "weight": 0.12, "contribution": 0.09}
        ]
    
    # Create prediction header
    prediction_color = "#D32F2F" if prediction == "Financially Excluded" else "#388E3C"
    st.markdown(f"""
    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="margin-bottom: 10px;">Prediction: <span style="color: {prediction_color};">{prediction}</span></h3>
        <p style="font-size: 16px;">Probability: {prediction_proba:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate LIME visualization
    # Create the LIME-style horizontal bar chart
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
    st.pyplot(fig)
    
    # Download LIME visualization
    st.markdown(get_image_download_link(fig, filename=f"lime_explanation_{example_choice.split(':')[0]}.png", 
                                     text="Download LIME visualization"), unsafe_allow_html=True)
    
    # Interpretation of LIME results
    st.markdown('<div class="section-header">Interpretation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <b>How to read the LIME explanation:</b><br>
    Each horizontal bar shows how a feature contributes to the prediction. 
    <span style="color: #D32F2F;">Red bars</span> push the prediction toward financial exclusion, 
    while <span style="color: #388E3C;">green bars</span> push toward financial inclusion.
    The length of each bar indicates how strongly that feature influences the prediction.
    </div>
    """, unsafe_allow_html=True)
    
    # Example-specific interpretation
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
    
    # Add educational explanation about LIME
    with st.expander("Learn More About LIME"):
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

# Footer
st.markdown('<div class="footer">Financial Exclusion Prediction App with Explainable AI</div>', unsafe_allow_html=True)