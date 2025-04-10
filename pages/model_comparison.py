import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import MODELS, F1_S1_DEMO, F1_S1_FULL, F1_S2_DEMO, F1_S2_FULL, F1_S3_DEMO, F1_S3_FULL
from config import AUC_S1_DEMO, AUC_S1_FULL, AUC_S2_DEMO, AUC_S2_FULL, AUC_S3_DEMO, AUC_S3_FULL
from utils.helper_functions import get_image_download_link

def show():
    """Display the Model Comparison page"""
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
    
    # Display metric selection
    metric_choice = st.radio("Choose metric to display:", ["F1 Score", "ROC AUC"])
    
    # Feature set selection
    feature_set = st.radio("Choose feature set:", ["Demographics Only", "Demographics + Behavior"])
    
    # Prepare data based on selection
    data = prepare_chart_data(metric_choice, feature_set)
    
    # Create plot
    fig = create_comparison_chart(data, metric_choice, feature_set)
    st.pyplot(fig)
    
    # Download button for plot
    st.markdown(get_image_download_link(fig, filename=f"{data['title'].replace(' ', '_')}.png", 
                                     text="Download this plot"), unsafe_allow_html=True)
    
    # Performance metrics table
    st.markdown('<div class="section-header">Detailed Performance Metrics</div>', unsafe_allow_html=True)
    
    scenario = st.selectbox("Select scenario:", ["No Weights", "Weighted", "SMOTE"])
    
    # Create and display metrics dataframe
    df_metrics = create_metrics_dataframe(scenario, feature_set)
    st.dataframe(df_metrics.set_index("Model"))
    
    # Highlight best model
    display_best_model(df_metrics)
    
    # Key insights based on scenario
    display_insights(feature_set)

def prepare_chart_data(metric_choice, feature_set):
    """Prepare data for chart based on user selection"""
    if metric_choice == "F1 Score":
        if feature_set == "Demographics Only":
            data = {
                "scenario1_data": F1_S1_DEMO,
                "scenario2_data": F1_S2_DEMO,
                "scenario3_data": F1_S3_DEMO,
                "title": "F1 Score Comparison (Demographics Only)",
                "y_label": "F1 Score"
            }
        else:  # Demographics + Behavior
            data = {
                "scenario1_data": F1_S1_FULL,
                "scenario2_data": F1_S2_FULL,
                "scenario3_data": F1_S3_FULL,
                "title": "F1 Score Comparison (Demographics + Behavior)",
                "y_label": "F1 Score"
            }
    else:  # ROC AUC
        if feature_set == "Demographics Only":
            data = {
                "scenario1_data": AUC_S1_DEMO,
                "scenario2_data": AUC_S2_DEMO,
                "scenario3_data": AUC_S3_DEMO,
                "title": "ROC AUC Comparison (Demographics Only)",
                "y_label": "ROC AUC"
            }
        else:  # Demographics + Behavior
            data = {
                "scenario1_data": AUC_S1_FULL,
                "scenario2_data": AUC_S2_FULL,
                "scenario3_data": AUC_S3_FULL,
                "title": "ROC AUC Comparison (Demographics + Behavior)",
                "y_label": "ROC AUC"
            }
    return data

def create_comparison_chart(data, metric_choice, feature_set):
    """Create comparison chart based on provided data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODELS))
    width = 0.25
    
    bars1 = ax.bar(x - width, data["scenario1_data"], width, label='No Weights', color='#4C72B0')
    bars2 = ax.bar(x, data["scenario2_data"], width, label='Weighted', color='#55A868')
    bars3 = ax.bar(x + width, data["scenario3_data"], width, label='SMOTE', color='#FF9500')
    
    # Add value annotations
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                      xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    ax.set_ylabel(data["y_label"])
    ax.set_title(data["title"])
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig

def create_metrics_dataframe(scenario, feature_set):
    """Create a dataframe of metrics based on scenario and feature set"""
    if scenario == "No Weights":
        if feature_set == "Demographics Only":
            df_metrics = pd.DataFrame({
                "Model": MODELS,
                "F1 Score": F1_S1_DEMO,
                "ROC AUC": AUC_S1_DEMO,
                "Precision": [0.4000, 1.0000, 0.0000, 0.0000],
                "Accuracy": [0.9141, 0.9148, 0.9146, 0.9146]
            })
        else:  # Demographics + Behavior
            df_metrics = pd.DataFrame({
                "Model": MODELS,
                "F1 Score": F1_S1_FULL,
                "ROC AUC": AUC_S1_FULL,
                "Precision": [0.5485, 0.4570, 0.0000, 0.4841],
                "Accuracy": [0.9297, 0.8985, 0.9146, 0.9090]
            })
    elif scenario == "Weighted":
        if feature_set == "Demographics Only":
            df_metrics = pd.DataFrame({
                "Model": MODELS,
                "F1 Score": F1_S2_DEMO,
                "ROC AUC": AUC_S2_DEMO,
                "Precision": [0.2500, 0.2464, 0.2303, 0.4412],
                "Accuracy": [0.9095, 0.9039, 0.9032, 0.9114]
            })
        else:  # Demographics + Behavior
            df_metrics = pd.DataFrame({
                "Model": MODELS,
                "F1 Score": F1_S2_FULL,
                "ROC AUC": AUC_S2_FULL,
                "Precision": [0.5586, 0.5631, 0.5185, 0.4979],
                "Accuracy": [0.9327, 0.9336, 0.9208, 0.9147]
            })
    else:  # SMOTE
        if feature_set == "Demographics Only":
            df_metrics = pd.DataFrame({
                "Model": MODELS,
                "F1 Score": F1_S3_DEMO,
                "ROC AUC": AUC_S3_DEMO,
                "Precision": [0.2147, 0.1895, 0.1925, 0.2698],
                "Accuracy": [0.7744, 0.7432, 0.7485, 0.8018]
            })
        else:  # Demographics + Behavior
            df_metrics = pd.DataFrame({
                "Model": MODELS,
                "F1 Score": F1_S3_FULL,
                "ROC AUC": AUC_S3_FULL,
                "Precision": [0.5298, 0.5637, 0.5134, 0.5125],
                "Accuracy": [0.9269, 0.9332, 0.9183, 0.9177]
            })
    
    return df_metrics

def display_best_model(df_metrics):
    """Display the best model from the metrics dataframe"""
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

def display_insights(feature_set):
    """Display insights based on feature set"""
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