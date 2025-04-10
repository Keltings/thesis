import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PATH, DEMOGRAPHIC_FEATURES
from utils.helper_functions import load_data
from utils.visualization import plot_target_distribution, plot_age_distribution, plot_behavioral_features, plot_correlation_heatmap

def show():
    """Display the Data Exploration page"""
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
        data_2016, error = load_data(DATA_PATH['2016'])
        if data_2016 is not None:
            st.success(f"Successfully loaded {DATA_PATH['2016']}")
        else:
            st.error(error)
    
    # Load 2021 data
    if uploaded_file_2021 is not None:
        data_2021 = pd.read_csv(uploaded_file_2021)
    elif use_default_2021:
        data_2021, error = load_data(DATA_PATH['2021'])
        if data_2021 is not None:
            st.success(f"Successfully loaded {DATA_PATH['2021']}")
        else:
            st.error(error)
    
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
                
                fig = plot_target_distribution(data_2021, class_counts_train, class_labels, "Target Distribution - 2021 Data")
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
                
                fig = plot_target_distribution(data_2016, class_counts_test, class_labels, "Target Distribution - 2016 Data")
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
        demographic_features = DEMOGRAPHIC_FEATURES
        
        behavioral_features = [col for col in data_2021.columns 
                              if col not in demographic_features + ["financially_excluded", "respondent_id"]]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Demographic Features:**")
            st.write(", ".join(demographic_features))
            
            # Age distribution
            if "age" in data_2021.columns:
                fig = plot_age_distribution(data_2021)
                st.pyplot(fig)
        
        with col2:
            st.write(f"**Behavioral Features ({len(behavioral_features)} features):**")
            if st.checkbox("Show all behavioral features"):
                st.write(", ".join(behavioral_features))
            
            # Sample behavioral feature distribution
            sample_features = behavioral_features[:5] if len(behavioral_features) > 5 else behavioral_features
            
            fig = plot_behavioral_features(data_2021, sample_features)
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
            
            # Plot correlation heatmap
            fig = plot_correlation_heatmap(
                corr_matrix.loc[top_features + ["financially_excluded"], top_features + ["financially_excluded"]],
                top_features
            )
            st.pyplot(fig)
            
            # Display features with strongest correlations
            st.markdown('<div class="section-header">Features with Strongest Correlation to Financial Exclusion</div>', unsafe_allow_html=True)
            
            top_corr_df = pd.DataFrame({
                'Feature': target_corr.head(10).index,
                'Correlation': target_corr.head(10).values
            })
            st.table(top_corr_df)
    
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
            st.pyplot(fig)