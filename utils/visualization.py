import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_target_distribution(data, class_counts, class_labels, title):
    """Plot target distribution bar chart"""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = sns.barplot(x=class_labels, y=class_counts.values, palette=["#4C72B0", "#55A868"], ax=ax)
    
    # Annotate bars
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 5), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
    
    plt.title(title)
    plt.ylabel("Number of Samples")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    return fig

def plot_age_distribution(data):
    """Plot age distribution with histogram and kde"""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data["age"], bins=20, kde=True, ax=ax)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    
    return fig

def plot_behavioral_features(data, features):
    """Plot distribution of behavioral features"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for feature in features:
        if feature in data.columns:
            counts = data[feature].value_counts().sort_index()
            ax.bar([f"{feature} - No", f"{feature} - Yes"], 
                  [counts.get("no", 0), counts.get("yes", 0)], alpha=0.7)
    
    plt.title("Sample Behavioral Features")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Count")
    
    return fig

def plot_correlation_heatmap(corr_matrix, top_features):
    """Plot correlation heatmap for top features"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    plt.title("Correlation between Top Features and Financial Exclusion")
    
    return fig

def create_lime_explanation_chart(features, weights):
    """Create LIME explanation chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create colors based on weight sign
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