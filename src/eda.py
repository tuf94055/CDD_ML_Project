# src/eda.py
# EDA Module
# Mathew Kuruvilla

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_histograms(df, feature_list):
    """
    Plot histograms for given features.
    Args:
        df (pd.DataFrame): Dataframe containing features.
        feature_list (list): List of feature column names to plot.
    """
    for feature in feature_list:
        plt.figure(figsize=(6,4))
        sns.histplot(df[feature], bins=30, kde=True)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

def plot_correlation_matrix(df, feature_list):
    """
    Plot a heatmap showing the correlation matrix of selected features.
    Args:
        df (pd.DataFrame): Dataframe containing features.
        feature_list (list): List of feature column names to calculate correlation.
    """
    corr_matrix = df[feature_list].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_feature_vs_target(df, feature, target):
    """
    Scatter plot of a feature versus target value.
    Args:
        df (pd.DataFrame): Dataframe.
        feature (str): Feature column name.
        target (str): Target column name (e.g., 'standard_value').
    """
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[feature], y=df[target])
    plt.title(f"{feature} vs {target}")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.tight_layout()
    plt.show()
