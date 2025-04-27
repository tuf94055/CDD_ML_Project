# src/dataset_preparation.py
# Dataset Preparation Module
# Mathew Kuruvilla

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def log_transform_target(df, target_column='standard_value'):
    """
    Apply log10 transformation to the target column.
    
    Args:
        df (pd.DataFrame): Input dataframe containing target column.
        target_column (str): Name of the column to transform (default is 'standard_value').
    
    Returns:
        pd.Series: Log-transformed target values.
    """
    return np.log10(df[target_column])

def prepare_features_and_target(df, feature_columns, target_column='standard_value'):
    """
    Separate features (X) and target (y) from dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        feature_columns (list): List of column names to use as features.
        target_column (str): Name of the column to use as target.
    
    Returns:
        X (pd.DataFrame): Feature dataframe.
        y (pd.Series): Target values (log-transformed).
    """
    X = df[feature_columns]
    y = log_transform_target(df, target_column)
    return X, y

def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target values.
        test_size (float): Proportion of dataset to include in test split.
        random_state (int): Seed for reproducibility.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)