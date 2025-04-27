# src/model_evaluation.py
# Model Evaluation Module
# Mathew Kuruvilla

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return key metrics.
    
    Args:
        model: Trained machine learning model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test targets.
    
    Returns:
        dict: Dictionary containing R², MAE, and MSE.
    """
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    metrics = {
        "R2 Score": r2,
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse
    }
    return metrics

def plot_actual_vs_predicted(y_test, y_pred):
    """
    Plot Actual vs Predicted values for visual evaluation.
    
    Args:
        y_test (pd.Series or np.array): True values.
        y_pred (np.array): Predicted values.
    """
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()