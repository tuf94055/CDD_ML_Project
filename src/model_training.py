# src/model_training.py
# Model Training Module
# Mathew Kuruvilla

from sklearn.ensemble import RandomForestRegressor
# (Later you can also add XGBoost, Gradient Boosting, etc.)

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Regressor model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed for reproducibility.

    Returns:
        model (RandomForestRegressor): Trained Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model