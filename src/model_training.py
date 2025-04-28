# src/model_training.py
# Model Training Module
# Mathew Kuruvilla

from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
    """
    Train a Random Forest Regressor model with more tuning options.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        n_estimators (int): Number of trees in the forest.
        max_depth (int, optional): Maximum depth of the trees (default is None).
        min_samples_split (int, optional): Minimum number of samples required to split an internal node.
        random_state (int): Random seed for reproducibility.

    Returns:
        model (RandomForestRegressor): Trained Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, 
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  random_state=random_state)
    model.fit(X_train, y_train)
    return model