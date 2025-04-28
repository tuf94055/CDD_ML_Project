# streamlit_app.py
# Streamlit App for IC50 Prediction
# Mathew Kuruvilla

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_data, clean_data, compute_descriptors
from src.dataset_preparation import prepare_features_and_target, split_train_test

# App Title
st.title("IC50 Prediction from Molecular Descriptors")

# Sidebar for uploading file
st.sidebar.header("Upload Your ChEMBL Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# If no file uploaded, use example dataset
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("Custom dataset uploaded successfully!")
else:
    st.info("No file uploaded. Using example dataset.")
    df = load_data('data/example_dataset.csv')

# Display raw data
st.subheader("Raw Dataset Preview")
st.dataframe(df.head(10))

# Preprocessing
df_clean = clean_data(df)
df_final = compute_descriptors(df_clean)

# Feature selection
features = ['MW', 'LogP', 'NumHAcceptors', 'NumHDonors']

# Prepare dataset
X, y = prepare_features_and_target(df_final, features)
X_train, X_test, y_train, y_test = split_train_test(X, y)

# Sidebars for model hyperparameters
st.sidebar.subheader("Random Forest Hyperparameters")
n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 50)
max_depth = st.sidebar.slider("Max Depth", 1, 30, 5)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 100, 10)
random_state = st.sidebar.slider("Random State (Seed)", 0, 10000, 1)

# Train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5


# Display metrics
st.subheader("Model Evaluation Metrics")
st.write(f"**R² Score:** {r2:.4f}")
st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")

# Plot Actual vs Predicted IC50 (Log Scale)
st.subheader("Actual vs Predicted IC50 (Log Scale)")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.7)

# Add a reference line y=x for perfect predictions
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

ax.set_xlabel('Actual IC50 (log scale)')
ax.set_ylabel('Predicted IC50 (log scale)')
ax.set_title('Actual vs Predicted IC50')

st.pyplot(fig)