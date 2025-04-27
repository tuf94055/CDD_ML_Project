# streamlit_app.py
# Streamlit App for IC50 Prediction
# Mathew Kuruvilla

import streamlit as st
import pandas as pd

from src.data_preprocessing import load_data, clean_data, compute_descriptors
from src.dataset_preparation import prepare_features_and_target, split_train_test
from src.model_training import train_random_forest
from src.model_evaluation import evaluate_model, plot_actual_vs_predicted

# App Title
st.title("IC50 Prediction from Molecular Descriptors")

# Sidebar for uploading file
st.sidebar.header("Upload Your Dataset")
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
st.dataframe(df.head())

# Preprocessing
df_clean = clean_data(df)
df_final = compute_descriptors(df_clean)

# Feature selection
features = ['MW', 'LogP', 'NumHAcceptors', 'NumHDonors']

# Prepare dataset
X, y = prepare_features_and_target(df_final, features)
X_train, X_test, y_train, y_test = split_train_test(X, y)

# Train model
model = train_random_forest(X_train, y_train)

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)

# Display metrics
st.subheader("Model Evaluation Metrics")
for metric, value in metrics.items():
    st.write(f"**{metric}:** {value:.4f}")

# Plot Actual vs Predicted
st.subheader("Actual vs Predicted IC50 (Log Scale)")
y_pred = model.predict(X_test)
plot_actual_vs_predicted(y_test, y_pred)
