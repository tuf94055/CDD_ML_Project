# CDD_ML_Project

### Machine Learning Pipeline for Bioactivity Prediction

This repository contains a machine learning project for predicting bioactivity based on chemical structure.  
The workflow was initially developed for molecules targeting the coronavirus Replicase Polyprotein 1ab, using SMILES strings and molecular descriptors.  
The current project has been generalized into a pipeline for analyzing bioactivity data for any protein target.

---

### Features:
- Load and clean bioactivity datasets from CSV.
- Compute Lipinski descriptors (Molecular Weight, LogP, H-bond Donors, H-bond Acceptors) from SMILES.
- Train various machine learning models (Random Forest amongst them) to predict bioactivity (IC50 values).
- Compare model performance and visualize results.
- Streamlit web app for interactive exploration (`streamlit_app.py`).

---

### Project Structure:
```
CDD_ML_Project/
├── notebooks/
│   └── 01_run_pipeline.ipynb    # Main Jupyter notebook pipeline
├── src/
│   ├── data_preprocessing.py    # Load, clean, descriptor calculation
│   ├── eda.py                   # Exploratory data analysis functions
│   ├── dataset_preparation.py   # Train/test split, feature preparation
│   ├── model_training.py        # Model training (Random Forest)
│   ├── model_evaluation.py      # Evaluation and plotting
├── data/
│   └── example_dataset.csv      # Example input dataset
├── streamlit_app.py             # Streamlit app interface
├── requirements.txt             # Required Python packages
└── README.md                    # Project overview
```
---

### Required Packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- rdkit-pypi
- streamlit

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

### How to Run:

1. Clone this repository:

```
git clone https://github.com/your-username/CDD_ML_Project.git
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Run the main notebook:

Open notebooks/01_run_pipeline.ipynb and run all cells sequentially.

4. (Optional) Launch the Streamlit app:

```
streamlit run streamlit_app.py
```

---

### Outputs:

- Cleaned datasets with calculated descriptors.
- Machine learning model training results.
- Scatter plots comparing actual vs. predicted bioactivity.
- Model evaluation metrics (e.g., RMSE, R² score).

---

### Author:

Mathew Kuruvilla
