# src/data_preprocessing.py
# Data Processing
# Mathew Kuruvilla

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def load_data(file_path):
    """
    Load dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def clean_data(df, standard_type="IC50"):
    """
    Clean the dataset by dropping missing values and duplicates, and filtering by standard_type.

    Args:
        df (pd.DataFrame): Raw dataset containing SMILES and bioactivity data.
        standard_type (str, optional): Bioactivity measurement type to filter by (default is 'IC50').

    Returns:
        pd.DataFrame: Cleaned dataset ready for descriptor calculation.
    """
    df = df.dropna(subset=['canonical_smiles', 'standard_value'])
    df = df.drop_duplicates(subset='canonical_smiles')
    df = df[df['standard_type'] == standard_type]
    return df

def lipinski(smiles_list, verbose=False):
    """
    Compute Lipinski descriptors: Molecular Weight (MW), LogP, Number of H-bond Donors, Number of H-bond Acceptors.

    Args:
        smiles_list (list or pd.Series): List or Series of SMILES strings.
        verbose (bool, optional): If True, prints errors encountered during descriptor calculation. Default is False.

    Returns:
        pd.DataFrame: DataFrame containing molecular descriptors for each SMILES.
    """
    moldata = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        moldata.append(mol)
       
    baseData = []
    for i, mol in enumerate(moldata):
        if mol is not None:
            try:
                desc_MolWt = Descriptors.MolWt(mol)
                desc_MolLogP = Descriptors.MolLogP(mol)
                desc_NumHDonors = Lipinski.NumHDonors(mol)
                desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)

                row = [desc_MolWt, desc_MolLogP, desc_NumHDonors, desc_NumHAcceptors]
                baseData.append(row)

            except Exception as e:
                if verbose:
                    print(f"Descriptor calculation failed for molecule at index {i}: {e}")
                baseData.append([None, None, None, None])
        else:
            if verbose:
                print(f"Invalid SMILES at index {i}")
            baseData.append([None, None, None, None])
    
    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)
    
    return descriptors

def compute_descriptors(df):
    """
    Compute molecular descriptors for all compounds in the dataset.

    Args:
        df (pd.DataFrame): Cleaned dataset containing a 'canonical_smiles' column.

    Returns:
        pd.DataFrame: Dataset with appended descriptor columns (MW, LogP, NumHDonors, NumHAcceptors).
    """
    descriptors = lipinski(df['canonical_smiles'])
    final_df = pd.concat([df.reset_index(drop=True), descriptors], axis=1)
    final_df = final_df.dropna(subset=['MW', 'LogP', 'NumHDonors', 'NumHAcceptors'])
    return final_df
