import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import json


def load_and_merge_data():
    """Load all data sources and merge them into a single DataFrame."""
    
    clients = pd.read_parquet("data/clients.parquet")
    clients.columns = clients.columns.str.lower()  # <--- ESTA LÍNEA ES CLAVE

    billing = pd.read_csv("data/billing.csv", sep=";")
    billing.columns = billing.columns.str.lower()

    churn = pd.read_parquet("data/data_churn.parquet")
    churn.columns = churn.columns.str.lower()

    with open("data/tenure.json") as f:
        tenure = pd.DataFrame(json.load(f))
    tenure.columns = tenure.columns.str.lower()

    tenure_latest = tenure.sort_values("date").groupby("customerid").tail(1)

    df = clients.merge(billing, on="customerid", how="left")
    df = df.merge(tenure_latest, on="customerid", how="left")
    df = df.merge(churn, on="customerid", how="left", suffixes=('', '_dup'))
    return df


def filter_by_customerid(df, valid_digits):
    """Filter customers whose last digit of customerid is in valid_digits list."""
    return df[df['customerid'].astype(str).str[-1].isin(valid_digits)]


def select_features(df):
    """Select relevant features based on dictionary and domain knowledge."""
    useful_cols = [
        'monthlycharges', 'increased_rate_diff', 'increased_rate_pct',
        'tenure_months', 'tenure_penalty', 'contract_months', 'age',
        'phone_lines', 'children', 'gender', 'married', 'internetservice',
        'paperlessbilling', 'paymentmethod', 'contract_channel',
        'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
        'streaming', 'country'
    ]
    return df[useful_cols]


def build_preprocessing_pipeline(numeric_features, categorical_features):
    """Create a preprocessing pipeline for numeric and categorical features."""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor

