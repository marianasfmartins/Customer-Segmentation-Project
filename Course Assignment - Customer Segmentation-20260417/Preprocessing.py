import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer


def preprocess_data_standardscaler(df):
    df = df.copy()

    drop_cols = [
        'customer_id',
        'customer_name', 'has_loyalty_card',
        'customer_gender', 'geometry'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # --- Impute missing values using KNN ---
    clustering_cols = df.select_dtypes(include=np.number).columns.tolist()
    imputer = KNNImputer(n_neighbors=5)
    df[clustering_cols] = imputer.fit_transform(df[clustering_cols])

    # Identify spend columns dynamically from the dataframe
    spend_cols = [col for col in df.columns if 'lifetime_spend' in col]
    log_cols = [col for col in spend_cols if col != 'lifetime_spend_vegetables'] + ['total_lifetime_spend']
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Standard Scaler
    scaler = StandardScaler()
    clustering_cols_final = df.select_dtypes(include=np.number).columns.tolist()
    df[clustering_cols_final] = scaler.fit_transform(df[clustering_cols_final])

    return df


def preprocess_data_robustscaler(df):
    df = df.copy()

    drop_cols = [
        'customer_id',
        'customer_name', 'has_loyalty_card',
        'customer_gender', 'geometry'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    clustering_cols = df.select_dtypes(include=np.number).columns.tolist()
    imputer = KNNImputer(n_neighbors=5)
    df[clustering_cols] = imputer.fit_transform(df[clustering_cols])

    spend_cols = [col for col in df.columns if 'lifetime_spend' in col]
    log_cols = [col for col in spend_cols if col != 'lifetime_spend_vegetables'] + ['total_lifetime_spend']
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    scaler = RobustScaler()
    clustering_cols_final = df.select_dtypes(include=np.number).columns.tolist()
    df[clustering_cols_final] = scaler.fit_transform(df[clustering_cols_final])

    return df


def preprocess_data_minmaxscaler(df):
    df = df.copy()

    drop_cols = [
        'customer_id',
        'customer_name', 'has_loyalty_card',
        'customer_gender', 'geometry'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    clustering_cols = df.select_dtypes(include=np.number).columns.tolist()
    imputer = KNNImputer(n_neighbors=5)
    df[clustering_cols] = imputer.fit_transform(df[clustering_cols])

    spend_cols = [col for col in df.columns if 'lifetime_spend' in col]
    log_cols = [col for col in spend_cols if col != 'lifetime_spend_vegetables'] + ['total_lifetime_spend']
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    scaler = MinMaxScaler()
    clustering_cols_final = df.select_dtypes(include=np.number).columns.tolist()
    df[clustering_cols_final] = scaler.fit_transform(df[clustering_cols_final])

    return df


### Preprocess data for analysis (keep unscaled values and categorical columns) 

def cluster_analysis(df):
    """
    Preprocesses data for cluster analysis.
    Returns preprocessed data with true (unscaled) values and all variables (categorical + numeric).
    """    
    # Impute missing values using KNN ---
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    imputer = KNNImputer(n_neighbors=5)
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # Identify spend columns dynamically from the dataframe
    spend_cols = [col for col in df.columns if 'lifetime_spend' in col]
    log_cols = [col for col in spend_cols if col != 'lifetime_spend_vegetables'] + ['total_lifetime_spend']
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    return df

