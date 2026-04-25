import os
import warnings
from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def preprocess_customer_data(df: pd.DataFrame, reference_year: int = 2026) -> pd.DataFrame:
    """Preprocess the customer dataset and return the scaled DataFrame."""
    df_clean = df.copy()

    # 1. Drop identifiers and irrelevant columns
    cols_to_drop = ['customer_id', 'customer_name']
    df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns], inplace=True)

    # 2. Date conversion and Age calculation
    if 'customer_birthdate' in df_clean.columns:
        df_clean['customer_birthdate'] = pd.to_datetime(df_clean['customer_birthdate'], errors='coerce')
        df_clean['Age'] = reference_year - df_clean['customer_birthdate'].dt.year
        df_clean.drop(columns=['customer_birthdate'], inplace=True)

    # 3. Feature Engineering
    if 'kids_home' in df_clean.columns and 'teens_home' in df_clean.columns:
        df_clean['kids_home'] = df_clean['kids_home'].fillna(0)
        df_clean['teens_home'] = df_clean['teens_home'].fillna(0)
        df_clean['children_at_home'] = df_clean['kids_home'] + df_clean['teens_home']

    spend_cols = [col for col in df_clean.columns if col.startswith('lifetime_spend')]
    if spend_cols:
        df_clean[spend_cols] = df_clean[spend_cols].fillna(0)
        df_clean['total_spent'] = df_clean[spend_cols].sum(axis=1)

    # 4. Impute missing values
    categorical_cols = df_clean.select_dtypes(include=['object', 'bool']).columns
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns

    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val[0])

    numeric_cols_to_fill = [col for col in numeric_cols if col not in spend_cols and col not in ['kids_home', 'teens_home']]
    for col in numeric_cols_to_fill:
        if df_clean[col].isnull().any():
            if col == 'loyalty_card_number':
                df_clean[col] = df_clean[col].fillna(0.0)
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 5. Encoding Categorical Variables
    if 'customer_gender' in df_clean.columns:
        df_clean = pd.get_dummies(df_clean, columns=['customer_gender'], drop_first=True)

    # 6. Scale Features
    df_scaled = df_clean.copy()
    numeric_cols = df_scaled.select_dtypes(include=['int64', 'float32', 'float64']).columns
    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    return df_scaled


def preprocess_customer_file(
    input_file: str,
    output_file: Optional[str] = 'customer_info_preprocessed.csv',
    reference_year: int = 2026,
) -> pd.DataFrame:
    """Load the dataset, preprocess it and optionally save the output to CSV."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    df_preprocessed = preprocess_customer_data(df, reference_year=reference_year)

    if output_file:
        df_preprocessed.to_csv(output_file, index=False)

    return df_preprocessed


if __name__ == '__main__':
    input_file = 'customer_info.csv'
    output_file = 'customer_info_preprocessed.csv'

    print(f'Loading data from {input_file}...')
    df_processed = preprocess_customer_file(input_file=input_file, output_file=output_file)
    print(f'Preprocessed dataset shape: {df_processed.shape}')
    print(f'Saved preprocessed data to {output_file}')

