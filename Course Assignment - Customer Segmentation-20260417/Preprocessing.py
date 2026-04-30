import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def preprocess_data(df):
    
    df = df.copy()
    
    # --- 1. Drop unnecessary features ---
    drop_cols = [
        'customer_id',
        'customer_name',
        'loyalty_card_number',
        'birth_month',
        'birth_day',
        'latitude',
        'longitude',
        'customer_birthdate'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    
    # --- 2. Feature Engineering ---
    df['age'] = 2026 - df['birth_year']
    df.drop(columns=['birth_year'], inplace=True)
    
    df['total_kids'] = df['kids_home'] + df['teens_home']
    df.drop(columns=['kids_home', 'teens_home'], inplace=True)
    
    spend_cols = [col for col in df.columns if 'lifetime_spend' in col]
    df['total_lifetime_spend'] = df[spend_cols].sum(axis=1)

    # --- 3. Fix percentage_of_products_bought_promotion ---
    df['percentage_of_products_bought_promotion'] = df['percentage_of_products_bought_promotion'].clip(lower=0)

    # --- 4. Impute missing values using KNN ---
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    imputer = KNNImputer(n_neighbors=5)
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # --- 5. Log transform right-skewed features ---
    log_cols = [col for col in spend_cols if col != 'lifetime_spend_vegetabl