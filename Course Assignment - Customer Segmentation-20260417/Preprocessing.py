import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define file paths
input_file = "customer_info.csv"
output_file = "customer_info_preprocessed.csv"

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found in the current directory.")
else:
    # 1. Load Data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Original dataset shape: {df.shape}\n")

    print("Starting preprocessing...")
    df_clean = df.copy()
    
    # 1. Drop identifiers and irrelevant columns
    cols_to_drop = ['customer_id', 'customer_name']
    df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns], inplace=True)
    
    # 2. Date conversion and Age calculation
    if 'customer_birthdate' in df_clean.columns:
        print("Calculating Age...")
        try:
            # customer_birthdate has format '02/12/1970 01:36 PM'
            df_clean['customer_birthdate'] = pd.to_datetime(df_clean['customer_birthdate'], errors='coerce')
            
            # Using 2026 as the reference year since year_first_transaction maxes out around there
            reference_year = 2026 
            df_clean['Age'] = reference_year - df_clean['customer_birthdate'].dt.year
            df_clean.drop(columns=['customer_birthdate'], inplace=True)
        except Exception as e:
            print(f"Warning: Issue converting birthdate. {e}")
            
    # 3. Feature Engineering
    print("Engineering features...")
    # Calculate total children at home
    if 'kids_home' in df_clean.columns and 'teens_home' in df_clean.columns:
        df_clean['kids_home'] = df_clean['kids_home'].fillna(0)
        df_clean['teens_home'] = df_clean['teens_home'].fillna(0)
        df_clean['children_at_home'] = df_clean['kids_home'] + df_clean['teens_home']
        
    # Calculate Total Spend across different categories
    spend_cols = [col for col in df_clean.columns if col.startswith('lifetime_spend')]
    if spend_cols:
        # Fill missing spend values with 0 so the summation works correctly
        df_clean[spend_cols] = df_clean[spend_cols].fillna(0)
        df_clean['total_spent'] = df_clean[spend_cols].sum(axis=1)

    # 4. Impute missing values
    print("Handling remaining missing values...")
    categorical_cols = df_clean.select_dtypes(include=['object', 'bool']).columns
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    
    # Fill categorical features with mode
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
            
    # Fill numerical features with median
    numeric_cols_to_fill = [col for col in numeric_cols if col not in spend_cols and col not in ['kids_home', 'teens_home']]
    for col in numeric_cols_to_fill:
        if df_clean[col].isnull().any():
            # Special case for Loyalty Card Number (Assume missing means no card = 0.0)
            if col == 'loyalty_card_number':
                df_clean[col] = df_clean[col].fillna(0.0)
            else:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
            
    # 5. Encoding Categorical Variables
    print("Encoding categorical variables...\n")
    if 'customer_gender' in df_clean.columns:
        # One-Hot Encoding to turn gender into binary format columns (e.g., customer_gender_male)
        df_clean = pd.get_dummies(df_clean, columns=['customer_gender'], drop_first=True)
        
    # 6. Scale Features
    print("Scaling numerical features...")
    df_scaled = df_clean.copy()
    
    # It is common practice to scale the numerical variables for ML clustering algorithms (e.g. K-Means)
    numeric_cols = df_scaled.select_dtypes(include=['int64', 'float32', 'float64']).columns
    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    
    print("\n--- Summary of missing values after preprocessing ---")
    print(df_scaled.isnull().sum().sum(), "total missing values remaining")
    
    print(f"Preprocessed dataset shape: {df_scaled.shape}")
    
    # 7. Save to CSV
    df_scaled.to_csv(output_file, index=False)
    print(f"Saved preprocessed data to {output_file}")
