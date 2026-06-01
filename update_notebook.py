import json

file_path = "c:/Users/maria/Documents/Universidade/Machine Learning II/Customer-Segmentation-Project/Course Assignment - Customer Segmentation-20260417/main.ipynb"
with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and len(cell['source']) > 0:
        if 'df = pd.read_csv(\'customer_info_engineered.csv\')\n' in cell['source']:
            cell['source'] = [
                "# Load the engineered customer dataset\n",
                "df = pd.read_csv('customer_info_engineered.csv')\n",
                "\n",
                "# Compute total kids (Option C: Perfil Familiar e Demográfico)\n",
                "df['total_kids'] = df['kids_home'] + df['teens_home']\n",
                "\n",
                "# Define the selected subset of features for clustering (Family & Demographics)\n",
                "option_c_features = [\n",
                "    'total_kids', \n",
                "    'age', \n",
                "    'lifetime_total_distinct_products',\n",
                "    'year_first_transaction'\n",
                "]\n",
                "\n",
                "# Filter the dataset before calling the preprocessing functions to prevent KNNImputer from hanging\n",
                "df_filtered = df[option_c_features].copy()\n",
                "\n",
                "# Preprocess and scale the dataset using Standard, Robust, and MinMax scalers\n",
                "df_processed_st = preprocess_data_standardscaler(df_filtered.copy())\n",
                "df_processed_rb = preprocess_data_robustscaler(df_filtered.copy())\n",
                "df_processed_mm = preprocess_data_minmaxscaler(df_filtered.copy())\n",
                "\n",
                "# Setup standard features datasets\n",
                "df_saude_st = df_processed_st\n",
                "df_saude_rb = df_processed_rb\n",
                "df_saude_mm = df_processed_mm\n",
                "\n",
                "# Run baseline profile analysis on the unscaled customer dataset (using fast median imputation)\n",
                "df_profile = df.copy()\n",
                "total_spend_prof = df_profile['total_lifetime_spend'].replace(0, np.nan)\n",
                "df_profile['pct_vegetables'] = (df_profile['lifetime_spend_vegetables'] / total_spend_prof).fillna(0)\n",
                "df_profile['pct_meat'] = (df_profile['lifetime_spend_meat'] / total_spend_prof).fillna(0)\n",
                "df_profile['pct_fish'] = (df_profile['lifetime_spend_fish'] / total_spend_prof).fillna(0)\n",
                "df_profile['pct_alcohol'] = (df_profile['lifetime_spend_alcohol_drinks'] / total_spend_prof).fillna(0)\n",
                "df_profile['pct_videogames'] = (df_profile['lifetime_spend_videogames'] / total_spend_prof).fillna(0)\n",
                "\n",
                "df_cluster_analysis = cluster_analysis(df_profile)"
            ]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("done")
