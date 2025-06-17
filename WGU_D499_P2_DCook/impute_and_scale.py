from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from WGU_D499_P2_DCook.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, EXTERNAL_DATA_DIR, ARCHIVED_DATA_DIR, MODELS_DIR

app = typer.Typer()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



def perform_scale_data(prescaled_dataframe, features_dataframe):
    """
    Scale data using parameters from rows with no missing values,
    then impute missing values and apply the precomputed scaling.
    """
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    # === Step 1: Helper to get missing % and column types ===
    def get_missing_info(df, features_df):
        missing_pct = (df.isnull().sum() / df.shape[0]).sort_values(ascending=False)
        missing_df = missing_pct[missing_pct > 0].reset_index()
        missing_df.columns = ['column_name', 'missing_percentage']
        missing_df['total_missing'] = missing_df['column_name'].apply(lambda col: df[col].isnull().sum())

        missing_df = missing_df.merge(
            features_df[['attribute', 'type', 'information_level']],
            left_on='column_name',
            right_on='attribute',
            how='left'
        ).drop(columns='attribute').rename(columns={'type': 'data_type'})

        return missing_df

    # === Step 2: Split by type ===
    def get_imputation_lists(missing_df):
        cat = missing_df[missing_df['data_type'].isin(['ordinal', 'categorical'])]['column_name'].tolist()
        num = missing_df[missing_df['data_type'] == 'numeric']['column_name'].tolist()
        return cat, num, cat + num

    # === Step 3: Get clean rows for scaling ===
    missing_df = get_missing_info(prescaled_dataframe, features_dataframe)
    cat_cols, num_cols, imputed_cols = get_imputation_lists(missing_df)

    fully_observed_rows = prescaled_dataframe.dropna().reset_index(drop=True)

    # Temporarily impute categorical to allow get_dummies for scaling
    cat_temp_imputer = SimpleImputer(strategy='most_frequent')
    cat_temp = pd.DataFrame(cat_temp_imputer.fit_transform(fully_observed_rows[cat_cols]), columns=cat_cols)

    # Recombine with numeric and untouched
    num_data = fully_observed_rows[num_cols].reset_index(drop=True)
    untouched_data = fully_observed_rows.drop(columns=imputed_cols).reset_index(drop=True)

    temp_encoded = pd.concat([untouched_data, num_data, cat_temp], axis=1)
    temp_encoded = pd.get_dummies(temp_encoded, columns=cat_cols, drop_first=False)

    # === Step 4: Fit scaler on fully observed, temporarily encoded data ===
    scaler = StandardScaler()
    scaler.fit(temp_encoded)

    # === Step 5: Now impute the full dataset ===
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    full_num = pd.DataFrame(num_imputer.fit_transform(prescaled_dataframe[num_cols]), columns=num_cols)
    full_cat = pd.DataFrame(cat_imputer.fit_transform(prescaled_dataframe[cat_cols]), columns=cat_cols)
    full_untouched = prescaled_dataframe.drop(columns=imputed_cols).reset_index(drop=True)

    full_combined = pd.concat([full_untouched, full_num, full_cat], axis=1)
    full_encoded = pd.get_dummies(full_combined, columns=cat_cols, drop_first=False)

    # === Step 6: Align to scaler's features and transform ===
    full_encoded = full_encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)
    scaled_output = pd.DataFrame(scaler.transform(full_encoded), columns=scaler.feature_names_in_)

    return scaled_output, scaler, missing_df, num_imputer, cat_imputer



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



def apply_existing_scaling(prescaled_df, features_df, num_imputer, cat_imputer, scaler):
    import pandas as pd

    # === Identify types and imputed columns ===
    def get_missing_info(df, features_df):
        missing_pct = (df.isnull().sum() / df.shape[0]).sort_values(ascending=False)
        missing_df = missing_pct[missing_pct > 0].reset_index()
        missing_df.columns = ['column_name', 'missing_percentage']
        missing_df['total_missing'] = missing_df['column_name'].apply(lambda col: df[col].isnull().sum())

        missing_df = missing_df.merge(
            features_df[['attribute', 'type', 'information_level']],
            left_on='column_name',
            right_on='attribute',
            how='left'
        ).drop(columns='attribute').rename(columns={'type': 'data_type'})

        return missing_df

    def get_imputation_lists(missing_df):
        cat = missing_df[missing_df['data_type'].isin(['ordinal', 'categorical'])]['column_name'].tolist()
        num = missing_df[missing_df['data_type'] == 'numeric']['column_name'].tolist()
        return cat, num, cat + num

    # === Process ===
    missing_df = get_missing_info(prescaled_df, features_df)
    cat_cols, num_cols, imputed_cols = get_imputation_lists(missing_df)

    num_data = pd.DataFrame(num_imputer.transform(prescaled_df[num_cols]), columns=num_cols)
    cat_data = pd.DataFrame(cat_imputer.transform(prescaled_df[cat_cols]), columns=cat_cols)
    untouched_data = prescaled_df.drop(columns=imputed_cols).reset_index(drop=True)

    combined = pd.concat([untouched_data, num_data, cat_data], axis=1)
    encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=False)

    # Align columns to match the training scaler
    encoded = encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)

    scaled_data = pd.DataFrame(scaler.transform(encoded), columns=scaler.feature_names_in_)
    return scaled_data

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def apply_existing_scaling_v2(prescaled_df, features_df, num_imputer, cat_imputer, scaler):
    # Get column names the imputers were trained on
    num_cols_fit = num_imputer.feature_names_in_
    cat_cols_fit = cat_imputer.feature_names_in_

    # Impute numerical and categorical data using the original training columns
    num_data = pd.DataFrame(num_imputer.transform(prescaled_df[num_cols_fit]), columns=num_cols_fit)
    cat_data = pd.DataFrame(cat_imputer.transform(prescaled_df[cat_cols_fit]), columns=cat_cols_fit)

    # Drop imputed columns to isolate untouched columns
    imputed_cols = list(num_cols_fit) + list(cat_cols_fit)
    untouched_data = prescaled_df.drop(columns=imputed_cols).reset_index(drop=True)

    # Recombine all parts
    combined = pd.concat([untouched_data, num_data, cat_data], axis=1)

    # Apply scaling
    scaled = pd.DataFrame(scaler.transform(combined), columns=combined.columns)

    return scaled


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def apply_existing_scaling_v3(prescaled_df, features_df, num_imputer, cat_imputer, scaler):
    import pandas as pd

    # Get imputation columns from trained imputers
    num_cols = list(num_imputer.feature_names_in_)
    cat_cols = list(cat_imputer.feature_names_in_)

    # Impute
    num_data = pd.DataFrame(num_imputer.transform(prescaled_df[num_cols]), columns=num_cols)
    cat_data = pd.DataFrame(cat_imputer.transform(prescaled_df[cat_cols]), columns=cat_cols)
    untouched_data = prescaled_df.drop(columns=num_cols + cat_cols).reset_index(drop=True)

    # Recombine
    combined = pd.concat([untouched_data, num_data, cat_data], axis=1)

    # One-hot encode categorical data
    encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=False)

    # Align to scaler's features
    encoded = encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale
    scaled = pd.DataFrame(scaler.transform(encoded), columns=scaler.feature_names_in_)

    return scaled

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def save_scaler_joblib(scaler, save_path: Path = MODELS_DIR / "scaler.pkl"):
    """
    Save the scaler to a file.
    """
    import joblib
    joblib.dump(scaler, save_path)
    logger.info(f"Scaler saved to {save_path}")


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def load_scaler_joblib(load_path: Path = MODELS_DIR / "scaler.pkl"):  
    """
    Load the scaler from a file.
    """
    import joblib
    scaler = joblib.load(load_path)
    logger.info(f"Scaler loaded from {load_path}")
    return scaler   


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def save_scaler_pickledump(scaler, save_path: Path = MODELS_DIR / "scaler.pkl"):
    """
    Save the scaler to a file.
    """
    from pickle import dump
    with open(save_path, 'wb') as file:
        dump(scaler, file)
    logger.info(f"Scaler saved to {save_path}")


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def load_scaler_pickledump(load_path: Path = MODELS_DIR / "scaler.pkl"):  
    """
    Load the scaler from a file.
    """
    import pickle
    
    with open(load_path, 'rb') as file:
        load_scaler = pickle.load(file)

        scaler = load_scaler

    logger.info(f"Scaler loaded from {load_path}")
    return scaler   




#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 




#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

if __name__ == "__main__":
    app()
