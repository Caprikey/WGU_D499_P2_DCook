from pathlib import Path

from loguru import logger

import pandas as pd
import numpy as np
import ast

from tqdm import tqdm
import typer

from WGU_D499_P2_DCook.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    #input_path: Path = RAW_DATA_DIR / "dataset.csv",
    #output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def get_info_to_dataframe(dataframe):
    column_info_dataframe = pd.DataFrame({
        "Column": dataframe.columns,
        "Non-Null Count": dataframe.notnull().sum().values,
        "Null Count": dataframe.isnull().sum().values,
        "Dtype": dataframe.dtypes.values
    })

    column_info_dataframe.reset_index(drop=True, inplace=True)

    return column_info_dataframe

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def compare_dataframe_info(dataframe1, dataframe2):
    info1 = get_info_to_dataframe(dataframe1)
    info2 = get_info_to_dataframe(dataframe2)

    # Rename columns for clarity before merging
    info1.columns = [f"{col} (df1)" for col in info1.columns]
    info2.columns = [f"{col} (df2)" for col in info2.columns]

    # Merge on column name
    merged = pd.merge(
        info1,
        info2,
        left_on="Column (df1)",
        right_on="Column (df2)",
        how="outer",
        suffixes=('_df1', '_df2')
    )

    # Drop redundant merge key
    merged.drop(columns=["Column (df2)"], inplace=True)

    # Optional: Add comparison flags
    merged["Column Match"] = merged["Column (df1)"] == merged["Column (df1)"]
    merged["Non-Null Match"] = merged["Non-Null Count (df1)"] == merged["Non-Null Count (df2)"]
    merged["Null Match"] = merged["Null Count (df1)"] == merged["Null Count (df2)"]
    merged["Dtype Match"] = merged["Dtype (df1)"] == merged["Dtype (df2)"]

    return merged

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def export_comparison_to_excel(comparison_df, filename="comparison.xlsx"):
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        comparison_df.to_excel(writer, index=False, sheet_name="Comparison")

        # Optional formatting
        workbook  = writer.book
        worksheet = writer.sheets["Comparison"]

        for i, col in enumerate(comparison_df.columns):
            max_len = max(comparison_df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_len)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

fs_string_conversions_failed = []

def convert_string_to_list(row):
    """
    Helper to safely evaluate stringified lists from the 'missing_or_unknown' column.
    """
    string_list = row['missing_or_unknown']
    try:
        return ast.literal_eval(string_list)
    except (SyntaxError, ValueError):
        fs_string_conversions_failed.append((row['attribute'], string_list))
        return string_list
    
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def get_fs_string_conversions_failed():

    return fs_string_conversions_failed


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# Converting list in string format back to list format
# https://stackoverflow.com/questions/27442093/how-to-convert-a-list-into-string-and-this-string-back-to-the-initial-list#:~:text=You%20can%20use%20ast.

null_dict = {}

def convert_unknown_missing_to_nan(main_df, missing_unknown_df, verbose=False):

    for index, row in missing_unknown_df.iterrows():

        column_name = row['attribute']

        pre_null_count = main_df[column_name].isnull().sum()

        match_items_total = 0
        
        for item in row['missing_or_unknown']:

            match_count = main_df[column_name].value_counts().get(item, 0)

            match_items_total += match_count
            
            replace_with = np.nan
            
            main_df[column_name].replace(to_replace=item, value=replace_with, inplace=True)
    
        post_null_count = main_df[column_name].isnull().sum()

        # Store null counts in null_dict
        null_dict[column_name] = {
            'pre_null_count': pre_null_count,
            'post_null_count': post_null_count
        }

        if verbose:
            print(f"{column_name}: {pre_null_count} --> {post_null_count} nulls after converting {match_items_total} values")

    return main_df, null_dict


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def null_dict_to_dataframe(null_dictionary):
    """
    Converts the null_dict to a DataFrame for easier analysis.
    """
    if not null_dictionary:
        print("null_dictionary is empty. Returning an empty DataFrame.")
        return pd.DataFrame()

    # Convert the dictionary to a DataFrame
    # Each key becomes a row, and each value becomes a column
    null_dataframe = pd.DataFrame.from_dict(null_dictionary, orient='index')

    # Reset index to turn column names into a column
    null_dataframe.reset_index(inplace=True)

    # Rename the index column to 'column'
    null_dataframe.rename(columns={'index': 'column'}, inplace=True)


    return null_dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


from IPython.display import display

def display_describe_column_comparison_side_by_side(dataframe, column_name_1, column_name_2, dc1_name=None, dc2_name=None):
    describe_column_1 = dataframe[column_name_1].describe()
    describe_column_2 = dataframe[column_name_2].describe()

    # Convert both to DataFrames for better side-by-side display
    describe_column_1_df = describe_column_1.to_frame(name=dc1_name if dc1_name else column_name_1)
    describe_column_2_df = describe_column_2.to_frame(name=dc2_name if dc2_name else column_name_2)
    

    combined_dataframes = pd.concat([describe_column_1_df, describe_column_2_df], axis=1)
    display(combined_dataframes)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



def check_and_remove_row_nan_columns(dataframe):

    temp_columns_double_check = ['row_nan_total_mean', 'row_nan_category']

    for column in dataframe.columns:

        if column in temp_columns_double_check:
            print(f"Temp Columns found in dataframe: {column}")
            dataframe = dataframe.drop(columns = column, errors='ignore')


    return dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def get_feature_types_and_count(summary_dataframe):

    feature_counts = summary_dataframe['type'].value_counts().reset_index().rename(columns={"index": "feature_type", 0: "count"})

    interval_columns = summary_dataframe.loc[summary_dataframe['type'] == 'interval']

    numeric_columns = summary_dataframe.loc[summary_dataframe['type'] == 'numeric']

    ordinal_columns = summary_dataframe.loc[summary_dataframe['type'] == 'ordinal'] 

    categorical_columns = summary_dataframe.loc[summary_dataframe['type'] == 'categorical']

    mixed_columns = summary_dataframe.loc[summary_dataframe['type'] == 'mixed']
    
    return feature_counts, interval_columns, numeric_columns, ordinal_columns, categorical_columns, mixed_columns

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# Decided this function as it was not needed in the final implementation.

#def convert_category_sets_to_list(binary_set, multi_set, error_set, suffix=None):
#    
#    if suffix is not None:
#        converted_category_lists = {
#            f"binary_columns_{suffix}_list": list(binary_set),
#            f"multilevel_columns_{suffix}_list": list(multi_set),
#            f"error_columns_{suffix}_list": list(error_set)
#        }
#    else:
#        converted_category_lists = {
#            "binary_columns_list": list(binary_set),
#            "multilevel_columns_list": list(multi_set),
#            "error_columns_list": list(error_set)
#        }
#
#    return converted_category_lists


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# Check value types are ints or floats:

def column_dtype_check(dataframe, column_list):
    print("Running column_dtype_check...")
    
    approved_types = ['int', 'float']
    
    for column in set(column_list):  # deduplicate here
        if dataframe[column].dtype not in approved_types:
            print(f"\nColumn {column} has a type of {dataframe[column].dtype} which is NOT an approved type.\n")
        else:
            print(f"Column {column} has an approved type of {dataframe[column].dtype}")



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def convert_column_names_to_uppercase(dataframe):

    uppercase_columns_names = []

    for column_name in dataframe.columns:
        uppercase_columns_names.append(column_name.upper())

    dataframe.columns = uppercase_columns_names

    return dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def get_imput_lists(dataframe):

    category_imputed_data_type = ['ordinal', 'categorical']

    numeric_imputed_data_type = ['numeric']

    category_imputed_columns_list = dataframe[dataframe['data_type'].isin(category_imputed_data_type)]['column_name'].tolist()

    numeric_imputed_columns_list = dataframe[dataframe['data_type'].isin(numeric_imputed_data_type)]['column_name'].tolist()

    imputed_columns_list = category_imputed_columns_list + numeric_imputed_columns_list 

    return category_imputed_columns_list, numeric_imputed_columns_list, imputed_columns_list

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def find_differences_between_columns(dataframe_one, dataframe_two):

    column_list_one = set(dataframe_one.columns)
    column_list_two = set(dataframe_two.columns)

    difference_list = list(column_list_one.symmetric_difference(column_list_two))


    return difference_list



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

if __name__ == "__main__":
    app()
