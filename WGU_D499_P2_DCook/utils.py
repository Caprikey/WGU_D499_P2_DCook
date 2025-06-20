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

    """
    Generates a DataFrame containing information about each column in the input DataFrame.
    
    Args:
        dataframe (pd.DataFrame): The input DataFrame for which column information is to be generated.
    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - 'Column': The name of each column in the input DataFrame.
            - 'Non-Null Count': The count of non-null values in each column.
            - 'Null Count': The count of null values in each column.
            - 'Dtype': The data type of each column.   
    """
        
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

    """
    Compares the column information of two DataFrames and returns a DataFrame containing the comparison.
    
    Args:
        dataframe1 (pd.DataFrame): The first DataFrame to compare.
        dataframe2 (pd.DataFrame): The second DataFrame to compare.
    Returns:
        pd.DataFrame: A DataFrame containing the comparison of column information between two DataFrames.
    """
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
    """
    Exports the comparison DataFrame to an Excel file with basic formatting.
    
    Args:
        comparison_df (pd.DataFrame): The DataFrame containing the comparison results to export.
        filename (str): The name of the output Excel file. Defaults to "comparison.xlsx".
    Returns:
        None
        Exports the comparison DataFrame to an Excel file with basic formatting.
    """

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
    Converts a string representation of a list in the 'missing_or_unknown' column of a DataFrame row to an actual list.
    
    Args:
        row (pd.Series): A row from a DataFrame containing a column 'missing_or_unknown' which is a string representation of a list.
    Returns:
        list: The converted list from the string representation in the 'missing_or_unknown' column.
    """

    string_list = row['missing_or_unknown']
    try:
        return ast.literal_eval(string_list)
    except (SyntaxError, ValueError):
        fs_string_conversions_failed.append((row['attribute'], string_list))
        return string_list
    
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def get_fs_string_conversions_failed():
    """
    Returns a list of tuples containing the attribute name and the string that failed to convert.
    """
    return fs_string_conversions_failed


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# Converting list in string format back to list format
# https://stackoverflow.com/questions/27442093/how-to-convert-a-list-into-string-and-this-string-back-to-the-initial-list#:~:text=You%20can%20use%20ast.

null_dict = {}

def convert_unknown_missing_to_nan(main_df, missing_unknown_df, verbose=False):
    """
    Quick note: this function is missing a docstring. Here's a simple one you can add to match your project style.
    
    Args:
        main_df (pd.DataFrame): The main DataFrame where missing or unknown values will be replaced with NaN.
        missing_unknown_df (pd.DataFrame): A DataFrame containing the attributes and their corresponding missing or unknown values.
        verbose (bool): If True, prints detailed information about the replacement process. Defaults to False.
    Returns:
        main_df (pd.DataFrame): The DataFrame with missing or unknown values replaced with NaN.
        null_dict (dict): A dictionary containing the count of null values before and after the replacement for each column.
    """
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

    """
    Displays the descriptive statistics of two columns side by side in a DataFrame format.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame containing the columns to be compared.
        column_name_1 (str): The name of the first column to describe.
        column_name_2 (str): The name of the second column to describe.
        dc1_name (str, optional): Custom name for the first column's description. Defaults to None.
        dc2_name (str, optional): Custom name for the second column's description. Defaults to None.
    Returns:
        None: This function displays the descriptive statistics of the two columns side by side.
    """
        
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
    """
    Checks for and removes temporary columns related to row NaN values from the input DataFrame.
    
    Args:
        dataframe (pd.DataFrame): The input DataFrame from which temporary columns are to be removed.
    Returns:
        pd.DataFrame: The input DataFrame with temporary columns removed.
    """

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
    """
    Analyzes the 'type' column in the summary DataFrame to categorize features and count their occurrences.
    
    Args:
        summary_dataframe (pd.DataFrame): A DataFrame containing a 'type' column that categorizes features.
    Returns:
        feature_counts (pd.DataFrame): A DataFrame containing the count of each feature type.
        interval_columns (pd.DataFrame): A DataFrame containing columns of type 'interval'. 
        numeric_columns (pd.DataFrame): A DataFrame containing columns of type 'numeric'.
        ordinal_columns (pd.DataFrame): A DataFrame containing columns of type 'ordinal'.
        categorical_columns (pd.DataFrame): A DataFrame containing columns of type 'categorical'.
        mixed_columns (pd.DataFrame): A DataFrame containing columns of type 'mixed'.

    """

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
    """
    Checks the data types of specified columns in a DataFrame and prints whether they are approved types (int or float).S
    
    Args:
        dataframe (pd.DataFrame): The DataFrame containing the columns to be checked.
        column_list (list): A list of column names to check the data types against approved types (int or float).
    Returns:
        None: This function prints the data types of specified columns in the DataFrame and checks if they are approved types (int or float).
    """
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
    """
    Converts all column names in the input DataFrame to uppercase.
    
    Args:
        dataframe (pd.DataFrame): The input DataFrame whose column names are to be converted to uppercase.
    Returns:r
        pd.DataFrame: The input DataFrame with all column names converted to uppercase.
    """

    uppercase_columns_names = []

    for column_name in dataframe.columns:
        uppercase_columns_names.append(column_name.upper())

    dataframe.columns = uppercase_columns_names

    return dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def get_imput_lists(dataframe):

    """
    Generates lists of columns that require imputation based on their data types.    

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing a 'data_type' column that categorizes columns by their data types.
    Returns:
        - category_imputed_columns_list: List of columns with 'ordinal' or 'categorical' data types that require imputation.
        - numeric_imputed_columns_list: List of columns with 'numeric' data type that require imputation.
        - imputed_columns_list: Combined list of all columns that require imputation (both category and numeric).
    """

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
    """
    Compares the columns of two DataFrames and returns a list of column names that are present in one DataFrame but not in the other.
    
    Args:
        dataframe_one (pd.DataFrame): The first DataFrame to compare.
        dataframe_two (pd.DataFrame): The second DataFrame to compare.
    Returns:
        difference_list (list): A list of column names that are present in one DataFrame but not in the other.
    """

    column_list_one = set(dataframe_one.columns)
    column_list_two = set(dataframe_two.columns)

    difference_list = list(column_list_one.symmetric_difference(column_list_two))


    return difference_list



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

if __name__ == "__main__":
    app()
