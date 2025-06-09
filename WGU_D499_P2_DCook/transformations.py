from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import pathlib as Path

from WGU_D499_P2_DCook.config import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, ARCHIVED_DATA_DIR
from WGU_D499_P2_DCook.dataset import write_checkpoints

app = typer.Typer()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



# Remove outlier columns from a DataFrame based on a list of outlier column names.

def remove_outlier_columns(dataframe, outlier_column_list, export_outliers_to_file = True, export_file_path = "archived_data", export_file_name = "outlier_columns", update_features_summary_dataframe = False, features_summary_dataframe_object = None):
    
    dataframe_outlier_columns = dataframe[outlier_column_list]
    
    print(f"Total of {dataframe_outlier_columns.shape[1]} were found out of a total of {dataframe.shape[1]} columns")
    
    if export_outliers_to_file:
        
        #current_folder = Path.cwd()
        
        #print(f"Export Path is: {current_folder}")

        write_checkpoints(export_file_path, export_file_name, dataframe_outlier_columns)

        #save_folder_path = 
        
        #save_file_name_pickle = export_file_name 

        #save_file_path_pickle = Path("./" + save_file_name_pickle + ".pkl")
        
        #dataframe_outlier_columns.to_pickle(save_file_path_pickle)
        
        #print(f"Export of {dataframe_outlier_columns.shape[1]} Outlier columns was successful")



    dataframe = dataframe.drop(columns=outlier_column_list, axis=1)
    print(f"Dataset now has {dataframe.shape[1]} total features")

    if update_features_summary_dataframe:

        if features_summary_dataframe_object is not None:

            features_summary_dataframe_object = update_features_summary_dataframe(features_summary_dataframe_object, outlier_column_list)

            return dataframe, features_summary_dataframe_object
        
        else:
            print("Warning: `features_summary_dataframe` was not provided.")
            return dataframe, None

    return dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


# Update the features summary DataFrame by removing outlier columns.

def update_features_summary_dataframe(features_summary_dataframe, outlier_column_list):
    """
    Updates the features summary DataFrame by removing outlier columns.

    Args:
        features_summary_dataframe (pd.DataFrame): The DataFrame containing feature summaries.
        outlier_column_list (list): List of outlier column names to be removed.

    Returns:
        pd.DataFrame: Updated features summary DataFrame without outlier columns.
    """
    
    updated_features_summary = features_summary_dataframe[~features_summary_dataframe['attribute'].isin(outlier_column_list)]
    
    return updated_features_summary


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

from sklearn.cluster import KMeans
import numpy as np

def row_nan_find_dynamic_nan_threshold(dataframe, num_clusters = 3, num_init = 'auto', rand_stat = 5654):

    nan_row_dataframe = dataframe.isna().mean(axis=1)

    nan_row_data = nan_row_dataframe.values.reshape(-1, 1)

    #kmeans = KMeans(n_clusters=2, n_init=1000, random_state=42)
    #kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42)
    kmeans = KMeans(n_clusters = num_clusters, n_init = num_init, random_state = rand_stat)

    kmeans.fit(nan_row_data)

    centers = sorted(kmeans.cluster_centers_.flatten())

    threshold = np.mean(centers)  # Midpoint between the two cluster centers

    return threshold



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

from WGU_D499_P2_DCook.plots import plot_countplot_compare_row_nan_dist_per_column

def row_nan_divide_by_threshold(dataframe, 
                                max_loss_allowed=0.20, 
                                max_nan_per_row=0.30, 
                                method='kmeans',
                                return_common_columns=True, 
                                allow_noise=False, 
                                noise_threshold=0.05, 
                                plot_graph=False
                                ):
    
    row_nan_dataframe = dataframe.isna().mean(axis=1)

    # 1. Try dynamic threshold from KMeans
    if method == 'kmeans':
        print("Entered Method")
        threshold = row_nan_find_dynamic_nan_threshold(dataframe)
        print(f"Threshold Value: {threshold}")
    else:
        # default fallback
        print("Entered Fallback threshold .1")
        threshold = 0.10  
        print(f"Default Threshold Value {threshold}")

    # 2. Check if too many rows would be removed
    actual_loss = (row_nan_dataframe > threshold).mean()
    print(f"Actual Loss Value: {actual_loss}")
    
    if actual_loss > max_loss_allowed:
        print(f"Entered Fallback Quantile If")
        # Fallback to quantile
        fallback_thresh = row_nan_dataframe.quantile(1 - max_loss_allowed)
        print(f"Fallback Threshold is: {fallback_thresh}")
        print(f"max_nan_per_row value is {max_nan_per_row}")
        
        # 3. Clamp to a max nan % allowed per row (e.g., 30%)
        threshold = min(fallback_thresh, max_nan_per_row)
        print(f"Threshold clamp result is {threshold}")
    else:
        print(f"If Skipped: threshold is set by previous if check - method vs default - threshold: {threshold}")

    threshold_source = 'kmeans' if actual_loss <= max_loss_allowed else 'quantile_fallback'

    #### #### #### #### #### #### #### #### #### #### #### #### 

    # 4. Copy Dataframe
    row_nan_df = dataframe.copy()

    # 4. Label the Row Nan Metan Values - Calculate the per-row NaN fraction
    row_nan_df['row_nan_total_mean'] = row_nan_df.isna().mean(axis=1)

    # 5. Categorize the rows
    row_nan_df['row_nan_category'] = row_nan_df['row_nan_total_mean'].apply(
        lambda x: 'high' if x > threshold else 'low'
    )


    # 6. Split into low and high NaN rows
    low_row_nan_df = row_nan_df[row_nan_df['row_nan_category'] == 'low']
    high_row_nan_df = row_nan_df[row_nan_df['row_nan_category'] == 'high']

    print(f"Low NaN Dataframe has a shape of: {low_row_nan_df.shape}")
    print(f"High NaN Dataframe has a shape of: {high_row_nan_df.shape}")
    print()
    print(f"Threshold final value: {threshold}")
    print()
    print(f"Percentage to be removed is: {high_row_nan_df.shape[0]/ len(dataframe)}")

    # 7. Identify common columns with no (or little) missing data
    row_nan_common_columns = []
    if return_common_columns or plot_graph:
        print("Return Common Columns - Enabled")
        if allow_noise:
            print(f"Generating with allowed noise in column matching, noise threshold is: {noise_threshold}")
            common_columns = (
                (low_row_nan_df.isna().mean() <= noise_threshold) & 
                (high_row_nan_df.isna().mean() <= noise_threshold)
            )
        else:
            print("Generating with zero allowed noise")
            common_columns = (
                (low_row_nan_df.isna().sum() == 0) & 
                (high_row_nan_df.isna().sum() == 0)
            )
        # Exclude EDA columns
        common_columns = common_columns.drop(['row_nan_total_mean', 'row_nan_category'], errors='ignore')
        row_nan_common_columns = common_columns[common_columns == True].index.tolist()

    # Optionally plot graphs
    if plot_graph:
        print("Generating Graph Output")
        for column in row_nan_common_columns:
            plot_countplot_compare_row_nan_dist_per_column(low_row_nan_df, high_row_nan_df, column)
            

    # 8. Return results
    if return_common_columns:
        return low_row_nan_df, high_row_nan_df, row_nan_common_columns, threshold_source, threshold
    else:
        return low_row_nan_df, high_row_nan_df, threshold_source, threshold

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


from WGU_D499_P2_DCook.dataset import write_checkpoints

def export_high_row_nan(dataframe, save_path, save_name):

    write_checkpoints(save_path, save_name, dataframe)



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def safe_remove(lst, item):
    if item in lst:
        lst.remove(item)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def apply_if_column_exists(
    dataframe,
    column_names,
    function,
    remove_if_missing=False,
    remove_from_lists=None,
    apply_per_column=True,
    *args,
    **kwargs
):
    """
    Apply func(df, *args, **kwargs) if col_name exists in df.columns.
    
    If col_name does NOT exist and remove_if_missing is True,
    remove col_name from all lists provided in remove_from_lists.
    
    - remove_from_lists: list of lists to remove col_name from.
    """
    if isinstance(column_names, str):
        column_names = [column_names]

    existing_columns = []
    
    for column_name in column_names:
        if column_name in dataframe.columns:
            existing_columns.append(column_name)

    missing_columns = set(column_names) - set(existing_columns)


    if remove_if_missing and remove_from_lists:
        for column_name in missing_columns:
            for lst in remove_from_lists:
                safe_remove(lst, column_name)


    if apply_per_column:
        for column_name in existing_columns:
            dataframe = function(dataframe, *args, **kwargs)
    else:

        if existing_columns:
            dataframe = function(dataframe, *args, **kwargs)

    return dataframe







#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


if __name__ == "__main__":
    app()
