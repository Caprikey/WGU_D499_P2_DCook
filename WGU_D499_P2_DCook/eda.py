from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from WGU_D499_P2_DCook.config import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

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



def column_nan_eda_total_and_percentage(dataframe):
    """
    Calculates the total missing values per column and calculates the total percentage of missing values per column returns two series
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze.

    References:
    - https://stackoverflow.com/questions/72083258/how-to-plot-distribution-of-missing-values-in-a-dataframe
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Calculate the percentage of NaN values per column
    #column_nan_eda_percentage_dataframe = dataframe.isnull().mean()

    dataframe_length = len(dataframe)

    # Calculate Total Missing Per Column

    column_nan_eda_total_series = dataframe.isna().sum()

    column_nan_eda_total_series = column_nan_eda_total_series[column_nan_eda_total_series > 0]


    # Calculates Percentage Missing Per Column 

    column_nan_eda_percentage_series = column_nan_eda_total_series / dataframe_length 
    
    column_nan_eda_percentage_series = column_nan_eda_percentage_series.sort_values(ascending=False)


    return column_nan_eda_total_series, column_nan_eda_percentage_series



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



invalid_level = []


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# Version 2 of the function
def eda_identify_outlier_columns(dataframe, features_summary_dataframe, use_threshold=False, threshold=.2, severity_levels_to_flag = None):

    VALID_SEVERITY_LEVELS = [
        'Very Low', 'Low', 'Medium Low', 'Medium',
        'Medium High', 'High', 'Very High'
    ]

    if not use_threshold:
        if severity_levels_to_flag is not None:
            invalid_levels = []
            
            for level in severity_levels_to_flag:
                if level not in VALID_SEVERITY_LEVELS:
                    invalid_level.append(level)
                
            if len(invalid_levels) > 0:
                raise ValueError(
                    f"Invalid severity level(s): {invalid_levels}. "
                    f"Allowed values are: {VALID_SEVERITY_LEVELS}")

    
    nan_eda_df = dataframe.copy()
    
    nan_counts = nan_eda_df.isna().sum()

    column_nan_eda_df = nan_counts[nan_counts > 0].sort_values(ascending=False).to_frame().reset_index()
    
    column_nan_eda_df.columns = ['attribute', 'total_nan']

    column_nan_eda_df['nan_percentage'] = column_nan_eda_df['total_nan'] / len(dataframe) * 100

    # Merging features information
    column_nan_eda_df = column_nan_eda_df.merge(features_summary_dataframe[['attribute', 'information_level']], on='attribute', how='left')

    column_nan_eda_df['missing_score'] = column_nan_eda_df['nan_percentage'] / 100

    # Calculating a mean
    percentage_mean = column_nan_eda_df['missing_score'].mean()

    # Severity category function
    def categorize_severity(score, mean):
        if score >= mean * 2.25:
            return 'Very High'
        elif score >= mean * 1.75:
            return 'High'
        elif score >= mean * 1.25:
            return 'Medium High'
        elif score >= mean * 1:
            return 'Medium'
        elif score >= mean * 0.75:
            return 'Medium Low'
        elif score >= mean * 0.25:
            return 'Low'
        else:
            return 'Very Low'

    column_nan_eda_df['severity_level'] = column_nan_eda_df['missing_score'].apply(
        lambda x: categorize_severity(x, percentage_mean))

    column_nan_eda_df = column_nan_eda_df.rename(columns={'attribute':'column_name'})
    
    if threshold > 1:
        threshold = threshold / 100

    if use_threshold:
        outlier_columns_df = column_nan_eda_df[column_nan_eda_df['missing_score'] >= threshold]
    else:
        if severity_levels_to_flag is None:
            severity_levels_to_flag = ['Medium High', 'High', 'Very High']

        outlier_columns_df = column_nan_eda_df[column_nan_eda_df['severity_level'].isin(severity_levels_to_flag)]
        
        #outlier_columns_df = column_nan_eda_df[column_nan_eda_df['severity_level'].isin(['Medium High', 'High', 'Very High'])]
        
    outlier_column_name_list = outlier_columns_df['column_name'].to_list()

    print(f"Mean of missing score is {percentage_mean}")
    
    return column_nan_eda_df, outlier_column_name_list



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



def get_invalid_level_eda_identify_outlier_columns():
    """
    Returns the list of invalid severity levels encountered during the last call to eda_identify_outlier_columns_v2.
    """
    global invalid_level

    if len(invalid_level) == 0:
        return None
    else:
        return invalid_level


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def eda_identify_row_nan(dataframe):

    nan_row_dataframe = dataframe.copy()

    original_dataframe_len = len(dataframe)
    
    nan_row_dataframe['row_nan_total_mean'] = nan_row_dataframe.isna().mean(axis=1)

    nan_row_dataframe['row_nan_total_sum'] = nan_row_dataframe.isna().sum(axis=1)


    print("Proportion of Missing Values per Row:")  
    print(nan_row_dataframe['row_nan_total_mean'].describe())   
    print("\n" + "-"*50 + "\n")
    

    print("Count of Missing Values per Row:") 
    print(nan_row_dataframe['row_nan_total_sum'].describe())

    return nan_row_dataframe, original_dataframe_len



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

from sklearn.cluster import KMeans
import numpy as np

def eda_row_nan_find_dynamic_nan_threshold(dataframe, num_clusters = 3, num_init = 'auto', rand_stat = 5654):
    
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

def eda_row_nan_divide_by_threshold(dataframe, 
                                    max_loss_allowed=0.20, 
                                    max_nan_per_row=0.30, 
                                    method='kmeans',
                                    return_common_columns=True, 
                                    allow_noise=False, 
                                    noise_threshold=0.05, 
                                    plot_graph=False
                                    ):
    
    eda_row_nan_dataframe = dataframe.isna().mean(axis=1)

    # 1. Try dynamic threshold from KMeans
    if method == 'kmeans':
        print("Entered Method")
        threshold = eda_row_nan_find_dynamic_nan_threshold(dataframe)
        print(f"Threshold Value: {threshold}")
    else:
        # default fallback
        print("Entered Fallback threshold .1")
        threshold = 0.10  
        print(f"Default Threshold Value {threshold}")

    # 2. Check if too many rows would be removed
    actual_loss = (eda_row_nan_dataframe > threshold).mean()
    print(f"Actual Loss Value: {actual_loss}")
    
    if actual_loss > max_loss_allowed:
        print(f"Entered Fallback Quantile If")
        # Fallback to quantile
        fallback_thresh = eda_row_nan_dataframe.quantile(1 - max_loss_allowed)
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
    eda_row_nan_df = dataframe.copy()

    # 4. Label the Row Nan Metan Values - Calculate the per-row NaN fraction
    eda_row_nan_df['row_nan_total_mean'] = eda_row_nan_df.isna().mean(axis=1)

    # 5. Categorize the rows
    eda_row_nan_df['row_nan_category'] = eda_row_nan_df['row_nan_total_mean'].apply(
        lambda x: 'high' if x > threshold else 'low'
    )

    # Split into low and high NaN rows
    eda_low_row_nan_df = eda_row_nan_df[eda_row_nan_df['row_nan_category'] == 'low']
    eda_high_row_nan_df = eda_row_nan_df[eda_row_nan_df['row_nan_category'] == 'high']

    print(f"Low NaN Dataframe has a shape of: {eda_low_row_nan_df.shape}")
    print(f"High NaN Dataframe has a shape of: {eda_high_row_nan_df.shape}")
    print()
    print(f"Threshold final value: {threshold}")
    print()
    print(f"Percentage to be removed is: {eda_high_row_nan_df.shape[0]/ len(dataframe)}")

    # Identify common columns with no (or little) missing data
    eda_row_nan_common_columns = []
    if return_common_columns or plot_graph:
        print("Return Common Columns - Enabled")
        if allow_noise:
            print(f"Generating with allowed noise in column matching, noise threshold is: {noise_threshold}")
            common_columns = (
                (eda_low_row_nan_df.isna().mean() <= noise_threshold) & 
                (eda_high_row_nan_df.isna().mean() <= noise_threshold)
            )
        else:
            print("Generating with zero allowed noise")
            common_columns = (
                (eda_low_row_nan_df.isna().sum() == 0) & 
                (eda_high_row_nan_df.isna().sum() == 0)
            )
        # Exclude EDA columns
        common_columns = common_columns.drop(['row_nan_total_mean', 'row_nan_category'], errors='ignore')
        eda_row_nan_common_columns = common_columns[common_columns == True].index.tolist()

    # Optionally plot graphs
    if plot_graph:
        print("Generating Graph Output")
        for column in eda_row_nan_common_columns:
            #compare_row_nan_distribution_per_column(eda_low_row_nan_df, eda_high_row_nan_df, column)
            plot_countplot_compare_row_nan_dist_per_column(eda_low_row_nan_df, eda_high_row_nan_df, column)

    # Return results
    if return_common_columns:
        return eda_low_row_nan_df, eda_high_row_nan_df, eda_row_nan_common_columns, threshold_source, threshold
    
    else:
        return eda_low_row_nan_df, eda_high_row_nan_df, threshold_source, threshold

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def countplot_data_summary(dataframe_low, dataframe_high, column_name, normalize=False):
    """
    Returns count data for a column from two dataframes instead of plotting.

    Args:
        dataframe_low (pd.DataFrame): First subset of data.
        dataframe_high (pd.DataFrame): Second subset of data.
        column_name (str): Column to analyze.
        normalize (bool): Whether to return relative frequencies instead of raw counts.

    Returns:
        pd.DataFrame: Combined count summary with counts from both subsets.
    """
    import pandas as pd

    low_counts = dataframe_low[column_name].value_counts(normalize=normalize).rename('low_subset')
    high_counts = dataframe_high[column_name].value_counts(normalize=normalize).rename('high_subset')

    # Combine into one DataFrame
    combined = pd.concat([low_counts, high_counts], axis=1).fillna(0).astype(float)

    return combined.sort_index()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def countplot_data_summary_multicol(dataframe_low, dataframe_high, column_list, normalize=False):
    """
    Returns count data for multiple columns from two dataframes instead of plotting.

    Args:
        dataframe_low (pd.DataFrame): First subset of data.
        dataframe_high (pd.DataFrame): Second subset of data.
        column_list (list): List of column names to analyze.
        normalize (bool): Whether to return relative frequencies instead of raw counts.

    Returns:
        dict: Dictionary where each key is a column and the value is a DataFrame with counts.
    """
    import pandas as pd

    summary_dict = {}

    for column in column_list:
        low_counts = dataframe_low[column].value_counts(normalize=normalize).rename('low_subset')
        high_counts = dataframe_high[column].value_counts(normalize=normalize).rename('high_subset')

        combined = pd.concat([low_counts, high_counts], axis=1).fillna(0).astype(float)
        summary_dict[column] = combined.sort_index()

    return summary_dict

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Not Using

from scipy.stats import chi2_contingency
import pandas as pd

def chi2_comparisons(summary_dict):
    chi2_results = {}

    for col, dataframe in summary_dict.items():
        contingency = dataframe.fillna(0).astype(int).T.values  # shape: (2, n_categories)
        try:
            chi2, p, dof, expected = chi2_contingency(contingency)
            chi2_results[col] = {'chi2_stat': chi2, 'p_value': p}
        except ValueError:
            chi2_results[col] = {'chi2_stat': None, 'p_value': None}

    return pd.DataFrame.from_dict(chi2_results, orient='index').sort_values(by='p_value')

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def calculate_proportion_diff(summary_dict):

    diff_dict = {}
    
    for column, dataframe in summary_dict.items():
    
        total_low = dataframe["low_subset"].sum()
        total_high = dataframe["high_subset"].sum()
        
        dataframe_copy = dataframe.copy()

        dataframe_copy["low_subset"] = dataframe["low_subset"].fillna(0)
        dataframe_copy["high_subset"] = dataframe["high_subset"].fillna(0)

        dataframe_copy["low_pct"] = dataframe["low_subset"] / total_low
        dataframe_copy["high_pct"] = dataframe["high_subset"] / total_high
        dataframe_copy["pct_diff"] = (dataframe_copy["high_pct"] - dataframe_copy["low_pct"]).abs()

        diff_dict[column] = dataframe_copy.sort_values("pct_diff", ascending=False)

    return diff_dict

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def print_proportion_diff(diff_dict, top_n=5):
    for column, dataframe in diff_dict.items():
        
        print(f"\n=== Column: {column} ===")
        # Select top N rows with largest difference
        top_diff = dataframe.head(top_n).copy()
        
        # Format percentages 
        top_diff["low_pct"] = (top_diff["low_pct"] * 100).map("{:.2f}%".format)
        top_diff["high_pct"] = (top_diff["high_pct"] * 100).map("{:.2f}%".format)
        top_diff["pct_diff"] = (top_diff["pct_diff"] * 100).map("{:.2f}%".format)
        
        print(top_diff[["low_subset", "high_subset", "low_pct", "high_pct", "pct_diff"]])
        print("-" * 50)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def combine_proportion_diff_to_dataframe(diff_dict, top_n=None):
    import pandas as pd

    combined_rows = []

    for column, dataframe in diff_dict.items():
        
        dataframe_copy = dataframe[["low_subset", "high_subset", "low_pct", "high_pct", "pct_diff"]].copy()

        
        dataframe_copy = dataframe_copy.reset_index()
        dataframe_copy = dataframe_copy.rename(columns={dataframe_copy.columns[0]: "column_value"})

        
        dataframe_copy["column_name"] = column

        # Limit to top_n rows if requested
        if top_n is not None:
            dataframe_copy = dataframe_copy.sort_values("pct_diff", ascending=False).head(top_n)

        combined_rows.append(dataframe_copy)

    # Combine all rows
    combined_dataframe = pd.concat(combined_rows, ignore_index=True)

    # Reorder columns for clarity
    combined_dataframe = combined_dataframe[["column_name", "column_value", "low_subset", "high_subset", "low_pct", "high_pct", "pct_diff"]]

    return combined_dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

binary_set = set()
multi_set = set()
error_set = set()

def eda_column_identify_binary_multilevel_category(dataframe, column_list, binary_set, multi_set, error_set):

    for index, row in column_list.iterrows():
        column = row['attribute']
        
        if column in dataframe.columns:
            unique_values = dataframe[column].nunique()
            
            if unique_values == 2:
                binary_set.add(column)
            elif unique_values > 2:
                multi_set.add(column)
            else:
                error_set.add(column)

    return binary_set, multi_set, error_set


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def eda_get_missing_percentages(prescaled_dataframe, features_dataframe):

    get_percentages = (prescaled_dataframe.isnull().sum(axis=0) / prescaled_dataframe.shape[0]).sort_values(ascending=False)

    non_zero_percentages = get_percentages[get_percentages > 0.0]

    missing_data_dataframe = non_zero_percentages.reset_index()

    missing_data_dataframe.columns = ['column_name', 'missing_percentage']

    missing_data_dataframe['missing_percentage'] = missing_data_dataframe['missing_percentage'].apply(lambda x: f"{x:.2%}")

    #for index, row in missing_data_dataframe.iterrows():
    #    missing_data_dataframe['total_missing'] = azdias_general_demographics[azdias_general_demographics[row[0]].isnull().sum(axis=0)]

    missing_data_dataframe['total_missing'] = missing_data_dataframe['column_name'].apply(
        lambda col: prescaled_dataframe[col].isnull().sum())

    missing_data_dataframe = missing_data_dataframe.merge(
        features_dataframe[['attribute', 'type', 'information_level']],
        left_on='column_name',
        right_on='attribute',
        how='left'
    )

    missing_data_dataframe = missing_data_dataframe.drop(columns=['attribute'])

    missing_data_dataframe = missing_data_dataframe.rename(columns={'type': 'data_type'})

    return missing_data_dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



if __name__ == "__main__":
    app()
