from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pathlib as Path

from sklearn.cluster import KMeans
from kneed import KneeLocator


from WGU_D499_P2_DCook.config import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, ARCHIVED_DATA_DIR

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

from sklearn.cluster import KMeans


def compute_average_within_cluster_distance(pca_data, clusters_range, step_interval=2):
    """
    Compute the average within-cluster distance for KMeans clustering.
    
    Parameters:
    - pca_data: The input pca data for clustering.
    - clusters_range: Tuple containing the minimum and maximum number of clusters to evaluate.
    - step_interval: The step interval for the number of clusters.
    
    Returns:
    - The average within-cluster distance as a list
    """

    model_score = []

    start, end = clusters_range
    clusters = range(start, end + 1, step_interval)

    for i in clusters:
    
        kmeans = KMeans(i, random_state=5654).fit(pca_data)

        model_score.append(abs(kmeans.score(pca_data)))


    return model_score

    
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



def get_Kmeans_scores(pca_data, cluster_range, step_interval = 1):
    
    model_scores = []
    
    start, end = cluster_range
    clusters_range = range(start, end + 1, step_interval)


    print("Performing K-Means clustering")
    print("Given range min:{}, max:{}, step:{}".format(start, end, step_interval))


    for i in clusters_range:
        
        print("\nTraining for n_clusters: ", i)

        
        kmeans = KMeans(i, random_state=5654)
        model = kmeans.fit(pca_data)
        model_scores.append(abs(model.score(pca_data)))
        
        print("Done! Score: ", model_scores[-1])

        
    return model_scores, clusters_range

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

from kneed import KneeLocator


def find_optimal_k_knee(model_scores, cluster_range):
    """
    Find the optimal number of clusters using the KneeLocator.
    
    Parameters:
    - scores: List of scores for each number of clusters.
    - range_: Range of cluster numbers.
    
    Returns:
    - The optimal number of clusters according to KneeLocator.
    """


    # Use KneeLocator to find the knee point
    knee = KneeLocator(cluster_range, model_scores, curve='convex', direction='decreasing')
    
    optimal_k = knee.knee

    print(f"Optimal number of clusters according to KneeLocator: {optimal_k}")

    return optimal_k


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def generate_cluster_comparision_dataframe(dataframe_list, labels_list, dataset_names=None, baseline_dataset=None):
    """
    Combine multiple dataframes into a single dataframe for comparison.
    
    Parameters:
    - dataframe_list: List of dataframes to combine.
    - labels_list: List of labels corresponding to each dataframe.
    - dataset_names: Optional list of names for each dataset. If not provided, defaults to 'dataset_1', 'dataset_2', etc.
    
    Returns:
    - A single dataframe with an additional 'cluster' and 'dataset' column for comparison.
    """
    
    if not isinstance(dataframe_list, list):
        dataframe_list = [dataframe_list]

    if not isinstance(labels_list, list):
        labels_list = [labels_list]

    if len(dataframe_list) != len(labels_list):
        raise ValueError("The length of dataframe_list and labels_list must be the same.")
    

    if dataset_names is None:
        dataset_names = [f"dataset_{i+1}" for i in range(len(dataframe_list))]

    if len(dataframe_list) != len(dataset_names):
        raise ValueError("The length of dataframe_list and dataset_names must be the same.")
    
    dataframes_with_lables = []

    for dataframe, label, dataset_name in zip(dataframe_list, labels_list, dataset_names):
        
        dataframe_copy = dataframe.copy()
        dataframe_copy['cluster'] = label
        dataframe_copy['dataset'] = dataset_name

        dataframes_with_lables.append(dataframe_copy)

    if not dataframes_with_lables:
        raise ValueError("No dataframes to concatenate. Please provide at least one dataframe.")
    
    if len(dataframes_with_lables) == 1:
        return dataframes_with_lables[0]
    
    combinded_clusters_dataframe = pd.concat(dataframes_with_lables, ignore_index=True)

    # End of combining clusters dataframe functionality
    # Start of cluster comparison functionality

    if combinded_clusters_dataframe['dataset'].nunique() >= 2:
        logger.warning("More than two datasets found. Proceeding with cluster comparison.") 

        cluster_count = combinded_clusters_dataframe.groupby(['dataset', 'cluster']).size().unstack(fill_value=0)
        cluster_count = cluster_count.fillna(0)

        proportions = cluster_count.div(cluster_count.sum(axis=1), axis=0).fillna(0)
        proportions = proportions.fillna(0)

        cluster_comparision_dataframe_list = []

        if baseline_dataset is None:
            baseline_dataset = dataset_names[0]

        if baseline_dataset not in cluster_count.index:
            raise ValueError(f"Baseline dataset '{baseline_dataset}' not found in dataset_names")
        
        baseline_dataset_dataframe = pd.DataFrame({
            'cluster': cluster_count.columns,
            'dataset': baseline_dataset,
            f'{baseline_dataset} Count': cluster_count.loc[baseline_dataset],
            f'{baseline_dataset} Proportion': proportions.loc[baseline_dataset],
            'Difference': 0,
            'Ratio': 1,
            'Compared To:': baseline_dataset   
            })
        
        cluster_comparision_dataframe_list.append(baseline_dataset_dataframe)

        for dataset_name in cluster_count.index:
            if dataset_name == baseline_dataset:
                continue
            
            comparison_dataframe = pd.DataFrame({
                'cluster': cluster_count.columns,
                'dataset': dataset_name,
                f'{dataset_name} Count': cluster_count.loc[dataset_name],
                f'{dataset_name} Proportion': proportions.loc[dataset_name],
                'Difference': proportions.loc[dataset_name] - proportions.loc[baseline_dataset],
                'Ratio': proportions.loc[dataset_name] / proportions.loc[baseline_dataset].replace(0, pd.NA),
                'Compared To': baseline_dataset
            })
            
            cluster_comparision_dataframe_list.append(comparison_dataframe)

        

        cluster_comparision_dataframe = pd.concat(cluster_comparision_dataframe_list, ignore_index=True).sort_values(['Compared To', 'cluster'])

        return combinded_clusters_dataframe, cluster_comparision_dataframe
    
    else:
        raise ValueError("At least two datasets are required for comparison. Please provide more than one dataset.")
    

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def generate_cluster_comparision_dataframe_v2(dataframe_list, labels_list, dataset_names=None, baseline_dataset=None):
    """
    Combine multiple dataframes into a single dataframe for comparison.
    
    Parameters:
    - dataframe_list: List of dataframes to combine.
    - labels_list: List of labels corresponding to each dataframe.
    - dataset_names: Optional list of names for each dataset. If not provided, defaults to 'dataset_1', 'dataset_2', etc.
    
    Returns:
    - A single dataframe with an additional 'cluster' and 'dataset' column for comparison.
    """
    
    if not isinstance(dataframe_list, list):
        dataframe_list = [dataframe_list]

    if not isinstance(labels_list, list):
        labels_list = [labels_list]

    if len(dataframe_list) != len(labels_list):
        raise ValueError("The length of dataframe_list and labels_list must be the same.")
    

    if dataset_names is None:
        dataset_names = [f"dataset_{i+1}" for i in range(len(dataframe_list))]

    if len(dataframe_list) != len(dataset_names):
        raise ValueError("The length of dataframe_list and dataset_names must be the same.")
    
    dataframes_with_lables = []

    for dataframe, label, dataset_name in zip(dataframe_list, labels_list, dataset_names):
        
        dataframe_copy = dataframe.copy()
        dataframe_copy['cluster'] = label
        dataframe_copy['dataset'] = dataset_name

        dataframes_with_lables.append(dataframe_copy)

    if not dataframes_with_lables:
        raise ValueError("No dataframes to concatenate. Please provide at least one dataframe.")
    
    if len(dataframes_with_lables) == 1:
        return dataframes_with_lables[0]
    
    combinded_clusters_dataframe = pd.concat(dataframes_with_lables, ignore_index=True)

    # End of combining clusters dataframe functionality
    # Start of cluster comparison functionality

    if combinded_clusters_dataframe['dataset'].nunique() >= 2:
        logger.warning("More than two datasets found. Proceeding with cluster comparison.") 

        cluster_count = combinded_clusters_dataframe.groupby(['dataset', 'cluster']).size().unstack(fill_value=0)
        cluster_count = cluster_count.fillna(0)

        proportions = cluster_count.div(cluster_count.sum(axis=1), axis=0).fillna(0)
        proportions = proportions.fillna(0)

        cluster_comparision_dataframe_list = []

        if baseline_dataset is None:
            baseline_dataset = dataset_names[0]

        if baseline_dataset not in cluster_count.index:
            raise ValueError(f"Baseline dataset '{baseline_dataset}' not found in dataset_names")
        
        baseline_dataset_dataframe = pd.DataFrame({
            'cluster': cluster_count.columns,
            'dataset': baseline_dataset,
            'count': cluster_count.loc[baseline_dataset],
            'proportion': proportions.loc[baseline_dataset],
            'difference_to_baseline': 0,
            'ratio_to_baseline': 1,
            'compared_to': baseline_dataset   
            })
        
        cluster_comparision_dataframe_list.append(baseline_dataset_dataframe)

        for dataset_name in cluster_count.index:
            if dataset_name == baseline_dataset:
                continue
            
            comparison_dataframe = pd.DataFrame({
                'cluster': cluster_count.columns,
                'dataset': dataset_name,
                'count': cluster_count.loc[dataset_name],
                'proportion': proportions.loc[dataset_name],
                'difference_to_baseline': proportions.loc[dataset_name] - proportions.loc[baseline_dataset],
                'ratio_to_baseline': proportions.loc[dataset_name] / proportions.loc[baseline_dataset].replace(0, pd.NA),
                'compared_to': baseline_dataset
            })
            
            cluster_comparision_dataframe_list.append(comparison_dataframe)

        

        cluster_comparision_dataframe = pd.concat(cluster_comparision_dataframe_list, ignore_index=True).sort_values(['compared_to', 'cluster'])

        return combinded_clusters_dataframe, cluster_comparision_dataframe
    
    else:
        raise ValueError("At least two datasets are required for comparison. Please provide more than one dataset.")
    


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



def show_top_features_per_centroid(centroids_dataframe, top_x=10, only_clusters=None):
    feature_dataframe = centroids_dataframe.copy()
    if 'cluster' in feature_dataframe.columns:
        feature_dataframe = feature_dataframe.drop(columns=['cluster'])

    if only_clusters is not None:
        feature_dataframe = feature_dataframe[feature_dataframe.index.isin(only_clusters)]

    for cluster_idx, row in feature_dataframe.iterrows():
        print(f"\nCluster {cluster_idx} - Top {top_x} POSITIVE features:")
        top_pos = row.sort_values(ascending=False).head(top_x)
        display(top_pos)

        print(f"\nCluster {cluster_idx} - Top {top_x} NEGATIVE features:")
        top_neg = row.sort_values().head(top_x)
        display(top_neg)

        print("\n" + "-"*60)



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def filter_clusters_by_threshold(dataframe, method='Difference', threshold=0.2, absolute=True):
    """
    Filter clusters where the ratio or difference deviates from baseline by at least threshold.

    Parameters:
    - dataframe: DataFrame with at least 'cluster' and the column to filter on (e.g., 'Ratio' or 'Difference')
    - method: What method to use for filtering ('Ratio' or 'Difference')
    - threshold: Threshold value to decide significance (e.g. 0.2 means 20% difference or ratio deviation)
    - absolute: Whether to consider absolute value (default True). For ratio threshold, usually absolute=False.



    Returns:
    - Threshold Comparision DataFrame containing only rows meeting threshold criteria.
    """

    if method == 'Ratio':
        
        threshold_comparision_dataframe = dataframe[(dataframe['Ratio'] <= (1 - threshold)) | (dataframe['Ratio'] >= (1 + threshold))]

    elif method == 'Difference':
        
        if absolute:
            threshold_comparision_dataframe = dataframe[dataframe['Difference'].abs() >= threshold]
        else:
            threshold_comparision_dataframe = dataframe[dataframe['Difference'] >= threshold]
    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'Ratio' or 'Difference'.")

    return threshold_comparision_dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### ####



# if i want to remove basline before filtering, you can do so like this:
#non_baseline_df = comparison_dataframe[comparison_dataframe['Ratio'] != 1]  # or 'Difference' != 0
#filtered_clusters = filter_clusters_by_threshold(non_baseline_dataframe, method='Ratio', threshold=0.2)


# If i want to filter by a specific baseline dataset, you can do so like this:
#baseline_name = 'population'  # or 'dataset_1' or whatever you set
#subset = comparison_dataframe[comparison_dataframe['Compared To:'] == baseline_name]


# filtering per group then combineding
#group_filtered_list = []

#for baseline in comparison_datframe['Compared To:'].unique():
#    subset = comparison_dataframe[comparison_dataframe['Compared To:'] == baseline]
#    filtered_subset = filter_clusters_by_threshold(subset, method='Ratio', threshold=0.2)
#    group_filtered_list.append(filtered_subset)

#filtered_all = pd.concat(group_filtered_list, ignore_index=True)




def filter_clusters_by_threshold_v2(dataframe, method='Difference', threshold=0.2, absolute=True):
    """
    Filter clusters where the ratio or difference deviates from baseline by at least threshold.

    Parameters:
    - dataframe: DataFrame with at least 'cluster' and the column to filter on (e.g., 'Ratio' or 'Difference')
    - method: What method to use for filtering ('Ratio' or 'Difference')
    - threshold: Threshold value to decide significance (e.g. 0.2 means 20% difference or ratio deviation)
    - absolute: Whether to consider absolute value (default True). For ratio threshold, usually absolute=False.



    Returns:
    - Threshold Comparision DataFrame containing only rows meeting threshold criteria.
    """

    if method == 'Ratio':
        
        threshold_comparision_dataframe = dataframe[(dataframe['ratio_to_baseline'] <= (1 - threshold)) | (dataframe['ratio_to_baseline'] >= (1 + threshold))]

    elif method == 'Difference':
        
        if absolute:
            threshold_comparision_dataframe = dataframe[dataframe['difference_to_baseline'].abs() >= threshold]
        else:
            threshold_comparision_dataframe = dataframe[dataframe['difference_to_baseline'] >= threshold]
    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'Ratio' or 'Difference'.")

    return threshold_comparision_dataframe

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def filter_clusters_by_threshold_with_type(comparision_dataframe, method='Difference', threshold=0.2, absolute=True, filter_type='both', exclude_baseline=False):
    """
    Filter clusters by threshold and optionally by cluster type.
    
    Parameters:
    - comparision_dataframe: DataFrame containing cluster comparison data.
    - method: Method to use for filtering ('Ratio' or 'Difference').
    - threshold: Threshold value for filtering.
    - absolute: Whether to consider absolute values (default True).
    - filter_type: Whether the filter is 'over', 'under', or 'both' (default 'both') to the threshold
    - exlude_baseline: Whether to exclude the baseline dataset from the filtering (default False).
    
    Returns:
    - Filtered DataFrame based on the specified criteria.
    """
    
    type_filtered_comparision_dataframe = comparision_dataframe.copy()

    if exclude_baseline:
        if method == "Ratio":
            type_filtered_comparision_dataframe = type_filtered_comparision_dataframe[type_filtered_comparision_dataframe['Ratio:'] != 1]
        elif method == "Difference":
            type_filtered_comparision_dataframe = type_filtered_comparision_dataframe[type_filtered_comparision_dataframe['Difference'] != 0] 

    if filter_type == 'over':
        type_filtered_comparision_dataframe = type_filtered_comparision_dataframe[type_filtered_comparision_dataframe[method] >= threshold]
    elif filter_type == 'under':
        type_filtered_comparision_dataframe = type_filtered_comparision_dataframe[type_filtered_comparision_dataframe[method] <= threshold]
    elif filter_type == 'both':
        type_filtered_comparision_dataframe = type_filtered_comparision_dataframe[(type_filtered_comparision_dataframe[method] >= threshold) | (type_filtered_comparision_dataframe[method] >= threshold)]
    else:
        raise ValueError(f"Unsupported filter_type '{filter_type}'. Use 'over', 'under', or 'both'.")  
    
    return type_filtered_comparision_dataframe


#### 

def filter_clusters_by_threshold_with_type_v2(comparision_dataframe, method='Difference', threshold=0.2, absolute=True, filter_type='both', exclude_baseline=False):
    """
    Filter clusters by threshold and optionally by cluster type.
    
    Parameters:
    - comparision_dataframe: DataFrame containing cluster comparison data.
    - method: Method to use for filtering ('Ratio' or 'Difference').
    - threshold: Threshold value for filtering.
    - absolute: Whether to consider absolute values (default True).
    - filter_type: Whether the filter is 'over', 'under', or 'both' (default 'both') to the threshold
    - exlude_baseline: Whether to exclude the baseline dataset from the filtering (default False).
    
    Returns:
    - Filtered DataFrame based on the specified criteria.
    """
    
    type_filtered_comparision_dataframe = comparision_dataframe.copy()

    if exclude_baseline:
        if method == "Ratio":
            type_filtered_comparision_dataframe = type_filtered_comparision_dataframe[type_filtered_comparision_dataframe['ratio_to_baseline:'] != 1]
        elif method == "Difference":
            type_filtered_comparision_dataframe = type_filtered_comparision_dataframe[type_filtered_comparision_dataframe['difference_to_baseline'] != 0] 

    method_column = 'ratio_to_baseline' if method == 'Ratio' else 'difference_to_baseline'

    if filter_type == 'over':
        type_filtered_comparision_dataframe = type_filtered_comparision_dataframe[type_filtered_comparision_dataframe[method_column] >= threshold]
    elif filter_type == 'under':
        type_filtered_comparision_dataframe = type_filtered_comparision_dataframe[type_filtered_comparision_dataframe[method_column] <= threshold]
    elif filter_type == 'both':
        type_filtered_comparision_dataframe = type_filtered_comparision_dataframe[(type_filtered_comparision_dataframe[method_column] >= threshold) | (type_filtered_comparision_dataframe[method_column] >= threshold)]
    else:
        raise ValueError(f"Unsupported filter_type '{filter_type}'. Use 'over', 'under', or 'both'.")  
    
    return type_filtered_comparision_dataframe

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



if __name__ == "__main__":
    app()
