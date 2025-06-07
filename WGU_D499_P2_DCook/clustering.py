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







#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


if __name__ == "__main__":
    app()
