from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns


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


def plot_hist_column_nan_percentage(column_nan_eda_percentage_dataframe):
    """
    Plots the percentage of NaN values per column in a DataFrame.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Plotting
    plt.figure(figsize=(10, 8))

    column_nan_eda_percentage_dataframe.hist();

    plt.title("Histogram - Number of Columns Per Pecentage of NAN Values")
    plt.ylabel("Number of Columns")
    plt.xlabel("Percentage of NAN Values per Column total (Decimal)")
    plt.yticks(np.arange(0, 41, 1)) 
    plt.xticks(np.arange(0, 1.05, .05), rotation=45)
    plt.xlim(0, 1)  
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Save the plot
    output_path = FIGURES_DIR / "column_nan_eda_percentage_hist_plot.png"
    plt.savefig(output_path)
    
    logger.info(f"Plot saved to {output_path}")

    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def plot_bar_column_nan_percentage(dataframe):
    """
    Plots a bar chart of the percentage of NaN values per column in a DataFrame.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # Calculate the percentage of NaN values per column
    nan_percentage = dataframe.isnull().mean()

    nan_percentage = nan_percentage.sort_values(ascending=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 8))

    ax.bar(nan_percentage.index, nan_percentage.values * 100)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
    
    ax.set_xticks(range(len(nan_percentage.index)))
    ax.set_xticklabels(nan_percentage.index, rotation=90)
    ax.set_ylim(0, 100)



    # Save the plot
    output_path = FIGURES_DIR / "column_nan_eda_percentage_bar_plot.png"
    plt.savefig(output_path)
    
    logger.info(f"Plot saved to {output_path}")

    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def plot_bar_eda_column_missing_severity(dataframe, top_n=30):
    
    # Sort and truncate
    sorted_df = dataframe.sort_values(by='missing_score', ascending=False).head(top_n)

    # Define color palette
    severity_palette = {
        'Very High': '#d73027',
        'High': '#fc8d59',
        'Medium High': '#fee08b',
        'Medium': '#ffffbf',
        'Medium Low': '#d9ef8b',
        'Low': '#91cf60',
        'Very Low': '#1a9850',
    }

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=sorted_df, 
        x='column_name', 
        y='nan_percentage', 
        hue='severity_level', 
        dodge=False, 
        palette=severity_palette
    )

    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {top_n} Columns by Missing Percentage')
    plt.ylabel('Missing %')
    plt.xlabel('Column Name')
    plt.legend(title='Severity Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()



    # Save the plot
    output_path = FIGURES_DIR / "column_nan_eda_severity_bar_plot.png"
    plt.savefig(output_path)
    
    logger.info(f"Plot saved to {output_path}")

    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


# This function performs EDA on the row-wise NaN counts of a DataFrame and plots two histograms, 
# One for total average per row
# One for total sum of NaN values per row 


def plot_hist_row_nan_eda(nan_row_eda_dataframe):

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    mean_values = nan_row_eda_dataframe['row_nan_total_mean']
    mean_bins = np.linspace(mean_values.min(), mean_values.max(), 30)

    #mean_counts, mean_bins, _ = axes[0].hist(mean_values, bins=30, edgecolor='black', color = 'skyblue')
    
    axes[0].hist(mean_values, bins=mean_bins, edgecolor='black', color = 'skyblue')
    axes[0].set_xlabel('Average Missing Totals per Row')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Histogram of Average Missing values per Row')
    axes[0].grid(True)

    sum_values = nan_row_eda_dataframe['row_nan_total_sum']
    sum_bins = range(0, sum_values.max() + 2) 

    #max_missing = sum_values.max()
    #sum_counts, sum_bins, _ = axes[1].hist(sum_values, bins=range(0, max_missing + 2), color='salmon', edgecolor='black', align='left')

    axes[1].hist(sum_values, bins=sum_bins, color='salmon', edgecolor='black', align='left')
    axes[1].set_xlabel('Count of Missing Values per Row')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Histogram: Missing Value Count per Row')
    axes[1].grid(True)

    plt.tight_layout()

    print("Proportion of Missing Values per Row:")  
    print(nan_row_eda_dataframe['row_nan_total_mean'].describe())   

    print("\n" + "-"*50 + "\n")
    
    print("Count of Missing Values per Row:") 
    print(nan_row_eda_dataframe['row_nan_total_sum'].describe())

    # Save the plot
    output_path = FIGURES_DIR / "plot_hist_row_nan_eda.png"
    plt.savefig(output_path)

    print("✅ Plot ready. Displaying...")
    plt.show()
    plt.close(fig)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


import matplotlib.pyplot as plt
import numpy as np

def plot_nan_proportion_histogram(mean_values):
    """
    Plot histogram of the proportion of missing values per row.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    mean_bins = np.linspace(mean_values.min(), mean_values.max(), 30)
    ax.hist(mean_values, bins=mean_bins, edgecolor='black', color='skyblue')

    ax.set_xlabel('Proportion of Missing Values per Row')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram: Missing Value Proportion per Row')
    ax.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


#### #### #### #### #### #### #### #### 


def plot_nan_count_histogram(sum_values):
    """
    Plot histogram of the count of missing values per row.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    sum_bins = range(0, int(sum_values.max()) + 2)
    ax.hist(sum_values, bins=sum_bins, edgecolor='black', color='salmon', align='left')

    ax.set_xlabel('Count of Missing Values per Row')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram: Missing Value Count per Row')
    ax.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


#### #### #### #### #### #### #### #### 


def plot_hist_row_nan_eda_v2(nan_row_eda_dataframe):
    """
    Wrapper function to call both histogram plotters.
    """
    print("✅ Plotting histograms for missing values per row...")
    plot_nan_proportion_histogram(nan_row_eda_dataframe['row_nan_total_mean'])
    plot_nan_count_histogram(nan_row_eda_dataframe['row_nan_total_sum'])


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


import matplotlib.pyplot as plt
import numpy as np

def plot_nan_proportion_histogram_side_by_side(ax, mean_values):
    """
    Plot histogram of the proportion of missing values per row in a side-by-side manner.
    """

    mean_bins = np.linspace(mean_values.min(), mean_values.max(), 30)
    ax.hist(mean_values, bins=mean_bins, edgecolor='black', color='skyblue')
    ax.set_xlabel('Proportion of Missing Values per Row')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram: Missing Value Proportion per Row')
    ax.grid(True)


#### #### #### #### #### #### #### #### 


def plot_nan_count_histogram_side_by_side(ax, sum_values):
    """
    Plot histogram of the count of missing values per row in a side-by-side manner.
    """

    sum_bins = range(0, int(sum_values.max()) + 2)
    ax.hist(sum_values, bins=sum_bins, edgecolor='black', color='salmon', align='left')
    ax.set_xlabel('Count of Missing Values per Row')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram: Missing Value Count per Row')
    ax.grid(True)


#### #### #### #### #### #### #### #### 


def plot_hist_row_nan_eda_side_by_side(nan_row_eda_dataframe):
    """
    Wrapper function to call both histogram plotters side by side
    """
    print("✅ Plotting histograms for missing values per row...")
    # This was replaced with the plot_nan_proportion_histogram_side_by_side and plot_nan_count_histogram_side_by_side functions above
    # To call them one after the other vertically do it this way without the stuff below.
    #plot_nan_proportion_histogram(nan_row_eda_dataframe['row_nan_total_mean'])
    #plot_nan_count_histogram(nan_row_eda_dataframe['row_nan_total_sum'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_nan_proportion_histogram_side_by_side(axes[0], nan_row_eda_dataframe['row_nan_total_mean'])
    plot_nan_count_histogram_side_by_side(axes[1], nan_row_eda_dataframe['row_nan_total_sum'])


    # This was replaced with the display_describe_column_comparison_side_by_side from the utils py file.
    #print("Proportion of Missing Values per Row:")  
    #print(nan_row_eda_dataframe['row_nan_total_mean'].describe())   

    #print("\n" + "-"*50 + "\n")
    
    #print("Count of Missing Values per Row:") 
    #print(nan_row_eda_dataframe['row_nan_total_sum'].describe())

    plt.tight_layout()
    plt.show()
    plt.close(fig)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# This one is called from in a loop on the notebook.

def plot_countplout_compare_row_nan_distribution_per_column(dataframe_low, dataframe_high, column_name):
    #figure_size = (14, 3)
    #figure_size = (12, 3)
    figure_size = (12, 4)
    
    fig, axes = plt.subplots(1, 2, figsize=figure_size)

    sns.countplot(
        data=dataframe_low,
        x=column_name,
        color='blue',
        ax=axes[0]
    )
    
    axes[0].set_title('Low subset')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel(column_name)

    sns.countplot(
        data=dataframe_high,
        x=column_name,
        color='red',
        ax=axes[1]
    )
    axes[1].set_title('High subset')
    axes[1].set_ylabel('Count')
    axes[1].set_xlabel(column_name)

    max_count = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, max_count)
    axes[1].set_ylim(0, max_count)

    plt.tight_layout()


    # Save the plot
    output_path = FIGURES_DIR / "plot_countplot_column_row_nan_dist_{column_name}.png"
    plt.savefig(output_path)

    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# This one is called from the eda_row_nan_divide_by_threshold in the eda.py file

def plot_countplot_compare_row_nan_dist_per_column(dataframe_low, dataframe_high, column_name):
    #figure_size = (14, 3)
    #figure_size = (12, 3)
    figure_size = (12, 4)
    
    fig, axes = plt.subplots(1, 2, figsize=figure_size)

    sns.countplot(
        data=dataframe_low,
        x=column_name,
        color='blue',
        ax=axes[0]
    )
    
    axes[0].set_title('Low subset')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel(column_name)

    sns.countplot(
        data=dataframe_high,
        x=column_name,
        color='red',
        ax=axes[1]
    )
    axes[1].set_title('High subset')
    axes[1].set_ylabel('Count')
    axes[1].set_xlabel(column_name)

    max_count = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, max_count)
    axes[1].set_ylim(0, max_count)

    plt.tight_layout()

    # file name creation
    file_name = f"plot_countplot_eda_column_row_nan_dist_{column_name}.png"

    # Save the plot
    output_path = FIGURES_DIR / file_name
    plt.savefig(output_path)

    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Not using

def plot_barplot_compare_row_nan_dist_per_column_percentage(dataframe_low, dataframe_high, column_name):
    figure_size = (10, 2.5)
    fig, axes = plt.subplots(1, 2, figsize=figure_size)

    # Calculate percentages for low subset
    low_counts = dataframe_low[column_name].value_counts(normalize=True).sort_index() * 100
    low_dataframe = pd.DataFrame({column_name: low_counts.index, 'percentage': low_counts.values})
    
    sns.barplot(x=column_name, y='percentage', data=low_dataframe, color='blue', ax=axes[0])
    axes[0].set_title('Low subset')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_ylim(0, 100)
    axes[0].set_xlabel(column_name)

    # Calculate percentages for high subset
    high_counts = dataframe_high[column_name].value_counts(normalize=True).sort_index() * 100
    high_dataframe = pd.DataFrame({column_name: high_counts.index, 'percentage': high_counts.values})
    
    sns.barplot(x=column_name, y='percentage', data=high_dataframe, color='red', ax=axes[1])
    axes[1].set_title('High subset')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel(column_name)

    plt.tight_layout()

    # Save file if you want (uncomment and define FIGURES_DIR)
    # file_name = f"plot_countplot_eda_column_row_nan_dist_{column_name}.png"
    # output_path = FIGURES_DIR / file_name
    # plt.savefig(output_path)

    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


# From Helper.py 

def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(40, 10))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i], -0.01), va="top", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    ax.set_ylim(bottom=-0.05)

    # Save the plot
    output_path = FIGURES_DIR / "plot_scree_v1.png"
    plt.savefig(output_path)

    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# From Helper.py 

def scree_plot_v2(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(40, 10))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        y_offset = -0.015 if i % 2 == 0 else -0.045  # Alternate y position
        ax.annotate(
            f"{vals[i]*100:.2f}%", 
            (ind[i], y_offset), 
            va="top", 
            ha="center", 
            fontsize=12
        )
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    ax.set_ylim(bottom=-0.05)

    # Save the plot
    output_path = FIGURES_DIR / "plot_scree_v2.png"
    plt.savefig(output_path)

    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def plot_pca_heatmap(pca, feature_names, num_components=5, figsize=(14,8), cmap='coolwarm'):
    """
    Plots a heatmap showing feature weights across PCA components.

    Parameters:
    - pca: fitted PCA object from sklearn
    - feature_names: list of feature names (columns after preprocessing)
    - num_components: number of PCA components to plot
    - figsize: size of the heatmap
    - cmap: colormap for heatmap
    """
    # Get the PCA component matrix (shape: n_components x n_features)
    components = pca.components_[:num_components]

    # Create a DataFrame for heatmap
    df_pca = pd.DataFrame(components, 
                          columns=feature_names,
                          index=[f'Dimension {i+1}' for i in range(num_components)])
    
    # Plot the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df_pca, cmap=cmap, center=0, cbar=True, 
                linewidths=0.5, linecolor='gray')
    plt.title(f'Top {num_components} PCA Components Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Principal Components')
    plt.xticks(rotation=90)
    plt.tight_layout()


    # Save the plot
    output_path = FIGURES_DIR / "plot_heatmap_pca.png"
    plt.savefig(output_path)

    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def plot_kmeans_elbow_method(model_score, clusters_range=(2, 30), step_interval=2):

    start, end = clusters_range
    
    clusters = range(start, end + 1, step_interval )

    fig, ax = plt.subplots()
    ax.plot(clusters, model_score)
    ax.set(ylabel='avg distance', xlabel='# Clusters: k')
    ax.grid()
    plt.xticks(np.arange(start, end, step_interval))


    # Save the plot
    output_path = FIGURES_DIR / "plot_elbow_kmeans.png"
    plt.savefig(output_path)

    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def plot_elbow(model_score, clusters_range):
    
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    
    ax.plot(clusters_range, model_score, linestyle= "--",marker = "o", color="orange")
    ax.set_xlabel("No. of Clusters")
    ax.set_ylabel("Sum of squared distances")
    ax.set_title("Kmeans - Cluster distances vs No. of Clusters")
    
    # Save the plot
    output_path = FIGURES_DIR / "plot_elbow_kmeans_v2.png"
    plt.savefig(output_path)
    
    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def plot_elbow_with_knee_locator(model_scores, cluster_range, optimal_k):

    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, model_scores, marker='o', label='Average Within-Cluster Distance')
    plt.axvline(optimal_k, color='red', linestyle='--', label=f'Elbow at k={optimal_k}')
    plt.title("Elbow Method with KneeLocator")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Within-Cluster Distance")
    plt.legend()
    plt.grid(True)


    # Save the plot
    output_path = FIGURES_DIR / "plot_elbow_kmeans_kneelocator.png"
    plt.savefig(output_path)


    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def plot_cluster_distribution(cluster_info):
    """
    Plots the distribution of clusters in terms of population and customers.
    
    Args:
        cluster_info (pd.DataFrame): DataFrame containing cluster information with columns 'Cluster', 'Population', and 'Customers'.
    """
    
    import matplotlib.pyplot as plt

    # Ensure the DataFrame has the required columns
    if not all(col in cluster_info.columns for col in ['Cluster', 'Population', 'Customers']):
        raise ValueError("DataFrame must contain 'Cluster', 'Population', and 'Customers' columns.")
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))

    ax1.bar(cluster_info["Cluster"], cluster_info["Population"])
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("No. of People")
    ax1.set_title("General Population")

    ax2.bar(cluster_info["Cluster"], cluster_info["Customers"])
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("No. of People")
    ax2.set_title("Customers")

    fig.suptitle("Cluster Distributions")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot
    output_path = FIGURES_DIR / "plot_bar_cluster_distribution.png"
    plt.savefig(output_path)


    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def plot_combined_cluster_distribution(cluster_info, title_suffix="", save_as="plot_combined_cluster_distribution.png"):
    """
    Plots a combined barplot comparing cluster proportions for general population vs customers.
    
    Args:
        cluster_info (pd.DataFrame): DataFrame with columns 'Cluster', 'Population', 'Customers' (as proportions or counts).
        title_suffix (str): Optional suffix for the plot title.
        save_as (str): Filename for saving the plot.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Melt the dataframe to long-form for seaborn
    cluster_info_melted = cluster_info.melt(id_vars='Cluster', value_vars=['Population', 'Customers'],
                                            var_name='Dataset', value_name='Proportion')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=cluster_info_melted, x='Cluster', y='Proportion', hue='Dataset')

    plt.title(f"Cluster Proportion Comparison {title_suffix}", fontsize=14)
    plt.ylabel("Proportion" if cluster_info[['Population', 'Customers']].max().max() <= 1 else "No. of People")
    plt.xlabel("Cluster")
    plt.axhline(1, color='red', linestyle='--', alpha=0.3)  # Optional reference line at 1 (for ratios)
    plt.legend(title='Dataset')

    output_path = FIGURES_DIR / save_as
    plt.savefig(output_path)
    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def plot_cluster_proportions(cluster_info):
    cluster_info["Pop_proportion"] = (cluster_info["Population"]/cluster_info["Population"].sum()*100).round(2)
    cluster_info["Cust_proportion"] = (cluster_info["Customers"]/cluster_info["Customers"].sum()*100).round(2)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))

    ax1.bar(cluster_info["Cluster"], cluster_info["Pop_proportion"])
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Proportion of Total (%)")
    ax1.set_title("General Population")

    ax2.bar(cluster_info["Cluster"], cluster_info["Cust_proportion"])
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Proportion of Total (%)")
    ax2.set_title("Customers")

    fig.suptitle("Percentage of people under each cluster")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


    # Save the plot
    output_path = FIGURES_DIR / "plot_bar_cluster_proportions.png"
    plt.savefig(output_path)

    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


import seaborn as sns
import matplotlib.pyplot as plt

def plot_cluster_proportions(comparison_dataframe):
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=comparison_dataframe,
        x='cluster',
        y='proportion',
        hue='dataset'
    )
    plt.title('Cluster Proportions by Dataset')
    plt.ylabel('Proportion')
    plt.xlabel('Cluster')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
 

def plot_cluster_counts(comparison_dataframe):
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=comparison_dataframe,
        x='cluster',
        y='count',
        hue='dataset'
    )
    plt.title('Cluster Proportions by Dataset')
    plt.ylabel('Proportion')
    plt.xlabel('Cluster')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def plot_ratio_to_baseline(comparison_dataframe):
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=comparison_dataframe,
        x='cluster',
        y='ratio_to_baseline',
        hue='dataset'
    )
    plt.axhline(1.0, color='gray', linestyle='--', label='Baseline (1.0)')
    plt.title('Ratio to Baseline by Cluster')
    plt.ylabel('Ratio (Customer / Population)')
    plt.xlabel('Cluster')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



def plot_difference_to_baseline(comparison_dataframe):
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=comparison_dataframe,
        x='cluster',
        y='difference_to_baseline',
        hue='dataset'
    )
    plt.axhline(1.0, color='gray', linestyle='--', label='Baseline (0.0)')
    plt.title('Difference to Baseline by Cluster')
    plt.ylabel('Difference (Customer / Population)')
    plt.xlabel('Cluster')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def plot_over_under_representation(comparison_dataframe, method = 'ratio', threshold=0.05):
    # Add flag for over/under representation
    dataframe = comparison_dataframe.copy()

    if method == 'ratio':
        dataframe['representation'] = dataframe['ratio_to_baseline'].apply(
            lambda x: 'Over-represented' if x > (1 + threshold) else ('Under-represented' if x < (1 - threshold) else 'Neutral')
        )
        method_y = 'ratio_to_baseline'
        method_label_y = 'Difference In Proportion'

    elif method == 'difference':
        dataframe['representation'] = dataframe['difference_to_baseline'].apply(
            lambda x: 'Over-represented' if x > threshold else ('Under-represented' if x < -threshold else 'Neutral')
        )
        method_y = 'difference_to_baseline'
        method_label_y = 'Difference in Count'
    else:
        raise ValueError("Method must be either 'ratio' or 'difference'.")  
    
    plt.figure(figsize=(12, 6))
    
    sns.barplot(
        data=dataframe,
        x='cluster',
        y=method_y,
        hue='representation',
        dodge=False,
        palette={'Over-represented': 'green', 'Under-represented': 'red', 'Neutral': 'gray'}
    )
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f'Difference to Baseline (Threshold = +/-{threshold})')
    plt.ylabel(method_label_y)
    plt.xlabel('Cluster')
    plt.legend(title='Representation')
    plt.tight_layout()
    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
 

import seaborn as sns
import matplotlib.pyplot as plt

def plot_absolute_cluster_proportions(cluster_comparison):
    """
    Plots the absolute proportions of clusters in the general population and customer data.
    
    Args:
        cluster_comparison (pd.DataFrame): DataFrame containing cluster information with columns 'cluster', 'proportion', and 'dataset'.
    """
    
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=cluster_comparison,
        x='cluster',
        y='proportion',
        hue='dataset',
        palette='Set2'
    )
    plt.title("Cluster Proportions: General Population vs Customer Data")
    plt.ylabel("Proportion")
    plt.xlabel("Cluster")
    plt.legend()
    plt.tight_layout()
    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def plot_ratio_to_baseline(cluster_comparison):
    """
    Plots the ratio of customer to population proportions for each cluster.
    
    Args:
        cluster_comparison (pd.DataFrame): DataFrame containing cluster information with columns 'cluster', 'ratio_to_baseline', and 'dataset'.
    """
    
    ratio_df = cluster_comparison[cluster_comparison['dataset'] != 'General Population']

    sns.barplot(
        data=ratio_df,
        x='cluster',
        y='ratio_to_baseline',
        hue='dataset',
        palette='coolwarm'
    )
    plt.axhline(1, color='gray', linestyle='--')
    plt.title("Customer Data Cluster Ratio to General Population")
    plt.ylabel("Ratio (Customer / General Pop)")
    plt.xlabel("Cluster")
    plt.tight_layout()
    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def plot_combined_cluster_distribution(cluster_info_dataframe, title_suffix="", save_as="plot_combined_cluster_distribution.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    cluster_info_dataframe_melted = cluster_info_dataframe.melt(id_vars='Cluster', value_vars=['Population', 'Customers'],
                                            var_name='Dataset', value_name='Proportion')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=cluster_info_dataframe_melted, x='Cluster', y='Proportion', hue='Dataset')

    plt.title(f"Cluster Proportion Comparison {title_suffix}", fontsize=14)
    plt.ylabel("Proportion" if cluster_info_dataframe[['Population', 'Customers']].max().max() <= 1 else "No. of People")
    plt.xlabel("Cluster")
    plt.axhline(1, color='red', linestyle='--', alpha=0.3)
    plt.legend(title='Dataset')

    output_path = FIGURES_DIR / save_as
    plt.savefig(output_path)
    plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 




#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



if __name__ == "__main__":
    app()
