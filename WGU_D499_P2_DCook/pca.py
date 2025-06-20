from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pathlib as Path

from sklearn.decomposition import PCA



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

# Inspired by/From helper.py
# Modified to remove Standardscaler since i'm scaling my data in sections above

def do_pca_v2(n_components, prescaled_data):
    '''
    Transforms data using PCA to create n_components, and provides back the results of the transformation.

    args: 
        n_components: int, the number of components to keep in the PCA transformation
        prescaled_data: the data to be transformed, already scaled (DataFrame or np.array)

    returns: 
        pca - the pca object created after fitting the data
        prescaled_data_pca - the transformed X matrix with new number of components
    '''
    #X = StandardScaler().fit_transform(data)
    pca = PCA(n_components)
    prescaled_data_pca = pca.fit_transform(prescaled_data)
    return pca, prescaled_data_pca


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# From helper.py

def pca_results(full_dataset, pca, plot=True):
	'''
	Create a DataFrame of the PCA results Includes dimension feature weights and explained variance Visualizes the PCA results
     
    args:
        full_dataset: the original dataset (DataFrame)
        pca: the PCA object after fitting the data
        plot: boolean, whether to plot the PCA results (default=True)
    returns:
        pd.DataFrame: a DataFrame containing explained variance and PCA components
	'''

	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = full_dataset.keys())
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions
    
	if plot:
		fig, ax = plt.subplots(figsize = (14,8))
		components.plot(ax = ax, kind = 'bar')
		ax.set_ylabel("Feature Weights")
		ax.set_xticklabels(dimensions, rotation=0)

		for i, ev in enumerate(pca.explained_variance_ratio_):
			ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

		plt.show()
		plt.close(fig)

        
	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)






#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# function adapted from Lesson 4.20's "PCA_Mini_Project_Solution" 

def explained_variance_component_check(dataframe, pca_function, explained_variance_threshold=0.85, start=3, plot=False):
    """
    Iteratively checks how many PCA components are needed to reach the desired explained variance.

    Parameters:
    - dataframe: the original scaled dataset (DataFrame)
    - pca_function: a function that accepts n_components and dataframe, and returns (pca, transformed_data)
    - explained_variance_threshold: float, the cumulative variance you want to reach (default=0.85)
    - start: the initial number of components to test (default=3)
    - plot: boolean, whether to plot the PCA results (default=False)

    Returns:
    - num_comps: number of components that meet/exceed the threshold
    - pca: the final fitted PCA object
    - comp_check: DataFrame with explained variance and component weights
    - plots the PCA results if plot=True
    """
    
    for comp in range(start, dataframe.shape[1] + 1):
        pca, transformed = pca_function(comp, dataframe)
        comp_check = pca_results(dataframe, pca, plot=plot)
        if comp_check['Explained Variance'].sum() >= explained_variance_threshold:
            break

    num_comps = comp_check.shape[0]
    total_variance = comp_check['Explained Variance'].sum()
    print(f"✅ Using {num_comps} components, we can explain {total_variance:.2%} of the variance in the data.")
    
    return num_comps, pca, comp_check



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

import matplotlib.pyplot as plt
import numpy as np

def auto_pick_pca_components(dataframe, explained_variance_threshold=0.85, max_components=None, show_plot=True):
    """
    Automatically finds the number of PCA components needed to reach a desired explained variance threshold.

    Parameters:
    - dataframe: Scaled input DataFrame
    - explained_variance_threshold: Target cumulative explained variance (default=0.85)
    - max_components: Max number of components to consider (default: all features)
    - show_plot: Whether to show the elbow plot (default=True)

    Returns:
    - pca: fitted PCA object
    - n_components: number of components selected
    - plots the cumulative explained variance if show_plot=True
    """
    from sklearn.decomposition import PCA

    n_features = dataframe.shape[1]
    max_components = max_components or n_features

    # Fit PCA with all components
    pca = PCA(n_components=max_components)
    pca.fit(dataframe)

    # Cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components where the threshold is met
    n_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1

    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(1, max_components + 1), cumulative_variance, marker='o', linestyle='--')
        plt.axhline(y=explained_variance_threshold, color='r', linestyle='--', label=f"{explained_variance_threshold:.0%} threshold")
        plt.axvline(x=n_components, color='g', linestyle='--', label=f"{n_components} components")
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance - Elbow Plot')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print(f"✅ Auto-selected {n_components} components explaining {cumulative_variance[n_components-1]:.2%} of the variance.")
    
    # Refit PCA with selected number of components
    pca = PCA(n_components=n_components)
    pca.fit(dataframe)

    return pca, n_components

# https://www.baeldung.com/cs/pca
# https://medium.com/@riya190304/finding-optimal-number-of-components-in-pca-2141d2891bed


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def investigate_PC(pca, component, feature_names):
    #code inspired by helper_functions.py in PCA lesson
    '''
    Prints association of a feature to the weights of its components
    
    args:
        pca: the PCA object after fitting the data
        component: int, the index of the principal component to investigate (0-indexed)
        feature_names: list of feature names corresponding to the PCA components

    returns: 
        None, prints the top and bottom features associated with the specified principal component
    '''
    
    num_out = 10
    
    if(component <= len(pca.components_)):
        pca_feature_map = pd.DataFrame({'weight': pca.components_[component],
                                        'name': feature_names})
        
        pca_feature_map = pca_feature_map.sort_values(by='weight', ascending=False)
        
        print('Principal Component {}\n---------------\n'.format(component+1))
        print('TOP {0} PRINCIPAL COMPONENTS \n {1}'.format(num_out, pca_feature_map.iloc[:num_out,:]))
        print('\n BOTTOM {0} PRINCIPAL COMPONENTS \n {1}'.format(num_out, pca_feature_map.iloc[-num_out:,:]))
            
    else:
        print('Error in selecting component')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 







#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


if __name__ == "__main__":
    app()
