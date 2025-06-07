from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np



from WGU_D499_P2_DCook.config import PROCESSED_DATA_DIR

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


def column_correction_ost_west_kz(dataframe, keep_original = True):

    
    # prefill nas
    dataframe['OST_WEST_KZ'] = dataframe['OST_WEST_KZ'].fillna('')

    # Get insertion point (after original column)
    insert_loc = dataframe.columns.get_loc('OST_WEST_KZ') + 1
    
    # Make new columns
    dataframe['OST_WEST_KZ_EAST'] = (dataframe['OST_WEST_KZ'] == 'O').astype(int)
    
    dataframe['OST_WEST_KZ_WEST'] = (dataframe['OST_WEST_KZ'] == 'W').astype(int)

    # Pop columns
    east_column = dataframe.pop('OST_WEST_KZ_EAST')
    west_column = dataframe.pop('OST_WEST_KZ_WEST')


    # Insert them back in desired order
    dataframe.insert(insert_loc, 'OST_WEST_KZ_EAST', east_column)
    
    dataframe.insert(insert_loc + 1, 'OST_WEST_KZ_WEST', west_column)

    if not keep_original:
        dataframe.drop(columns='OST_WEST_KZ', inplace=True)
    
    return dataframe

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



# Remapping the values to be more consistent in binary encoding
def column_correction_vers_type(dataframe):

    dataframe['VERS_TYP'] = dataframe['VERS_TYP'].replace({1: 0, 2: 1})
    
    return dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



# Remapping the values to be more consistent in binary encoding
def column_correction_anrede_kz(dataframe):

    dataframe['ANREDE_KZ'] = dataframe['ANREDE_KZ'].replace({1: 0, 2: 1})
    
    return dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 




def column_correction_cameo_deu_2015(dataframe):

    # Removing first character
    def check_value_cameo_deu_2015(value):
        if isinstance(value, str):
            if len(value) == 2:
                return value[-1]
            return value
        
    dataframe['CAMEO_DEU_2015'] = dataframe['CAMEO_DEU_2015'].apply(check_value_cameo_deu_2015)

    # Remapping values
    #cameo_deu_2015_hex_to_int_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}

    #dataframe['CAMEO_DEU_2015'] = dataframe['CAMEO_DEU_2015'].replace(cameo_deu_2015_hex_to_int_map)
    
    return dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



def feature_mapping(
    dataframe, 
    original_column_names, 
    map_dictionary_list, 
    new_columns_list, 
    map_values=True, 
    create_new=True, 
    keep_index=True, 
    keep_original=True
):

    # Normalize to list if single column is passed
    if isinstance(original_column_names, str):
        original_column_names = [original_column_names]
        map_dictionary_list = [map_dictionary_list]
        new_columns_list = [new_columns_list]

    if create_new:
        if new_columns_list is None:
            raise ValueError("new_columns_list must be provided when create_new is True.")
        if not (len(original_column_names) == len(map_dictionary_list) == len(new_columns_list)):
            raise ValueError("original_column_names, map_dictionary_list, and new_columns_list must be the same length.")
    else:
        if not (len(original_column_names) == len(map_dictionary_list)):
            raise ValueError("original_column_names and map_dictionary_list must be the same length when create_new is False.")


    for index, (col_name, col_dicts) in enumerate(zip(original_column_names, map_dictionary_list)):

        # Normalize if single mapping and suffix are passed
        if isinstance(col_dicts, dict):
            col_dicts = [col_dicts]

        if create_new:
            col_suffixes = new_columns_list[index]
            
            if isinstance(col_suffixes, str):
                col_suffixes = [col_suffixes]

            if len(col_dicts) != len(col_suffixes):
                raise ValueError(f"For column '{col_name}', map_dictionary_list and new_columns_list must match in length.")

        else:
            col_suffixes = [None] * len(col_dicts)

        if map_values:

            for dict_map, suffix in zip(col_dicts, col_suffixes):
                if create_new:
                    new_column_name = f"{col_name}_{suffix}"
                    
                else:
                    new_column_name = col_name
                    
                dataframe[new_column_name] = dataframe[col_name].map(dict_map)


        if keep_index and create_new:

            insert_index_loc = dataframe.columns.get_loc(col_name) + 1
            
            for suffix in col_suffixes:
                new_column_name = f"{col_name}_{suffix}"

                if new_column_name in dataframe.columns:

                    popped_column = dataframe.pop(new_column_name)

                    dataframe.insert(insert_index_loc, new_column_name, popped_column)

                    insert_index_loc += 1

        if not keep_original:

            dataframe.drop(columns=col_name, inplace=True)


    return dataframe



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



def column_encode_multi_category_to_single_onehot(dataframe, column_name, drop_first_flag = False, drop_original=True):

    if dataframe[column_name].dtype != 'category':
        dataframe[column_name] = dataframe[column_name].astype('category')

    dummies = pd.get_dummies(dataframe[column_name], prefix = column_name, prefix_sep="_", drop_first=drop_first_flag)

    if drop_original == True:
        dataframe = dataframe.drop(column_name, axis=1)

    dataframe = pd.concat([dataframe, dummies], axis = 1)
    
    return dataframe

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def column_reengineering_pragende_jugendjahre(dataframe, one_hot_encode=False, keep_original = True):
    """
    Custom function to re-engineer the 'PRAEGENDE_JUGENDJAHRE' feature from a single number code into two categorical features:
    - 'PRAEGENDE_JUGENDJAHRE_GENERATION': the generation decade of the person
    - 'PRAEGENDE_JUGENDJAHRE_MOVEMENT': the movement of the person, Mainstream or Avantgarde

    Steps:
    1. Capture the position of the original column in the DataFrame.
    2. Create two new columns.
    3. Insert these new columns directly after the original column's position
    4. Drop the original column.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the 'PRAEGENDE_JUGENDJAHRE' column.
    one_hot_encode (bool): If True, returns one-hot encoded columns instead of raw digits.
    
    Returns:
    pd.DataFrame: The updated DataFrame with the new columns.
    """

    # Dictionary

    generation_map = {1 : 1, 2 : 2, 
                      3: 2, 4: 2, 
                      5: 3, 6: 3, 7:3, 
                      8: 4, 9: 4, 
                      10:5, 11:5, 12: 5, 13: 5, 
                      14: 6, 15: 6}

    movement_map = {1: 1, 3: 1, 5: 1, 8: 1, 10: 1, 12: 1, 14: 1,
                    2: 2, 4: 2, 6: 2, 7: 2, 9: 2, 11: 2, 13: 2, 15: 2
    }


    # Create the new columns 
    dataframe['PRAEGENDE_JUGENDJAHRE_GENERATION'] = dataframe['PRAEGENDE_JUGENDJAHRE'].map(generation_map)
    
    dataframe['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = dataframe['PRAEGENDE_JUGENDJAHRE'].map(movement_map)

    # Pop both columns
    generation = dataframe.pop('PRAEGENDE_JUGENDJAHRE_GENERATION')
    
    movement = dataframe.pop('PRAEGENDE_JUGENDJAHRE_MOVEMENT')

    # Get insertion point (after original column)
    insert_loc = dataframe.columns.get_loc('PRAEGENDE_JUGENDJAHRE') + 1

    # Insert them back in desired order
    dataframe.insert(insert_loc, 'PRAEGENDE_JUGENDJAHRE_GENERATION', generation)
    dataframe.insert(insert_loc + 1, 'PRAEGENDE_JUGENDJAHRE_MOVEMENT', movement)

    # Drop original column
    #dataframe.drop(columns='PRAEGENDE_JUGENDJAHRE', inplace=True)

    # Convert to category
    dataframe['PRAEGENDE_JUGENDJAHRE_GENERATION'] = dataframe['PRAEGENDE_JUGENDJAHRE_GENERATION'].astype('category')
    dataframe['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = dataframe['PRAEGENDE_JUGENDJAHRE_MOVEMENT'].astype('category')
    
    # One-hot encode = True
    if one_hot_encode:
        dataframe = pd.get_dummies(dataframe, columns=['PRAEGENDE_JUGENDJAHRE_GENERATION', 'PRAEGENDE_JUGENDJAHRE_MOVEMENT'], prefix=['PRAEGENDE_JUGENDJAHRE_GENERATION', 'PRAEGENDE_JUGENDJAHRE_MOVEMENT'])

    if not keep_original:
        dataframe.drop(columns='PRAEGENDE_JUGENDJAHRE', inplace=True)

    return dataframe

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def column_reengineering_cameo_intl_2015(dataframe, one_hot_encode=False, keep_original = True):
    """
    Custom function to re-engineer the 'CAMEO_INTL_2015' feature from a single two-digit code into two categorical features:
    - 'CAMEO_INTL_2015_WEALTH': the tens digit (wealth status)
    - 'CAMEO_INTL_2015_LIFE_STAGE': the ones digit (life stage typology)

    Steps:
    1. Capture the position of the original column in the DataFrame.
    2. Create two new columns based on the original's tens and ones digits.
    3. Insert these new columns directly after the original column's position.
    4. Drop the original column.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the 'CAMEO_INTL_2015' column.
    one_hot_encode (bool): If True, returns one-hot encoded columns instead of raw digits.
    
    Returns:
    pd.DataFrame: The updated DataFrame with the new columns.
    """

    # convert the coumn to string format
    cameo = dataframe['CAMEO_INTL_2015'].astype('str')

    # Create the new columns assign all nulls
    dataframe['CAMEO_INTL_2015_WEALTH'] = np.nan
    dataframe['CAMEO_INTL_2015_LIFE_STAGE'] = np.nan

    # Verify the values are digits
    valid_mask = cameo.str.isdigit()

    # Apply the mask to the column and extract only the signal digit
    dataframe.loc[valid_mask, 'CAMEO_INTL_2015_WEALTH'] = cameo[valid_mask].str[0].astype(int)
    dataframe.loc[valid_mask, 'CAMEO_INTL_2015_LIFE_STAGE'] = cameo[valid_mask].str[1].astype(int)
    
    # Pop both columns
    wealth = dataframe.pop('CAMEO_INTL_2015_WEALTH')
    life_stage = dataframe.pop('CAMEO_INTL_2015_LIFE_STAGE')
    
    # Get insertion point (after original column)
    insert_loc = dataframe.columns.get_loc('CAMEO_INTL_2015') + 1
    
    # Insert them back in desired order
    dataframe.insert(insert_loc, 'CAMEO_INTL_2015_WEALTH', wealth)
    dataframe.insert(insert_loc + 1, 'CAMEO_INTL_2015_LIFE_STAGE', life_stage)

    # Drop original column
    #dataframe.drop(columns='CAMEO_INTL_2015', inplace=True)

    # Convert to category
    dataframe['CAMEO_INTL_2015_WEALTH'] = dataframe['CAMEO_INTL_2015_WEALTH'].astype('category')
    dataframe['CAMEO_INTL_2015_LIFE_STAGE'] = dataframe['CAMEO_INTL_2015_LIFE_STAGE'].astype('category')

    # One-hot encode = True
    if one_hot_encode:
        dataframe = pd.get_dummies(dataframe, columns=['CAMEO_INTL_2015_WEALTH', 'CAMEO_INTL_2015_LIFE_STAGE'], prefix=['CI2015_WEALTH', 'CI2015_LIFE_STAGE'])

    if not keep_original:
        dataframe.drop(columns='CAMEO_INTL_2015', inplace=True)
        
    return dataframe

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def column_reengineering_lp_lebenshase(dataframe, one_hot_encode=False, keep_original=True):
    """
    Custom function to re-engineer the 'LP_LEBENSPHASE_FEIN' feature from a single number code into three categorical features:
    - 'LP_LEBENSPHASE_FEIN_LIFE_STAGE': The age stage of the person
    - 'LP_LEBENSPHASE_FEIN_FAMILY_STAGE': The family stage of the person
    - 'LP_LEBENSPHASE_FEIN_WEALTH_STAGE': The wealth stage of the person

    Steps:
    1. Capture the position of the original column in the DataFrame.
    2. Create three new columns.
    3. Insert these new columns directly after the original column's position
    4. Drop the original column.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the 'LP_LEBENSPHASE_FEIN' column.
    one_hot_encode (bool): If True, returns one-hot encoded columns instead of raw digits.
    keep_original (bool): If true, original column is kept; If false, it is deleted. 
    
    Returns:
    pd.DataFrame: The updated DataFrame with the new columns.
    """

    # Replacement Dictionary Maps

    # LP_LEBENSPHASE_FEIN
    life_stages = {1: 'younger_age', 2: 'middle_age', 3: 'younger_age', 4: 'middle_age', 
                   5: 'advanced_age', 6: 'retirement_age', 7: 'advanced_age', 8: 'retirement_age', 
                   9: 'middle_age', 10: 'middle_age', 11: 'advanced_age', 12: 'retirement_age', 
                   13: 'advanced_age', 14: 'younger_age', 15: 'advanced_age', 16: 'advanced_age', 
                   17: 'middle_age', 18: 'younger_age', 19: 'advanced_age', 20: 'advanced_age', 
                   21: 'middle_age', 22: 'middle_age', 23: 'middle_age', 24: 'middle_age', 
                   25: 'middle_age', 26: 'middle_age', 27: 'middle_age', 28: 'middle_age', 
                   29: 'younger_age', 30: 'younger_age', 31: 'advanced_age', 32: 'advanced_age', 
                   33: 'younger_age', 34: 'younger_age', 35: 'younger_age', 36: 'advanced_age',
                   37: 'advanced_age', 38: 'retirement_age', 39: 'middle_age', 40: 'retirement_age'}

    family_stages = {1: 'single', 2: 'single', 3: 'single', 4: 'single', 
                   5: 'single', 6: 'single', 7: 'single', 8: 'single', 
                   9: 'single', 10: 'single', 11: 'single', 12: 'single', 
                   13: 'single', 14: 'couples', 15: 'couples', 16: 'couples', 
                   17: 'couples', 18: 'couples', 19: 'couples', 20: 'couples', 
                   21: 'single_parent', 22: 'single_parent', 23: 'single_parent', 24: 'family', 
                   25: 'family', 26: 'family', 27: 'family', 28: 'family', 
                   29: 'multihousehold', 30: 'multihousehold', 31: 'multihousehold', 32: 'multihousehold', 
                   33: 'multihousehold', 34: 'multihousehold', 35: 'multihousehold', 36: 'multihousehold',
                   37: 'multihousehold', 38: 'multihousehold', 39: 'multihousehold', 40: 'multihousehold'}

    wealth_stages = {1: 'low', 2: 'low', 3: 'average', 4: 'average', 
                    5: 'low', 6: 'low', 7: 'average', 8: 'average', 
                    9: 'average', 10: 'wealthy', 11: 'average', 12: 'average', 
                    13: 'top', 14: 'average', 15: 'low', 16: 'average', 
                    17: 'average', 18: 'wealthy', 19: 'wealthy', 20: 'top', 
                    21: 'low', 22: 'average', 23: 'wealthy', 24: 'low', 
                    25: 'average', 26: 'average', 27: 'average', 28: 'top', 
                    29: 'low', 30: 'average', 31: 'low', 32: 'average', 
                    33: 'average', 34: 'average', 35: 'top', 36: 'average', 
                    37: 'average', 38: 'average', 39: 'top', 40: 'top'}


    # Create the new columns 

    dataframe["LP_LEBENSPHASE_FEIN_LIFE_STAGE"] = dataframe["LP_LEBENSPHASE_FEIN"].map(life_stages)
    
    dataframe["LP_LEBENSPHASE_FEIN_FAMILY_STAGE"] = dataframe["LP_LEBENSPHASE_FEIN"] .map(family_stages)
    
    dataframe["LP_LEBENSPHASE_FEIN_WEALTH_STAGE"] = dataframe["LP_LEBENSPHASE_FEIN"] .map(wealth_stages)


    
    # Pop both columns
    life_stage = dataframe.pop('LP_LEBENSPHASE_FEIN_LIFE_STAGE')
    family_stage = dataframe.pop('LP_LEBENSPHASE_FEIN_FAMILY_STAGE')
    wealth_stage = dataframe.pop('LP_LEBENSPHASE_FEIN_WEALTH_STAGE')


    # Get insertion point (after original column)
    insert_loc = dataframe.columns.get_loc('LP_LEBENSPHASE_FEIN') + 1

    # Insert them back in desired order
    dataframe.insert(insert_loc, 'LP_LEBENSPHASE_FEIN_LIFE_STAGE', life_stage)
    dataframe.insert(insert_loc + 1, 'LP_LEBENSPHASE_FEIN_FAMILY_STAGE', family_stage)
    dataframe.insert(insert_loc + 2, 'LP_LEBENSPHASE_FEIN_WEALTH_STAGE', wealth_stage)

    # Drop original column


    # Convert to category
    dataframe['LP_LEBENSPHASE_FEIN_LIFE_STAGE'] = dataframe['LP_LEBENSPHASE_FEIN_LIFE_STAGE'].astype('category')
    dataframe['LP_LEBENSPHASE_FEIN_FAMILY_STAGE'] = dataframe['LP_LEBENSPHASE_FEIN_FAMILY_STAGE'].astype('category')
    dataframe['LP_LEBENSPHASE_FEIN_WEALTH_STAGE'] = dataframe['LP_LEBENSPHASE_FEIN_WEALTH_STAGE'].astype('category')
    
    # One-hot encode = True
    if one_hot_encode:
        dataframe = pd.get_dummies(dataframe, columns=['LP_LEBENSPHASE_FEIN_LIFE_STAGE', 'LP_LEBENSPHASE_FEIN_FAMILY_STAGE', 'LP_LEBENSPHASE_FEIN_WEALTH_STAGE'], prefix=['LP_LEBENSPHASE_FEIN_LIFE_STAGE', 'LP_LEBENSPHASE_FEIN_FAMILY_STAGE', 'LP_LEBENSPHASE_FEIN_WEALTH_STAGE'])

    if not keep_original:
        dataframe.drop(columns='LP_LEBENSPHASE_FEIN', inplace=True)

    return dataframe

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def column_reengineering_wohnlage(dataframe, keep_original = True):

    rural_mapping = {1:0, 2:0, 3:0, 4:0, 5:0, 7:1, 8:1}
    neighbourhood_mapping = {1:1, 2:1, 3:1, 4:1, 5:1, 7:0, 8:0}

    # Map flag to correct WOHNLAGE value
    dataframe['WOHNLAGE_RURAL'] = dataframe['WOHNLAGE'].map(rural_mapping).fillna(0).astype(int)
    dataframe['WOHNLAGE_NEIGHBOURHOOD'] = dataframe['WOHNLAGE'].map(neighbourhood_mapping).fillna(0).astype(int)

    # Pop RURAL and Neighbourhood columns
    rural_flag = dataframe.pop('WOHNLAGE_RURAL')
    neighbourhood_flag = dataframe.pop('WOHNLAGE_NEIGHBOURHOOD')

    # Get insertion point (after original column)
    rural_insert_loc = dataframe.columns.get_loc('WOHNLAGE') + 1
    neighbourhood_insert_loc = dataframe.columns.get_loc('WOHNLAGE') + 2

    # Insert them back in desired order
    dataframe.insert(rural_insert_loc, 'WOHNLAGE_RURAL', rural_flag)
    dataframe.insert(neighbourhood_insert_loc, 'WOHNLAGE_NEIGHBOURHOOD', neighbourhood_flag)

    if not keep_original:
        dataframe = dataframe.drop(columns='WOHNLAGE')

    return dataframe

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def column_reengineering_plz8_family_business_building_flag(dataframe, keep_original = True):
    """
    Custom function to re-engineering a building type flag column based on the  "PLZ8_BAUMAX" feature. 
    """
    family_building_type_mapping = {1: 1, 2: 1, 3: 1, 4: 1, 5: 0}
    business_building_type_mapping = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1}

    # Map flag to correct PLZ8_BAUMAX value
    dataframe['PLZ8_BAUMAX_FAMILY_BUILDING'] = dataframe['PLZ8_BAUMAX'].map(family_building_type_mapping).fillna(0).astype(int)
    dataframe['PLZ8_BAUMAX_BUSINESS_BUILDING'] = dataframe['PLZ8_BAUMAX'].map(business_building_type_mapping).fillna(0).astype(int)

    # Pop FAMILY BUILDING TYPE FLAG COLUMN
    family_building_type_flag = dataframe.pop('PLZ8_BAUMAX_FAMILY_BUILDING')
    business_building_type_flag = dataframe.pop('PLZ8_BAUMAX_BUSINESS_BUILDING')
    
    # Get insertion point (after original column)
    family_insert_loc = dataframe.columns.get_loc('PLZ8_BAUMAX') + 1
    business_insert_loc = dataframe.columns.get_loc('PLZ8_BAUMAX') + 2

    # Insert them back in desired order
    dataframe.insert(family_insert_loc, 'PLZ8_BAUMAX_FAMILY_BUILDING', family_building_type_flag)
    dataframe.insert(business_insert_loc, 'PLZ8_BAUMAX_BUSINESS_BUILDING', business_building_type_flag)

    if not keep_original:
        dataframe = dataframe.drop(columns='PLZ8_BAUMAX')
    
    return dataframe

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


class FeatureTransformer:

    def __init__(self, operations=None):
        self.operations = operations or []
        self.skipped_columns = []

    #### #### #### ####   

    def map_column(self, dataframe, column, mappings, pre_fill_na, create_new, fill_na_value = None, new_names=None, keep_index = False, keep_original=True, encode_column = False):


        if pre_fill_na:
            dataframe[column] = dataframe[column].fillna(fill_na_value)
            
        if isinstance(column, str):
            column = [column]
        
        # Handle mappings: make sure it's a list of lists of dicts
        if isinstance(mappings, dict):
            mappings = [[mappings]]
        elif isinstance(mappings[0], dict) and len(column) == 1:
            mappings = [mappings]  # [[dict1, dict2, dict3]]
        elif isinstance(mappings[0], dict) and len(column) > 1:
            mappings = [[m] for m in mappings]  # One mapping per column
        elif all(isinstance(m, list) for m in mappings):
            pass  # Already nested
        else:
            raise ValueError("Invalid format for mappings")


        # Handle new_names
        if new_names:
            if isinstance(new_names[0], str) and len(column) == 1:
                new_names = [new_names]
            elif isinstance(new_names[0], str):
                new_names = [new_names] * len(column)
            elif not all(isinstance(n, list) and all(isinstance(i, str) for i in n) for n in new_names):
                raise ValueError("new_names must be a list of lists of strings")

                
        for idx, col in enumerate(column):
            col_mappings = mappings[idx]

            if isinstance(col_mappings, dict):
                col_mappings = [col_mappings]
            
            #col_names = new_names[idx] if new_names else [f"{col}_mapped{i}" for i in range(len(col_mappings))]
            col_names = new_names[idx] if new_names else [f"mapped{i}" for i in range(len(col_mappings))]

            
            if create_new:
                new_columns_list = []
    
                for mapping, suffix in zip(col_mappings, col_names):
                    new_column = f"{col}_{suffix}"
                    new_columns_list.append(new_column)
                    dataframe[new_column] = dataframe[col].map(mapping)
      
                if keep_index:
                    
                    insert_index_loc = dataframe.columns.get_loc(col) + 1
    
                    for new_column_name in new_columns_list:
                        popped_column = dataframe.pop(new_column_name)
                        dataframe.insert(insert_index_loc, new_column_name, popped_column)
                        insert_index_loc += 1
    
                    if not keep_original:
                        dataframe.drop(columns=col, inplace=True)

    
            else:             
                dataframe[col] = dataframe[col].map(col_mappings[0])

            if encode_column and create_new:
                dataframe = self.encode_new_map_columns(dataframe, new_columns_list, False, True)
    
        return dataframe

    #### #### #### #### 

    def replace_column(self, dataframe, column, replacements):        

        if isinstance(column, list):
            for col in column:

                dataframe[col] = dataframe[col].replace(replacements)

        else:
            dataframe[column] = dataframe[column].replace(replacements)
            
        return dataframe

    #### #### #### #### 


    def convert_data_type(self, dataframe, column, data_type):

        if isinstance(column, list):
            for col in column:
                dataframe[col] = dataframe[col].astype(data_type)
        
        else:
            dataframe[column] = dataframe[column].astype(data_type)

        return dataframe
    
    #### #### #### ####

    def encode_new_map_columns(self, dataframe, column_name_list, drop_first_flag = False, drop_original=True):

        for column in column_name_list:
            
            if dataframe[column].dtype != 'category':
                dataframe[column] = dataframe[column].astype('category')
        
            dummies = pd.get_dummies(dataframe[column], prefix = column, prefix_sep="_", drop_first=drop_first_flag)
        
            if drop_original == True:
                dataframe = dataframe.drop(column, axis=1)
        
            dataframe = pd.concat([dataframe, dummies], axis = 1)
    
        return dataframe
    

    #### #### #### ####      
    
    def transform(self, dataframe):
        
        for op in self.operations:
        
            if op["type"] == "map":    
                dataframe = self.map_column(dataframe, **op["params"])
                
            elif op["type"] == "replace":
                dataframe = self.replace_column(dataframe, **op["params"])
                
            elif op["type"] == "convert_data_type":
                dataframe = self.convert_data_type(dataframe, **op["params"])

        return dataframe

    #### #### #### ####    

    def transform_safe(self, dataframe):
        self.skipped_columns = []
        
        for op in self.operations:
            
            column = op["params"].get("column")
            
            if column is None:
                should_apply = True
            
            elif isinstance(column, list):
                
                should_apply = all(col in dataframe.columns for col in column)
            
            else:
                should_apply = column in dataframe.columns
                
            if should_apply:
                if op["type"] == "map":
                    dataframe = self.map_column(dataframe, **op["params"])
                    
                elif op["type"] == "replace":
                    dataframe = self.replace_column(dataframe, **op["params"])
                    
                elif op["type"] == "convert_data_type":
                    dataframe = self.convert_data_type(dataframe, **op["params"])
        
            else:
                print(f'Column: {column} was not found. Skipping operation: {op['type']}.')
                self.skipped_columns.append(column)
                pass
            
        return dataframe

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

#ops = [
#    {"type": "map", "params": {
#        "column": "grade",
#        "mappings": [{"A": 4, "B": 3, "C": 2}],
#        "new_names": ["numeric"],
#        "create_new": True,
#        "keep_index": False,
#        "keep_original": True,
#    }},
#    {"type": "map", "params": {
#        "column": "student_names",
#        "mappings": [{"John": 4, "Jess": 3, "Charles": 2}, {"Jim": 1, "Jones": 2, "Edward": 3}],
#        "new_names": ["first", "last"],
#        "create_new": True,
#        "keep_index": True
#    }},
#    {"type": "replace", "params": {
#        "column": "status",
#        "replacements": {"fail": "✘", "pass": "✔"}
#    }}
#]

#ft = FeatureTransformer(operations=ops)
#dataframe = ft.transform(dataframe)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

if __name__ == "__main__":
    app()
