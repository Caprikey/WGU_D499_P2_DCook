from pathlib import Path

from loguru import logger
import pandas as pd
from pathlib import Path

from tqdm import tqdm
import typer

from WGU_D499_P2_DCook.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, EXTERNAL_DATA_DIR, ARCHIVED_DATA_DIR

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

#raw_data_folder = RAW_DATA_DIR
input_folder_path = RAW_DATA_DIR
#output_path = PROCESSED_DATA_DIR

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def load_csv(filename) -> pd.DataFrame:
    """
    Loads a CSV file from the input folder path.
    
    Args:
        filename (str): The name of the CSV file to load from the input folder path.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded CSV data.
    """

    import pandas as pd

    input_path = input_folder_path / filename
    
    dataframe = pd.read_csv(input_path)
    
    logger.info(f"Loaded dataset from {input_path} with {len(dataframe)} rows and {len(dataframe.columns)} columns.")

    
    return dataframe
    
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def load_csv_with_parameters(filename, **kwargs) -> pd.DataFrame:
    """
    Loads a CSV file from the input folder path with additional parameters.
    
    Args:
        filename (str): The name of the CSV file to load from the input folder path.
        **kwargs: Additional keyword arguments to pass to `pd.read_csv`.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded CSV data.
    """

    import pandas as pd
        
    input_path = input_folder_path / filename

    dataframe = pd.read_csv(input_path, **kwargs)
        
    logger.info(f"Loaded dataset from {input_path} with {len(dataframe)} rows and {len(dataframe.columns)} columns.")

    return dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def write_checkpoints(save_path, save_name, dataframe):
    """
    Saves a DataFrame to CSV and Pickle files at the specified path.
    
    Args:
        save_path (str): The path where the DataFrame should be saved. Options include 'raw_data', 'processed_data', 'interim_data', 'external_data', or 'archived_data'.
        save_name (str): The name of the file to save the DataFrame as.
        dataframe (pd.DataFrame): The DataFrame to be saved.
    Returns:
        None, this outputs the DataFrame to CSV and Pickle files at the specified path.
    """
    #print(save_path)
    #print(save_name)
    #print(dataframe)


    if save_path is None:
        # If no save path is provided, use the default processed data directory
        save_path = PROCESSED_DATA_DIR

    else:
        if save_path == "raw_data":
            # If the save path is 'raw_data', use the raw data directory
            save_path = RAW_DATA_DIR
        elif save_path == "processed_data":
            # If the save path is 'processed_data', use the processed data directory
            save_path = PROCESSED_DATA_DIR  
        elif save_path == "interim_data":
            # If the save path is 'interim_data', use the interim data directory
            save_path = INTERIM_DATA_DIR
        elif save_path == "external_data":
            # If the save path is 'external_data', use the external data directory
            save_path = EXTERNAL_DATA_DIR
        elif save_path == "archived_data":
            #print("entered elif archived_data")
            # If the save path is 'archived_data', use the archived data directory
            save_path = ARCHIVED_DATA_DIR
        else:
            # If the save path is not recognized, raise an error
            save_path = PROCESSED_DATA_DIR

            #raise ValueError(f"Unrecognized save path: {save_path}. Please use 'raw_data', 'processed_data', 'interim_data', 'external_data', or 'archived_data'.")

    print(type(save_path))



    # Set Save Folder Path 
    #save_folder_path = save_path
    save_folder_path = str(save_path)

    #print(type(save_folder_path))

    save_folder_path_csv = save_folder_path
    #print(save_folder_path_csv)

    save_folder_path_pickle = save_folder_path
    #save_folder_path = fm.get_folder_path(save_path)
    #print(save_folder_path_pickle)
    
    # Set Save File Name
    save_file_name = save_name

    # Set Save File Name For CSV Export 
    save_file_name_csv = save_name
    #print(save_file_name_csv)

    save_file_name_pickle = save_name
    #print(save_file_name_pickle)

    # Create Export File Path For CSV
    #save_file_path_csv = Path(save_folder_path_csv + "/" + save_file_name_csv + ".csv")
    save_file_path_csv = save_folder_path_csv + "/" + save_file_name_csv + ".csv"
    #print(save_file_path_csv)
    
    # Create Export File Path For Pickle
    #save_file_path_pickle = Path(save_folder_path_pickle + "/" + save_file_name_pickle + ".pkl")
    save_file_path_pickle = save_file_path_csv.replace(".csv", ".pkl")
    #print(save_file_path_pickle)
    
    try:      
        # Export CSV
        dataframe.to_csv(save_file_path_csv, sep=",", index=False)
        #print(f"Dataframe was successfully exported to CSV, {save_file_path}")
        
        # Export Pickle
        dataframe.to_pickle(save_file_path_pickle)
        #print(f"Dataframe was successfully exported to CSV, {save_file_path_pickle}")
        #print("Check Point Successful")

        
        
    except:
        
        
        print(f"Save Foler Path: {save_folder_path}")
        print(f"Save_File_Name: {save_file_name}")
        print(f"Save_Save_Path_CSV: {save_file_path_csv}")
        print(f"Save_Save_Path_Pickle: {save_file_path_pickle}")
        print("Checkpoint Failed")



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 




if __name__ == "__main__":
    app()
