import pandas as pd
# import json
import os
import sys
from sklearn.model_selection import train_test_split

from helper import Helper


def get_min_csv_length(directory: str) -> int:
    """
    Determines the minimum length of the files stored in the directory with extention .csv .

    Args: 
        directory: path string to the directory.
    Returns: 
        min_length: minimum length observed for the files in the directory.
    """
    # Lists all files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    # Initializes a variable to store the minimum length
    min_length = float('inf')
    
    # Iterates through each file and determine its length
    for file in files:
        file_path = os.path.join(directory, file)
        try:
            Helper.remove_commas(file_path)
            df = pd.read_csv(file_path, header=None)
            num_rows = len(df) #length of the dataframe

            # Updating the minimum length if the current file is shorter
            if num_rows < min_length:
                min_length = num_rows
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return min_length


def format_input_window(input_directory, window_size, output_directory):
    """
    Creates .csv files with required number of sliding windowed input as 'prompt' and the schema structure of the file as the 'response'.

    Agrs:
        input_directory: path string to the input directory.
        window_size: 
        output_directory: path string to the output directory.

    Returns: 
        one train file, one test file and as many test file as the input files saved as .csv files
    """  
    # List all .csv files in the directory
    csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]

    # make output directories if does not exist
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(f'{output_directory}/test', exist_ok=True)

    # Initialize an empty list to store train and validation DataFrames
    train_dataframes = []
    val_dataframes = []

    # Read each .csv file and store it in the list
    for file in csv_files:
        file_path = os.path.join(input_directory, file)
        schema_file = file.replace('.csv', '.xml')
        schema_file_path = os.path.join(input_directory, schema_file)

        # checks if the schema file exits. If there is no schema files, then stops further execution
        if schema_file not in os.listdir(input_directory):
            print(f"Schema file {schema_file} not found for {file}")
            sys.exit(1)

        # # for json schema files
        # schema_file = file.replace('.csv', '.json')
        # schema_file_path = os.path.join(input_directory, schema_file)

        try:
            Helper.remove_commas(file_path)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path, header=None)

            # Read the xml file as a string
            with open(schema_file_path, "r") as schema_xml:
                df_schema = schema_xml.read()

            # # load the stringyfied json schema file
            # with open(schema_file_path, 'r') as schema_json:
            #     df_schema = json.load(schema_json)

            # Split the CSV into sliding windows
            windows = [df.iloc[i:i+window_size] for i in range(0, len(df))]
            # Create the training data
            input_data = [
                (f"{window.to_csv(index=False, header=False, mode='a')}", df_schema)
                for window in windows
            ]

            # Split the dataset to train, test, validation
            if len(df) <= 8 :
                train_data = test_data = val_data = input_data
            else:
                train_data, test_data = train_test_split(input_data, test_size=0.1, random_state=42)
                train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
            
            train_data_df = pd.DataFrame(train_data, columns= ['prompt', 'response'])
            val_data_df = pd.DataFrame(val_data, columns= ['prompt', 'response'])
            test_data_df = pd.DataFrame(test_data, columns= ['prompt', 'response'])

            # Append the Dataframes into the respective lists to create one train and one validation file
            train_dataframes.append(train_data_df)
            val_dataframes.append(val_data_df)

            # Save the test data to a new .csv file
            test_data_df.to_csv(f'{output_directory}/test/{file}', index=False)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Concatenate all DataFrames horizontally for train and validation dataframes
    if train_dataframes:
        train_df_combined = pd.concat(train_dataframes, ignore_index=True)
        # Save the combined train DataFrame to a new .csv file
        train_df_combined.to_csv(f'{output_directory}/train.csv', index=False)

    if val_dataframes:
        val_df_combined = pd.concat(val_dataframes, ignore_index=True)
        # Save the combined test DataFrame to a new .csv file
        val_df_combined.to_csv(f'{output_directory}/val.csv', index=False)


def main():
    database = 'adventureworks'
    input_directory = f'./data/{database}/raw'
    output_directory = f'data/{database}/processed/xml_output'
    window = get_min_csv_length(input_directory)
    
    for i in range(window):
        format_input_window(input_directory, i+1, f'{output_directory}/window_{i+1}')

        if database=='adventureworks':
            Helper.stratify_adventure(f'{output_directory}/window_{i+1}/trai.csv')
            Helper.stratify_adventure(f'{output_directory}/window_{i+1}/val.csv')

if __name__=="__main__":
    main()
