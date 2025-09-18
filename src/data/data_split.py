from pathlib import Path
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import os
import yaml

project_dir = Path(__file__).resolve().parents[2]

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        splitted data ready to be analyzed (saved in../processed_data).
    """
    logger = logging.getLogger(__name__)
    logger.info('Splitting raw dataset from raw data')

    input_filepath_raw = f"{project_dir}/data/raw_data/raw.csv"
    output_filepath = f"{project_dir}/data/processed_data"

    process_data(input_filepath_raw, output_filepath)

def process_data(input_filepath_raw, output_filepath):
    df = import_dataset(input_filepath_raw, sep=",", encoding='utf-8')
    df = df.drop(columns=['date'], errors='ignore')

    X_train, X_test, y_train, y_test = split_data(df)

    create_folder_if_necessary(output_filepath)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train, X_test, y_train, y_test, output_filepath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def split_data(df):
    # Split data into training and testing sets
    params_file_path = f"{project_dir}/params.yaml"
    params = load_params(params_file_path)
    test_size = params['split']['test_size']
    random_state = params['split']['random_state']

    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    Path(output_folderpath).mkdir(parents=True, exist_ok=True)

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

def load_params(params_file='params.yaml'):
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()