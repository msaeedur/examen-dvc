from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging
from pathlib import Path
import os

def main():
    """ Runs data normalization scripts to turn splited data from (../processed_data) into
        normalized data (saved in../processed_data).
    """
    logger = logging.getLogger(__name__)
    logger.info('Normalize splitted dataset from processed_data')

    project_dir = Path(__file__).resolve().parents[2]

    X_train_path = f"{project_dir}/data/processed_data/X_train.csv"
    X_test_path = f"{project_dir}/data/processed_data/X_test.csv"
    output_folderpath = f"{project_dir}/data/processed_data"

    process_data(X_train_path,X_test_path,output_folderpath)

def process_data(X_train_path,X_test_path,output_folderpath):
    X_train = import_dataset(X_train_path, sep=",", encoding='utf-8')
    X_test = import_dataset(X_test_path, sep=",", encoding='utf-8')
    
    X_train_scaled, X_test_scaled = normalize_features(X_train,X_test)

    Path(output_folderpath).mkdir(parents=True, exist_ok=True)

    save_dataframes(X_train_scaled, X_test_scaled, output_folderpath)

def normalize_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if isinstance(X_train, pd.DataFrame):
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def save_dataframes(X_train, X_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()