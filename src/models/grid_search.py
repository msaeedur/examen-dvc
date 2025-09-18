import logging
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import yaml

project_dir = Path(__file__).resolve().parents[2]
params_file_path = f"{project_dir}/params.yaml"

def main():
    """ Runs grid search to find the best model parms
    """
    logger = logging.getLogger(__name__)
    logger.info('Runs grid search to find the best RandomForestRegressor model parms')

    X_train_path = f"{project_dir}/data/processed_data/X_train_scaled.csv"
    y_train_path = f"{project_dir}/data/processed_data/y_train.csv"
    output_folderpath = f"{project_dir}/models/"

    X_train = import_dataset(X_train_path, sep=",", encoding='utf-8')
    y_train = import_dataset(y_train_path, sep=",", encoding='utf-8').values.ravel()
    
    find_and_save_best_parms(X_train, y_train,output_folderpath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def ensure_models_dir(output_folderpath):
    models_dir = Path(output_folderpath)
    models_dir.mkdir(parents=True, exist_ok=True)

def get_model_and_grid(model_params):

    model = RandomForestRegressor(random_state=model_params['random_state'])
    param_grid = model_params['param_grid']

    # Convert YAML 'null' to Python None
    if 'max_depth' in param_grid:
        param_grid['max_depth'] = [
            None if v == 'null' else v for v in param_grid['max_depth']
        ]
        
    return model, param_grid

def run_gridsearch(X_train, y_train):
    params = load_params(params_file_path)
    model_params = params['grid_search']

    model, param_grid = get_model_and_grid(model_params)
    grid = GridSearchCV(
        model,
        param_grid,
        cv=model_params['cv'],
        scoring=model_params['scoring'],
        refit=model_params['refit'],
        n_jobs=model_params['n_jobs']
    )
    logging.info("Starting GridSearchCV...")
    grid.fit(X_train, y_train)

    best_mse = -grid.cv_results_['mean_test_neg_mean_squared_error'][grid.best_index_]
    best_r2 = grid.cv_results_['mean_test_r2'][grid.best_index_]

    logging.info(f"Best MSE: {best_mse:.4f}")
    logging.info(f"Best R2: {best_r2:.4f}")
    logging.info(f"Best parameters: {grid.best_params_}")

    return grid

def save_best_model(grid,output_folderpath):
    ensure_models_dir(output_folderpath)
    parm_path = f"{output_folderpath}/best_model_parms.pkl"
    joblib.dump(grid.best_params_, parm_path)
    
    logging.info(f"Best parms saved to: {parm_path}")

def find_and_save_best_parms(X_train, y_train,output_folderpath):
    grid = run_gridsearch(X_train, y_train)
    save_best_model(grid,output_folderpath)

    # return grid.best_estimator_

def load_params(params_file='params.yaml'):
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()