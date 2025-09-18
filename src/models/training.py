import joblib
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def main():
    """ Runs model training script using best parameters from GridSearch.
        Trains final model and saves to ../models/final_model.pkl
    """
    logger = logging.getLogger(__name__)
    logger.info('Training final model using best parameters')

    project_dir = Path(__file__).resolve().parents[2]

    # Load data
    X_train_path = f"{project_dir}/data/processed_data/X_train_scaled.csv"
    y_train_path = f"{project_dir}/data/processed_data/y_train.csv"

    X_train = import_dataset(X_train_path)
    y_train = import_dataset(y_train_path).values.ravel()

    # Load best params
    best_params_path = f"{project_dir}/models/best_model_parms.pkl"
    best_params = joblib.load(best_params_path)
    
    # Train and save
    final_model = train_and_save_final_model(
        X_train, y_train, best_params, 
        model_save_path=f"{project_dir}/models/final_model.pkl"
    )

def train_and_save_final_model(X_train, y_train, best_params, model_save_path):
    ensure_models_dir(model_save_path)
    print(best_params)
    model = RandomForestRegressor(**best_params, random_state=87)
    model.fit(X_train, y_train)
    joblib.dump(model, model_save_path)
    logging.getLogger(__name__).info(f"Final model trained and saved to {model_save_path}")
    return model

def ensure_models_dir(model_save_path):
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()