import pandas as pd 
import numpy as np
from joblib import load
import json
from pathlib import Path

from sklearn.metrics import mean_squared_error, r2_score

def main(repo_path):
    X_test = pd.read_csv(repo_path / "data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv(repo_path / "data/processed_data/y_test.csv")
    y_test = np.ravel(y_test)

    model = load(repo_path / "models/final_model.pkl")
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "mse": mse,
        "r2": r2
    }

    predictions_df = X_test.copy()
    predictions_df['actual'] = y_test
    predictions_df['predicted'] = predictions
    predictions_df.to_csv(repo_path / "data/predictions.csv", index=False)

    metrics_path = repo_path / "metrics/scores.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)

