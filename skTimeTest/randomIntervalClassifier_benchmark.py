import time
import json
import numpy as np
from sktime.classification.feature_based import RandomIntervalClassifier
from sktime.classification.sklearn import RotationForest
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any
from sktime.datatypes import convert_to

ALGO_NAME: str = "RIC_RF"

def run_ric_benchmark(dataset_name: str, data: Dict[str, Any], seed: int, 
                      variant: str = "Standard", **kwargs) -> Dict[str, Any]:
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    # --- FIX PER IL FORMATO DATI (Numba friendly) ---
    X_train = convert_to(X_train, to_type="numpy3D").astype('float64')
    X_test = convert_to(X_test, to_type="numpy3D").astype('float64')

    # Estrazione parametri dalle varianti
    n_intervals = kwargs.get("n_intervals", 100)
    rf_estimators = kwargs.get("n_estimators", 200)

    # Inizializzazione Rotation Forest come base estimator
    base_estimator = RotationForest(
        n_estimators=rf_estimators,
        random_state=seed,
        n_jobs=1,
    )

    # Inizializzazione Random Interval Classifier
    classifier = RandomIntervalClassifier(
        n_intervals=n_intervals,
        estimator=base_estimator,
        random_state=seed,
        n_jobs=-1,
    )

    # Addestramento (Fit)
    start_fit = time.time()
    classifier.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    # Previsione (Predict)
    start_pred = time.time()
    y_pred = classifier.predict(X_test)
    predict_time = time.time() - start_pred

    return {
        "dataset": dataset_name,
        "algorithm": ALGO_NAME,
        "variant": variant,
        "seed": seed,
        "hyperparameters": json.dumps(kwargs),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average='macro', zero_division=0),
        "fit_time": fit_time,
        "predict_time": predict_time,
        "total_time_seconds": fit_time + predict_time,
        "train_size": data['train_size'],
        "series_length": data['series_length'],
        "num_classes": len(np.unique(y_train))
    }