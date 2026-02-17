import time
import json
from time import time_ns

import numpy as np
from sktime.classification.interval_based import DrCIF
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any
from sktime.datatypes import convert_to

ALGO_NAME: str = "drCIF"

def run_drcif_benchmark(dataset_name: str, data: Dict[str, Any], seed: int, 
                        variant: str = "Standard", **kwargs) -> Dict[str, Any]:
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    # --- FIX PER SKTIME/NUMBA ---
    # Convertiamo in numpy3D float64 per massimizzare la velocità con Numba
    X_train = convert_to(X_train, to_type="numpy3D").astype('float64')
    X_test = convert_to(X_test, to_type="numpy3D").astype('float64')

    # Estrazione parametri con i default bilanciati che avevi scelto
    n_estimators = kwargs.get("n_estimators", 100)
    att_subsample_size = kwargs.get("att_subsample_size", 10)
    n_intervals = kwargs.get("n_intervals", None)
    time_limit = kwargs.get("time_limit_in_minutes", 360)#6 ore

    classifier = DrCIF(
        n_estimators=n_estimators,
        n_intervals=n_intervals,  # La formula automatica è la scelta migliore per la tesi
        att_subsample_size=att_subsample_size,
        n_jobs=-1,          # Parallelismo gestito dai seed nel main
        random_state=seed,
        time_limit_in_minutes = time_limit,
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