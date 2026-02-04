import time
import json
import numpy as np
from sktime.classification.dictionary_based import WEASEL
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any

ALGO_NAME: str = "WEASEL"

def run_weasel_benchmark(dataset_name: str, data: Dict[str, Any], seed: int, 
                         variant: str = "Standard", **kwargs) -> Dict[str, Any]:
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    series_len = data.get('series_length', X_train.shape[-1])

    # 1. Estrazione parametri dalle varianti
    w_inc = kwargs.get("window_inc", 2)
    p_th = kwargs.get("p_threshold", 0.05)
    
    # 2. Logica Adattiva per WEASEL (Paracadute)
    # WEASEL fallisce se prova a estrarre feature da finestre inconsistenti.
    # Se la serie è molto corta, forziamo window_inc a 1 per non perdere informazioni.
    effective_window_inc = w_inc
    if series_len < 30:
        effective_window_inc = 1

    classifier = WEASEL(
        anova=True,
        bigrams=True,
        binning_strategy='information-gain',
        window_inc=effective_window_inc,
        p_threshold=p_th,
        alphabet_size=2,
        feature_selection='chi2',
        support_probabilities=False,
        random_state=seed,
        n_jobs=1 
    )

    # Addestramento (Fit)
    start_fit = time.time()
    classifier.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    # Previsione (Predict)
    start_pred = time.time()
    y_pred = classifier.predict(X_test)
    predict_time = start_pred_time = time.time() - start_pred

    # Salvataggio parametri reali usati
    actual_params = kwargs.copy()
    actual_params["effective_window_inc"] = effective_window_inc

    return {
        "dataset": dataset_name,
        "algorithm": ALGO_NAME,
        "variant": variant,
        "seed": seed,
        "hyperparameters": json.dumps(actual_params),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average='macro', zero_division=0),
        "fit_time": fit_time,
        "predict_time": predict_time,
        "total_time_seconds": fit_time + predict_time,
        "train_size": data.get('train_size', len(X_train)),
        "series_length": series_len,
        "num_classes": len(np.unique(y_train))
    }