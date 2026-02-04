import time
import json
import numpy as np
from sktime.classification.hybrid import HIVECOTEV2
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any

ALGO_NAME: str = "HC2"

def run_hc2_benchmark(dataset_name: str, data: Dict[str, Any], seed: int, 
                      variant: str = "Standard", **kwargs) -> Dict[str, Any]:
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    # Estrazione parametri dal dizionario varianti (o default)
    # time_limit_in_minutes = 0 significa NESSUN limite di tempo.
    time_limit = kwargs.get("time_limit_in_minutes", 0)
    
    # Se passiamo parametri specifici nel dizionario, HC2 li userà.
    # Altrimenti sktime userà i default del paper (molto pesanti).
    stc_p = kwargs.get("stc_params", None)
    drcif_p = kwargs.get("drcif_params", None)
    arsenal_p = kwargs.get("arsenal_params", None)
    tde_p = kwargs.get("tde_params", None)

    classifier = HIVECOTEV2(
        stc_params=stc_p,
        drcif_params=drcif_p,
        arsenal_params=arsenal_p,
        tde_params=tde_p,
        time_limit_in_minutes=time_limit,
        n_jobs=1, # Fondamentale: teniamo 1 perché il main lancia già 3 seed paralleli
        random_state=seed,
        verbose=0
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