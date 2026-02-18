import time
import json
import numpy as np
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.sklearn import RotationForest
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any

ALGO_NAME: str = "STC"

def run_stc_benchmark(dataset_name: str, data: Dict[str, Any], seed: int, 
                      variant: str = "Standard", **kwargs) -> Dict[str, Any]:
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    # Estrazione parametri dalle varianti
    # Se passiamo un numero di alberi per la Rotation Forest interna
    n_est = kwargs.get("n_estimators", 200) 
    n_samples = kwargs.get("n_shapelet_samples", 10000)
    max_sh = kwargs.get("max_shapelets", 1000)
    time_limit = kwargs.get("time_limit_in_minutes", 120)

    # Definiamo l'estimator interno (Rotation Forest)
    # Se vuoi usare il default assoluto di sktime (molto pesante), metti n_estimators=200
    rotf = RotationForest(
        n_estimators=n_est,
        random_state=seed,
        n_jobs=-1
    )

    classifier = ShapeletTransformClassifier(
        estimator=rotf,
        n_shapelet_samples=n_samples,
        max_shapelets=max_sh,
        batch_size=100,
        random_state=seed,
        n_jobs=-1,  # Parallelismo gestito dal main sui seed
        time_limit_in_minutes= time_limit,
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