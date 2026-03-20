import time
import json
from sktime.classification.kernel_based import RocketClassifier as Rocket
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any

ALGO_NAME: str = "Rocket"

def run_rocket_benchmark(dataset_name: str, data: Dict[str, Any], seed: int, 
                         variant: str = "Standard", **kwargs) -> Dict[str, Any]:

    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    n_kernels = kwargs.get("num_kernels", 10000)

    classifier = Rocket(
        num_kernels=n_kernels,
        random_state=seed,
        n_jobs=-1
    )

    start_fit = time.time()
    classifier.fit(X_train, y_train)
    fit_time = time.time() - start_fit

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
        "num_classes": len(set(y_train))
    }