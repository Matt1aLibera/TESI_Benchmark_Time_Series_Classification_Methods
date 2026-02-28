import time
import json
import numpy as np
from tensorflow.keras.optimizers import Adam
from sktime.classification.deep_learning.resnet import ResNetClassifier
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any
import tensorflow as tf

ALGO_NAME: str = "ResNet"

def run_resnet_benchmark(dataset_name: str, data: Dict[str, Any], seed: int, 
                         variant: str = "Standard", **kwargs) -> Dict[str, Any]:
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    # Estrazione parametri (Default: Paper Wang et al. 2017)
    epochs = kwargs.get("n_epochs", 1500)
    batch = kwargs.get("batch_size", 16)
    learning_rate = kwargs.get("learning_rate", 0.01)
    # Configurazione conservativa per dataset "difficili" come ElectricDevices
    new_optimizer = Adam(learning_rate=learning_rate)

    classifier = ResNetClassifier(
        n_epochs=epochs,
        batch_size=batch,
        random_state=seed,
        verbose=False,
        activation='sigmoid',
        activation_hidden='relu',
        use_bias=True,
        optimizer = new_optimizer
    )

    # Addestramento (Fit)
    start_fit = time.time()
    classifier.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    # Previsione (Predict)
    start_pred = time.time()
    y_pred = classifier.predict(X_test)
    predict_time = time.time() - start_pred
    # PULIZIA FINALE: Libera la memoria del grafo di TensorFlow
    tf.keras.backend.clear_session()

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