import time
import json
from sktime.classification.dictionary_based import BOSSEnsemble
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any

ALGO_NAME: str = "BOSS"

def run_boss_benchmark(dataset_name: str, data: Dict[str, Any], seed: int, 
                       variant: str = "Standard", **kwargs) -> Dict[str, Any]:
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    # --- LOGICA ADATTIVA (COMMENTATA PER TEST FULL DEFAULT) ---
    # if hasattr(X_train, "iloc"):
    #     series_len = len(X_train.iloc[0, 0])
    # else:
    #     series_len = X_train.shape[-1]
    #
    # if series_len < 30:
    #     calc_min_window, calc_max_prop = 10, 1.0
    # elif series_len < 50:
    #     calc_min_window, calc_max_prop = 10, 1.0
    # else:
    #     calc_min_window, calc_max_prop = 10, 1.0
    # ---------------------------------------------------------

    # Estrazione parametri con i default originali di sktime
    # 1. Prendi i parametri richiesti dal main
    series_len = data['series_length']
    requested_min_win = kwargs.get("min_window", 10)
    requested_max_prop = kwargs.get("max_win_len_prop", 1.0)
    
    # 2. LOGICA ADATTIVA (Evoluzione della tua)
    # Calcoliamo la finestra massima reale che BOSS proverà
    effective_max_win = int(series_len * requested_max_prop)
    
    # Se la finestra minima richiesta è troppo grande per questo dataset:
    if requested_min_win >= effective_max_win - 2:
        # La impostiamo al 25% della lunghezza della serie (abbastanza piccola da non crashare)
        # ma non meno di 6 punti per mantenere senso statistico
        calc_min_window = max(6, int(effective_max_win * 0.25))
    else:
        calc_min_window = requested_min_win

    # 3. CONFIGURAZIONE
    classifier = BOSSEnsemble(
        threshold=kwargs.get("threshold", 0.92),
        max_ensemble_size=kwargs.get("max_ensemble_size", 500),
        min_window=calc_min_window,
        max_win_len_prop=requested_max_prop,
        random_state=seed,
        n_jobs=1
    )

    start_fit = time.time()
    classifier.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    start_pred = time.time()
    y_pred = classifier.predict(X_test)
    predict_time = time.time() - start_pred

    # Salviamo cosa è successo realmente nel dizionario iperparametri
    actual_params_used = kwargs.copy()
    actual_params_used["effective_min_window"] = calc_min_window
    actual_params_used["effective_max_prop"] = effective_max_win

    return {
        "dataset": dataset_name,
        "algorithm": ALGO_NAME,
        "variant": variant,
        "seed": seed,
        "hyperparameters": json.dumps(actual_params_used),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average='macro', zero_division=0),
        "fit_time": fit_time,
        "predict_time": predict_time,
        "total_time_seconds": fit_time + predict_time,
        "train_size": data['train_size'],
        "series_length": series_len,
        "num_classes": len(set(y_train))
    }