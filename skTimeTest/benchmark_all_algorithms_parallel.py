import sys
import os
import gc
import traceback
# Reindirizza stderr a null per silenziare i warning a livello C/C++
#sys.stderr = open(os.devnull, 'w')

import pandas as pd
from typing import List, Dict, Any
from joblib import Parallel, delayed
from datetime import datetime
from sktime.datasets import load_from_ucr_tsv_to_dataframe

# Import dei tuoi moduli benchmark
from weasel_benchmark import run_weasel_benchmark, ALGO_NAME as WEASEL_NAME
from rocket_benchmark import run_rocket_benchmark, ALGO_NAME as ROCKET_NAME
from randomIntervalClassifier_benchmark import run_ric_benchmark, ALGO_NAME as RIC_RF_NAME
from boss_benchmark import run_boss_benchmark, ALGO_NAME as BOSS_NAME
from ShapeletTransformClassifier_benchmark import run_stc_benchmark, ALGO_NAME as STC_NAME
from drCIF_benchmark import run_drcif_benchmark, ALGO_NAME as DRCIF_NAME
from arsenal_benchmark import run_arsenal_benchmark, ALGO_NAME as ARSENAL_NAME
from HC2_benchmark import run_hc2_benchmark, ALGO_NAME as HC2_NAME
from ResNetClassifier_benchmark import run_resnet_benchmark, ALGO_NAME as RESNET_NAME
from InceptionTimeClassifier_benchmark import run_inception_benchmark, ALGO_NAME as INCEPTION_NAME

# --- Configurazione Globale ---
UCR_BASE_PATH: str = os.path.join(os.getcwd(), "ucr")
SEEDS: List[int] = [0, 1, 2]

# Lista dei dataset (puoi decommentare quelli che vuoi far girare stanotte)
DATASET_NAMES: List[str] = [
    "Crop",
    "Chinatown",
    "DiatomSizeReduction",
    "ElectricDevices",
    "FordB",
    "Fungi",
    "HandOutlines",
    "HouseTwenty",
    "InsectEPGSmallTrain",
    "ItalyPowerDemand",
    "Rock",
    "SmoothSubspace",#
]

ALGORITHMS_TO_RUN = [
    {
        "name": INCEPTION_NAME,
        "variant": "Inception_Standard",
        "params": {"n_epochs": 1500, "batch_size": 64}
    },
    {
        "name": RESNET_NAME,
        "variant": "ResNet_Standard",
        "params": {"n_epochs": 1500, "batch_size": 16}
    },
{
        "name": ROCKET_NAME,
        "variant": "ROCKET_Standard",
        "params": {"num_kernels": 10000}
    },
    {
        "name": ARSENAL_NAME,
        "variant": "Arsenal_Standard",
        "params": {"n_estimators": 25, "num_kernels": 2000}
    },
    {
        "name": BOSS_NAME,
        "variant": "BOSS_Standard",
        "params": {
            "max_ensemble_size": 500,
            "feature_selection": "none", # Come da documentazione standard
            "threshold": 0.92}
    },
    {
        "name": WEASEL_NAME,
        "variant": "WEASEL_Standard",
        "params": {"window_inc": 2, "p_threshold": 0.05}
    },
]

all_benchmark_results: List[Dict[str, Any]] = []


def load_all_ucr_datasets(dataset_names: List[str], base_path: str) -> Dict[str, Dict[str, Any]]:
    loaded = {}
    
    print(f"Inizio caricamento e normalizzazione dei dataset da: {base_path}")
    for name in dataset_names:
        try:
            train_path = os.path.join(base_path, name, f"{name}_TRAIN.tsv")
            test_path = os.path.join(base_path, name, f"{name}_TEST.tsv")
            X_train, y_train = load_from_ucr_tsv_to_dataframe(train_path)
            X_test, y_test = load_from_ucr_tsv_to_dataframe(test_path)

            # --- Z-NORMALIZZAZIONE MANUALE (Instance-wise) ---
            def z_normalize_panel(X):
                # X è un DataFrame sktime (nested: ogni cella è una Series)
                # Facciamo una copia per evitare problemi di puntatori
                X_norm = X.copy()
                for i in range(len(X_norm)):
                    for j in range(len(X_norm.columns)):
                        series = X_norm.iloc[i, j]
                        m = series.mean()
                        s = series.std()
                        # Se la serie è costante (std=0), sottraiamo solo la media
                        if s == 0:
                            X_norm.iloc[i, j] = series - m
                        else:
                            X_norm.iloc[i, j] = (series - m) / s
                return X_norm

            X_train = z_normalize_panel(X_train)
            X_test = z_normalize_panel(X_test)
            # ------------------------------------------------

            loaded[name] = {
                "X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test,
                "train_size": X_train.shape[0],
                "series_length": X_train.iloc[0, 0].shape[0],
                "num_classes": len(set(y_train)),
            }
            print(f" Caricato: {name} (L={loaded[name]['series_length']})")
        except Exception as e:
            print(f" Errore caricamento {name}: {e}")
    return loaded


def run_specific_benchmark(dataset_name: str, data: Dict[str, Any], seed: int, algo_cfg: Dict[str, Any]):
    name = algo_cfg["name"]
    variant = algo_cfg["variant"]
    params = algo_cfg["params"]
    # --- SILENZIATORE PER IL WORKER PARALLELO ---
    import sys
    import os
    import warnings
    
    # Chiude il canale dei messaggi di errore per questo specifico processo
    #sys.stderr = open(os.devnull, 'w')
    # Ignora i warning a livello Python
    #warnings.filterwarnings("ignore")
    # --------------------------------------------

   # Smistamento degli algoritmi con passaggio parametri
    if name == ROCKET_NAME:
        return run_rocket_benchmark(dataset_name, data, seed, variant, **params)
    elif name == ARSENAL_NAME:
        return run_arsenal_benchmark(dataset_name, data, seed, variant, **params)
    elif name == BOSS_NAME:
        return run_boss_benchmark(dataset_name, data, seed, variant, **params)
    elif name == DRCIF_NAME:
        return run_drcif_benchmark(dataset_name, data, seed, variant, **params)
    elif name == HC2_NAME:
        return run_hc2_benchmark(dataset_name, data, seed, variant, **params)
    elif name == RIC_RF_NAME:
        return run_ric_benchmark(dataset_name, data, seed, variant, **params)
    elif name == STC_NAME:
        return run_stc_benchmark(dataset_name, data, seed, variant, **params)
    elif name == WEASEL_NAME:
        return run_weasel_benchmark(dataset_name, data, seed, variant, **params)
    elif name == INCEPTION_NAME:
        return run_inception_benchmark(dataset_name, data, seed, variant, **params)
    elif name == RESNET_NAME:
        return run_resnet_benchmark(dataset_name, data, seed, variant, **params)
    else:
        raise ValueError(f"Algoritmo '{name}' non supportato o non aggiornato.")


if __name__ == "__main__":
    all_data = load_all_ucr_datasets(DATASET_NAMES, UCR_BASE_PATH)

    all_benchmark_results = []
    # Definiamo il nome del file CSV per il salvataggio incrementale
    csv_filename = "benchmark_results_FAST.csv"
    # --- PULIZIA AUTOMATICA ---
    if os.path.exists(csv_filename):
        print(f"Rilevato vecchio file {csv_filename}. Rimozione in corso per nuova run...")
        os.remove(csv_filename)
    # --------------------------
    # Registra l'ora di inizio assoluta
    start_global = datetime.now()
    print(f"\nAVVIO SESSIONE BENCHMARK: {start_global.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    for algo_cfg in ALGORITHMS_TO_RUN:
        algo_name = algo_cfg["name"]
        variant_name = algo_cfg["variant"]
        algo_start = datetime.now()
        print(f"\n{'='*70}\n ESECUZIONE: {variant_name}\n{'='*70}", flush=True)
        print(f" ORA INIZIO: {algo_start.strftime('%H:%M:%S')}", flush=True)

        for dataset_name, data in all_data.items():
            ds_start = datetime.now()
            print(f"Dataset: {dataset_name} | Esecuzione sequenziale dei {len(SEEDS)} Seed...", flush=True)

            dataset_runs = []

            # Sostituiamo Joblib con un ciclo for standard
            for seed in SEEDS:
                seed_start = datetime.now()
                # Gestione n_jobs per l'algoritmo
                # Se Deep Learning (ResNet/Inception), TF gestisce i thread da solo.
                # Per gli altri, diciamo all'algoritmo di usare tutti i core (-1).
                if algo_name in [RESNET_NAME, INCEPTION_NAME]:
                    algo_cfg["params"]["n_jobs"] = 1
                else:
                    algo_cfg["params"]["n_jobs"] = -1

                try: #proteggiamo ogni singolo seed con try catch
                    # Esecuzione diretta del benchmark per il singolo seed
                    res = run_specific_benchmark(dataset_name, data, seed, algo_cfg)
                    dataset_runs.append(res)

                    # 2. SALVATAGGIO INCREMENTALE SUL CSV
                    df_row = pd.DataFrame([res])
                    # mode='a' aggiunge la riga; header viene scritto solo se il file non esiste ancora
                    df_row.to_csv(csv_filename, mode='a', index=False, header=not os.path.exists(csv_filename))

                    # Feedback immediato per il seed appena concluso
                    print(f" Seed {res['seed']} | Acc: {res['accuracy']:.4f} | Fit: {res['fit_time']:.2f}s | Pred: {res['predict_time']:.2f}s", flush=True)

                except Exception as e: # <--- Cattura qualsiasi errore (Memoria, Valore, Sistema)
                    print(f"\n[!!!] ERRORE durante {variant_name} su {dataset_name} (Seed {seed}):", flush=True)
                    print(f"Dettaglio errore: {e}", flush=True)
                    traceback.print_exc() # Stampa l'errore completo con la riga esatta nel codice
                    # Non aggiungiamo nulla a dataset_runs, quindi questo seed verrà saltato nel riassunto.
                    # Il programma NON si ferma, passerà al prossimo seed o dataset.
                    continue

            all_benchmark_results.extend(dataset_runs)

            ds_end = datetime.now()
            print(f"Tempo Reale Totale per {dataset_name}: {ds_end - ds_start}", flush=True)
            print("-" * 30, flush=True)
            del dataset_runs # Libera la lista di risultati in memoria
            gc.collect()     # Forza la pulizia della RAM

        algo_end = datetime.now()
        print(f"\n--- COMPLETATO: {variant_name} ---", flush=True) 
        print(f" Durata Totale Algoritmo: {algo_end - algo_start}", flush=True)
        print("=" * 70, flush=True)

    # --- RIASSUNTO FINALE ---
    # --- 1. CREAZIONE DATAFRAME TOTALE ---
    final_df = pd.DataFrame(all_benchmark_results)

    # --- 3. LOGICA DEL RIASSUNTO ---
    metrics_to_agg = ['accuracy', 'f1_score', 'fit_time', 'predict_time', 'total_time_seconds']

    # Invece di drop_duplicates generico, raggruppiamo per dataset 
    # e prendiamo il valore massimo (o il primo) per ogni metadato.
    # Questo garantisce UNA riga per dataset.
    metadata_df = final_df.groupby('dataset')[['train_size', 'series_length', 'num_classes']].max()

    # Calcoliamo medie e deviazioni standard
    performance_summary = final_df.groupby(['dataset', 'variant'])[metrics_to_agg].agg(['mean', 'std'])
    performance_summary.columns = ['_'.join(col).strip() for col in performance_summary.columns.values]

    # Uniamo: ora il join sarà 1-a-1 per ogni coppia (dataset, variant)
    final_summary_combined = performance_summary.reset_index(level='variant').join(metadata_df)

    # Definiamo l'ordine delle colonne includendo TUTTE le medie e le deviazioni standard
    STANDARD_ORDER = [
        'variant', 'train_size', 'series_length', 'num_classes',
        'accuracy_mean', 'accuracy_std', 
        'f1_score_mean', 'f1_score_std',
        'fit_time_mean', 'predict_time_mean', 
        'total_time_seconds_mean', 'total_time_seconds_std' # <--- AGGIUNTA QUI
    ]
    
    final_summary_combined = final_summary_combined.reindex(columns=STANDARD_ORDER)

    # --- 4. STAMPA FINALE ---
    print("\n" + "#" * 100, flush=True)
    print(" RIASSUNTO GENERALE (Medie e Deviazione Standard sui Seed)", flush=True)
    print("#" * 100, flush=True)
    print(final_summary_combined.to_markdown(floatfmt=".4f"))