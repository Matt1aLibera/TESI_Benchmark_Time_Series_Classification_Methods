import sys
import os
import gc
import traceback


import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from sktime.datasets import load_from_ucr_tsv_to_dataframe

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

UCR_BASE_PATH: str = os.path.join(os.getcwd(), "ucr")
SEEDS: List[int] = [0, 1, 2]

# Lista dei dataset
DATASET_NAMES: List[str] = [
    "InsectEPGSmallTrain",
    "ItalyPowerDemand",
    "SmoothSubspace",#
    "Fungi",
    "DiatomSizeReduction",
    "Chinatown",
    "HouseTwenty",
    "Rock",
    "HandOutlines",
    "Crop",
    "FordB",
    "ElectricDevices",
]

ALGORITHMS_TO_RUN = [
    {
        "name": RIC_RF_NAME,
        "variant": "RIC_Standard",
        "params": {"n_intervals": 100, "n_estimators": 200}
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

            # Z-NORMALIZZAZIONE (Instance-wise) ---
            def z_normalize_panel(X):
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

# Chiama la funzione di benchmark appropriata in base al nome dell'algoritmo.
def run_specific_benchmark(dataset_name: str, data: Dict[str, Any], seed: int, algo_cfg: Dict[str, Any]):
    name = algo_cfg["name"]
    variant = algo_cfg["variant"]
    params = algo_cfg["params"]
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
    csv_filename = "benchmark_results_SLOWRIC.csv"
    # Pulizia automatica
    if os.path.exists(csv_filename):
        print(f"Rilevato vecchio file {csv_filename}. Rimozione in corso per nuova run...")
        os.remove(csv_filename)

    # Registra l'ora di inizio assoluta
    start_global = datetime.now()
    print(f"\nAVVIO SESSIONE BENCHMARK: {start_global.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    for algo_cfg in ALGORITHMS_TO_RUN:
        algo_name = algo_cfg["name"]
        variant_name = algo_cfg["variant"]

        #LOGICA SPECIFICA PER HIVE-COTE 2.0
        is_hc2 = (algo_name == HC2_NAME)

        algo_start = datetime.now()
        print(f"\n{'='*70}\n ESECUZIONE: {variant_name}\n{'='*70}", flush=True)
        print(f" ORA INIZIO: {algo_start.strftime('%H:%M:%S')}", flush=True)

        for dataset_name, data in all_data.items():

            # 1. Se HC2 e il dataset è Crop, salta completamente
            if is_hc2 and dataset_name.upper() == "CROP":
                print(f"\n[SKIP] Saltando Crop per {variant_name} (evitiamo freeze RAM/NFS)", flush=True)
                continue

            ds_start = datetime.now()

            # 2. Se HC2, eseguiamo solo il PRIMO seed della tua lista SEEDS
            current_seeds = [SEEDS[0]] if is_hc2 else SEEDS

            print(f"Dataset: {dataset_name} | Esecuzione sequenziale dei {len(SEEDS)} Seed...", flush=True)

            dataset_runs = []

            #Loop sui seed
            for seed in current_seeds:
                seed_start = datetime.now()
                if algo_name in [RESNET_NAME, INCEPTION_NAME]:
                    algo_cfg["params"]["n_jobs"] = 1
                else:
                    algo_cfg["params"]["n_jobs"] = -1

                try:
                    res = run_specific_benchmark(dataset_name, data, seed, algo_cfg)
                    dataset_runs.append(res)

                    # SALVATAGGIO INCREMENTALE SUL CSV
                    df_row = pd.DataFrame([res])
                    df_row.to_csv(csv_filename, mode='a', index=False, header=not os.path.exists(csv_filename))

                    print(f" Seed {res['seed']} | Acc: {res['accuracy']:.4f} | Fit: {res['fit_time']:.2f}s | Pred: {res['predict_time']:.2f}s", flush=True)

                except Exception as e:
                    print(f"\n[!!!] ERRORE durante {variant_name} su {dataset_name} (Seed {seed}):", flush=True)
                    print(f"Dettaglio errore: {e}", flush=True)
                    traceback.print_exc()
                    continue

            all_benchmark_results.extend(dataset_runs)

            ds_end = datetime.now()
            print(f"Tempo Reale Totale per {dataset_name}: {ds_end - ds_start}", flush=True)
            print("-" * 30, flush=True)
            del dataset_runs
            gc.collect()

        algo_end = datetime.now()
        print(f"\n--- COMPLETATO: {variant_name} ---", flush=True) 
        print(f" Durata Totale Algoritmo: {algo_end - algo_start}", flush=True)
        print("=" * 70, flush=True)

    # RIASSUNTO FINALE
    final_df = pd.DataFrame(all_benchmark_results)

    metrics_to_agg = ['accuracy', 'f1_score', 'fit_time', 'predict_time', 'total_time_seconds']

    metadata_df = final_df.groupby('dataset')[['train_size', 'series_length', 'num_classes']].max()

    # Calcoliamo medie e deviazioni standard
    performance_summary = final_df.groupby(['dataset', 'variant'])[metrics_to_agg].agg(['mean', 'std'])
    performance_summary.columns = ['_'.join(col).strip() for col in performance_summary.columns.values]

    final_summary_combined = performance_summary.reset_index(level='variant').join(metadata_df)

    # Definiamo l'ordine delle colonne
    STANDARD_ORDER = [
        'variant', 'train_size', 'series_length', 'num_classes',
        'accuracy_mean', 'accuracy_std', 
        'f1_score_mean', 'f1_score_std',
        'fit_time_mean', 'predict_time_mean', 
        'total_time_seconds_mean', 'total_time_seconds_std'
    ]
    
    final_summary_combined = final_summary_combined.reindex(columns=STANDARD_ORDER)

    # STAMPA FINALE
    print("\n" + "#" * 100, flush=True)
    print(" RIASSUNTO GENERALE (Medie e Deviazione Standard sui Seed)", flush=True)
    print("#" * 100, flush=True)
    print(final_summary_combined.to_markdown(floatfmt=".4f"))