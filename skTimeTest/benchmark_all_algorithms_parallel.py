import sys
import os
# Reindirizza stderr a null per silenziare i warning a livello C/C++
#sys.stderr = open(os.devnull, 'w')

import pandas as pd
from typing import List, Dict, Any
from joblib import Parallel, delayed
from datetime import datetime

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
    #"Crop",
    "Chinatown",
    #"DiatomSizeReduction",
    #"ElectricDevices",
    #"FordB",
    #"Fungi",
    #"HandOutlines",
    #"HouseTwenty",
    #"InsectEPGSmallTrain",
    #"ItalyPowerDemand",
    #"Rock",
    #"SmoothSubspace",#
]

ALGORITHMS_TO_RUN = [
    {
        "name": ROCKET_NAME, 
        "variant": "ROCKET_Super_Lite", 
        "params": {"num_kernels": 1000}
    },
    #
    #    "name": ROCKET_NAME, 
    #    "variant": "ROCKET_Lite", 
    #    "params": {"num_kernels": 5000}
    #},
    #{
    #    "name": ROCKET_NAME, 
    #    "variant": "ROCKET_Standard", 
    #    "params": {"num_kernels": 10000}
    #},
    #{
    #    "name": ARSENAL_NAME, 
    #    "variant": "Arsenal_Standard", 
    #    "params": {"n_estimators": 25, "num_kernels": 2000} # Veloce, per test
    #},
    #{
    #    "name": ARSENAL_NAME, 
    #    "variant": "Arsenal_Lite", 
    #    "params": {"n_estimators": 10, "num_kernels": 1000} # Veloce, per test
    #},
    {
        "name": ARSENAL_NAME, 
        "variant": "Arsenal_Super_Lite", 
        "params": {"n_estimators": 5, "num_kernels": 500} # Veloce, per test
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
        "name": BOSS_NAME, 
        "variant": "BOSS_Lite", 
        "params": {
            "max_ensemble_size": 50,
            "feature_selection": "chi2", # Molto più veloce e meno RAM
            "threshold": 0.95,
            "min_window": 20}
    },
    {
        "name": BOSS_NAME, 
        "variant": "BOSS_Super_Lite", 
        "params": {
            "max_ensemble_size": 5,
            "feature_selection": "chi2",
            "threshold": 0.99,
            "min_window": 50}
    },
    #{
    #    "name": DRCIF_NAME, 
    #    "variant": "DrCIF_Standard", 
    #    "params": {"n_estimators": 200, "att_subsample_size": 10}
    #},
    #{
    #    "name": DRCIF_NAME, 
    #    "variant": "DrCIF_Light", 
    #    "params": {"n_estimators": 50, "att_subsample_size": 5, "n_intervals": 4}
    #},
    {
        "name": DRCIF_NAME, 
        "variant": "DrCIF_Super_Light", 
        "params": {"n_estimators": 10, "att_subsample_size": 3, "n_intervals": 2}
    },
    #{
    #    "name": HC2_NAME, 
    #    "variant": "HC2_Standard", 
    #    "params": {
    #        "stc_params": {"n_shapelet_samples": 10000, "max_shapelets": None},
    #        "drcif_params": {"n_estimators": 200, "att_subsample_size": 10},
    #        "arsenal_params": {"num_kernels": 2000, "n_estimators": 25},
    #        "tde_params": {"n_parameter_samples": 250, "max_ensemble_size": 50}
    #    }
    #},
    #{
    #    "name": HC2_NAME, 
    #    "variant": "HC2_Lite", 
    #    "params": {
    #        "stc_params": {"n_shapelet_samples": 1000, "max_shapelets": 200},
    #        "drcif_params": {"n_estimators": 50, "att_subsample_size": 5, "n_intervals": 4},
    #        "arsenal_params": {"num_kernels": 1000, "n_estimators": 10},
    #        "tde_params": {"n_parameter_samples": 100, "max_ensemble_size": 10}
    #    }
    #},
    {
        "name": HC2_NAME, 
        "variant": "HC2_Super_Lite", 
        "params": {
            "stc_params": {"n_shapelet_samples": 100, "max_shapelets": 50},
            "drcif_params": {"n_estimators": 10, "att_subsample_size": 3, "n_intervals": 2},
            "arsenal_params": {"num_kernels": 500, "n_estimators": 5},
            "tde_params": {"n_parameter_samples": 51, "max_ensemble_size": 5}
        }
    },
    #{
    #    "name": RIC_RF_NAME, 
    #    "variant": "RIC_Standard", 
    #    "params": {"n_intervals": 100, "n_estimators": 200}
    #},
    #{
    #    "name": RIC_RF_NAME, 
    #    "variant": "RIC_Lite", 
    #    "params": {"n_intervals": 20, "n_estimators": 50} 
    {
        "name": RIC_RF_NAME, 
        "variant": "RIC_Super_Lite", 
        "params": {"n_intervals": 10, "n_estimators": 10} 
    },
    #{
    #    "name": STC_NAME, 
    #    "variant": "STC_Standard", 
    #    "params": {
    #        "n_shapelet_samples": 10000, "n_estimators": 200,
    #        "max_shapelets": None}
    #},
    #{
    #    "name": STC_NAME, 
    #    "variant": "STC_Light", 
    #    "params": {
    #        "n_shapelet_samples": 1000, "n_estimators": 50,
    #        "max_shapelets": 200}
    #},
    {
        "name": STC_NAME, 
        "variant": "STC_Super_Lite", 
        "params": {
            "n_shapelet_samples": 100, "n_estimators": 10,
            "max_shapelets": 50}
    },
    #{
    #    "name": WEASEL_NAME, 
    #    "variant": "WEASEL_Standard", 
    #    "params": {"window_inc": 2, "p_threshold": 0.05} 
    #},
    #{
    #    "name": WEASEL_NAME, 
    #    "variant": "WEASEL_Lite", 
    #    "params": {"window_inc": 6, "p_threshold": 0.01} 
    #},
    {
        "name": WEASEL_NAME, 
        "variant": "WEASEL_Super_Lite", 
        "params": {"window_inc": 12, "p_threshold": 0.001} 
    },
    #{
    #    "name": INCEPTION_NAME, 
    #    "variant": "Inception_Standard", 
    #    "params": {"n_epochs": 1500, "batch_size": 64} 
    #},
    #{
    #    "name": INCEPTION_NAME, 
    #    "variant": "Inception_Lite", 
    #    "params": {"n_epochs": 150, "batch_size": 64, "kernel_size":20, "n_filters":16} 
    #},
    {
       "name": INCEPTION_NAME, 
        "variant": "Inception_Super_Lite", 
        "params": {"n_epochs": 15, "batch_size": 64, 
                   "kernel_size":10, "n_filters":8
                   } 
    },
    #{
    #    "name": RESNET_NAME, 
    #    "variant": "ResNet_Standard", 
    #    "params": {"n_epochs": 1500, "batch_size": 16} 
    #},
    #{
    #    "name": RESNET_NAME, 
    #    "variant": "ResNet_Lite", 
    #    "params": {"n_epochs": 150, "batch_size": 32} 
    #},
    {
        "name": RESNET_NAME, 
        "variant": "ResNet_Super_Lite", 
        "params": {"n_epochs": 15, "batch_size": 64}
    }
]

all_benchmark_results: List[Dict[str, Any]] = []


def load_all_ucr_datasets(dataset_names: List[str], base_path: str) -> Dict[str, Dict[str, Any]]:
    from sktime.datasets import load_from_ucr_tsv_to_dataframe
    loaded = {}
    print(f"Inizio caricamento dei dataset da: {base_path}")
    for name in dataset_names:
        try:
            train_path = os.path.join(base_path, name, f"{name}_TRAIN.tsv")
            test_path = os.path.join(base_path, name, f"{name}_TEST.tsv")
            X_train, y_train = load_from_ucr_tsv_to_dataframe(train_path)
            X_test, y_test = load_from_ucr_tsv_to_dataframe(test_path)

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
    # Registra l'ora di inizio assoluta
    start_global = datetime.now()
    print(f"\nAVVIO SESSIONE BENCHMARK: {start_global.strftime('%Y-%m-%d %H:%M:%S')}")

    all_data = load_all_ucr_datasets(DATASET_NAMES, UCR_BASE_PATH)

    all_benchmark_results = []

    for algo_cfg in ALGORITHMS_TO_RUN:
        algo_name = algo_cfg["name"]
        variant_name = algo_cfg["variant"]
        algo_start = datetime.now()
        print(f"\n{'='*70}\n ESECUZIONE: {variant_name}\n{'='*70}")
        print(f" ORA INIZIO: {algo_start.strftime('%H:%M:%S')}")

        for dataset_name, data in all_data.items():
            ds_start = datetime.now()
            # SE l'algoritmo è ResNet, forziamo n_jobs a 1 nel main
            # perché TensorFlow usa già tutti i core CPU/GPU internamente.
            actual_n_jobs = 1 if algo_name in [RESNET_NAME, INCEPTION_NAME] else 3

            print(f"Dataset: {dataset_name} | Parallelizzazione su {len(SEEDS)} Seed...")

            # --- PARALLELIZZAZIONE ESTERNA ---
            # Esegue i 3 seed contemporaneamente
            dataset_runs = Parallel(n_jobs=actual_n_jobs)(
                delayed(run_specific_benchmark)(dataset_name, data, seed, algo_cfg)
                for seed in SEEDS
            )

            all_benchmark_results.extend(dataset_runs)

            # Stampa i dettagli per ogni singolo seed appena concluso
            # Print di feedback immediato
            for res in dataset_runs:
                print(f" Seed {res['seed']} | Acc: {res['accuracy']:.4f} | Fit: {res['fit_time']:.2f}s | Pred: {res['predict_time']:.2f}s")

            ds_end = datetime.now()
            print(f"Tempo Reale Totale per {dataset_name}: {ds_end - ds_start}")
            print("-" * 30)

        algo_end = datetime.now()
        print(f"\n--- COMPLETATO: {variant_name} ---") 
        print(f" Durata Totale Algoritmo: {algo_end - algo_start}")
        print("=" * 70)

    # --- RIASSUNTO FINALE ---
    final_df = pd.DataFrame(all_benchmark_results)

    # --- 1. CREAZIONE DATAFRAME TOTALE ---
    final_df = pd.DataFrame(all_benchmark_results)

    # --- 2. SALVATAGGIO CSV DETTAGLIATO (Fondamentale per i grafici futuri) ---
    # Questo salva ogni singola riga (ogni seed) con tutte le colonne nuove
    final_df.to_csv("benchmark_results_detailed.csv", index=False)

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
    print("\n" + "#" * 100) # Allunghiamo ancora un po'
    print(" RIASSUNTO GENERALE (Medie e Deviazione Standard sui Seed)")
    print("#" * 100)
    print(final_summary_combined.to_markdown(floatfmt=".4f"))