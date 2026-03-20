import os
import pandas as pd
from sktime.datasets import load_from_ucr_tsv_to_dataframe
from typing import List, Dict, Any

from weasel_benchmark import run_weasel_benchmark, ALGO_NAME as WEASEL_NAME
from rocket_benchmark import run_rocket_benchmark, ALGO_NAME as ROCKET_NAME
from randomIntervalClassifier_benchmark import run_ric_benchmark, ALGO_NAME as RIC_RF_NAME
from boss_benchmark import run_boss_benchmark, ALGO_NAME as BOSS_NAME
from ShapeletTransformClassifier_benchmark import run_stc_benchmark, ALGO_NAME as STC_NAME
from drCIF_benchmark import run_drcif_benchmark, ALGO_NAME as DRCIF_NAME
from arsenal_benchmark import run_arsenal_benchmark, ALGO_NAME as ARSENAL_NAME
from HC2_benchmark import run_hc2_benchmark, ALGO_NAME as HC2_NAME

UCR_BASE_PATH: str = os.path.join(os.getcwd(), "ucr") #percorso al file dei dataet ucr
SEEDS: List[int] = [0, 1, 2]

# Lista dei nomi delle cartelle/dataset da testare
DATASET_NAMES: List[str] = [
    "Crop",
    #"Chinatown",
    #"DiatomSizeReduction",
    #"ElectricDevices",
    #"FordB",
    #"Fungi",
    #"HandOutlines",
    #"HouseTwenty",
    #"InsectEPGSmallTrain",
    #"ItalyPowerDemand",
    #"Rock",
    #"SmoothSubspace",
]

# Dizionario per salvare i dataset caricati e i metadati
# strutturato come: { 'NomeDataset': { 'X_train': ..., 'y_train': ..., 'X_test': ..., 'y_test': ... } }
loaded_datasets: Dict[str, Dict[str, Any]] = {}
all_benchmark_results: List[Dict[str, Any]] = []

# Funzione di Caricamento datasets. Carica i dataset UCR locali e calcola Train Size e Series Length.
def load_all_ucr_datasets(dataset_names: List[str], base_path: str) -> Dict[str, Dict[str, Any]]:
    print(f"Inizio caricamento dei dataset da: {base_path}")

    for name in dataset_names:
        print(f"\nCaricamento dataset: {name}...")

        train_path = os.path.join(base_path, name, f"{name}_TRAIN.tsv")
        test_path = os.path.join(base_path, name, f"{name}_TEST.tsv")

        try:
            X_train, y_train = load_from_ucr_tsv_to_dataframe(train_path)
            X_test, y_test = load_from_ucr_tsv_to_dataframe(test_path)

            # calcolo metadati
            train_size = X_train.shape[0]

            series_length = X_train.iloc[0, 0].shape[0]

            # Salva i dati caricati
            loaded_datasets[name] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "train_size": train_size,
                "series_length": series_length,
            }
            print(f" Caricato. Train Size: {train_size}, Series Length: {series_length}, Test Size: {X_test.shape[0]}")

        except Exception as e:
            print(f" Errore nel caricamento di {name}. Dettagli errore: {e}")
            continue

    return loaded_datasets

# Chiama la funzione di benchmark appropriata in base al nome dell'algoritmo.
def run_specific_benchmark(dataset_name: str, data: Dict[str, Any], seed: int, algo_name: str):
    if algo_name == WEASEL_NAME:
        return run_weasel_benchmark(dataset_name, data, seed)
    elif algo_name == ROCKET_NAME:
        return run_rocket_benchmark(dataset_name, data, seed)
    elif algo_name == RIC_RF_NAME:
        return run_ric_benchmark(dataset_name, data, seed)
    elif algo_name == BOSS_NAME:
        return run_boss_benchmark(dataset_name, data, seed)
    elif algo_name == STC_NAME:
        return run_stc_benchmark(dataset_name, data, seed)
    elif algo_name == DRCIF_NAME:
        return run_drcif_benchmark(dataset_name, data, seed)
    elif algo_name == ARSENAL_NAME:
        return run_arsenal_benchmark(dataset_name, data, seed)
    elif algo_name == HC2_NAME:
        return run_hc2_benchmark(dataset_name, data, seed)
    else:
        raise ValueError(f"Algoritmo '{algo_name}' non supportato.")


if __name__ == "__main__":
    # Caricamento di tutti i dati
    all_data = load_all_ucr_datasets(DATASET_NAMES, UCR_BASE_PATH)

    # Lista di algoritmi da eseguire
    ALGORITHMS_TO_RUN = [ROCKET_NAME]
    #ROCKET_NAME, WEASEL_NAME, RIC_RF_NAME, BOSS_NAME, STC_NAME, DRCIF_NAME, ARSENAL_NAME, HC2_NAME
    for current_algo_name in ALGORITHMS_TO_RUN:

        print("\n" + "=" * 70)
        print(f" INIZIO BENCHMARK PER CLASSIFICATORE {current_algo_name}")
        print("=" * 70)

        # Ciclo sui dataset caricati
        for dataset_name, data in all_data.items():
            print(f"\n--- Esecuzione su Dataset: {dataset_name} ---")

            # Ciclo sui seed richiesti
            dataset_runs: List[Dict[str, Any]] = []
            for seed in SEEDS:
                # Esegui l'esperimento chiamando la funzione generica
                result = run_specific_benchmark(dataset_name, data, seed, algo_name=current_algo_name)
                all_benchmark_results.append(result)
                dataset_runs.append(result)

                print(
                    f"  -> Run con seed: {seed}. Acc: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}, Tempo Tot: {result['total_time_seconds']:.2f}s")

            # Calcolo Media e Deviazione Standard per il dataset corrente
            runs_df = pd.DataFrame(dataset_runs)
            summary = runs_df[['accuracy', 'f1_score', 'total_time_seconds']].agg(['mean', 'std']).T

            print("\n--- Riepilogo Statistico (Media ± Dev. Standard) ---")
            print(f"  Metadati: Train Size={data['train_size']}, Length={data['series_length']}")

            for index, row in summary.iterrows():
                print(f"  {index.replace('_seconds', '')}: {row['mean']:.4f} ± {row['std']:.4f}")

    # stampa dati finale
    final_df = pd.DataFrame(all_benchmark_results)

    print("\n" + "#" * 70)
    print(" RIASSUNTO GENERALE (Media e Dev. Standard su tutti i dataset e algoritmi)")
    print("#" * 70)

    metadata_df = final_df[['dataset', 'train_size', 'series_length']].drop_duplicates().set_index('dataset')
    performance_summary = final_df.groupby(['dataset', 'algorithm'])[
        ['accuracy', 'f1_score', 'total_time_seconds']].agg(
        ['mean', 'std'])

    performance_summary.columns = ['_'.join(col).strip() for col in performance_summary.columns.values]
    performance_summary_reset = performance_summary.reset_index(level='algorithm')
    final_summary_combined = performance_summary_reset.join(metadata_df)

    # 5. Definisci l'ordine standardizzato delle colonne
    STANDARD_ORDER = [
        'algorithm',
        'train_size',
        'series_length',
        'accuracy_mean',
        'accuracy_std',
        'f1_score_mean',
        'f1_score_std',
        'total_time_seconds_mean',
        'total_time_seconds_std',
    ]

    #  Riordina le colonne utilizzando reindex
    final_summary_combined = final_summary_combined.reindex(columns=STANDARD_ORDER).set_index('algorithm', append=True)

    print("\n### Performance Media Dettagliata (Ordine Standardizzato) ###")
    print(final_summary_combined.to_markdown(floatfmt=".4f"))

    # Salva i risultati su un file CSV
    final_df.to_csv("benchmark_all_algorithms_raw_results.csv", index=False)
    print("\n Risultati completi salvati in 'benchmark_all_algorithms_raw_results.csv'")