# 📊 Benchmarking di algoritmi allo Stato dell'Arte per Time Series Classification (TSC)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Sktime](https://img.shields.io/badge/library-sktime-orange?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Target-blue?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Time Series](https://img.shields.io/badge/Focus-Time%20Series-green?style=for-the-badge)
![UCR Archive](https://img.shields.io/badge/Data-UCR%20Archive-red?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

🎓 **Tesi di Laurea Triennale**  
Università degli Studi di Torino  
Dipartimento di Informatica  

**Candidato:** Mattia Liberatore  
**Ambito:** Machine Learning / Time Series Analysis  

---

## 📝 Abstract

La classificazione di serie temporali (TSC) è un task critico in settori quali sanità, finanza e cybersecurity. Questo lavoro di tesi presenta un benchmarking comparativo tra i modelli allo stato dell'arte, valutandoli su 12 dataset eterogenei dell'archivio UCR.

L'analisi valuta l'efficacia predittiva (*accuracy*) e l'efficienza computazionale (*fit/predict time*). Dai risultati emerge come:

- **ROCKET** si confermi l'algoritmo più efficiente (tempi nell'ordine dei `10^2` secondi)
- **HIVE-COTE 2.0** rappresenti il gold standard per l'accuratezza, a fronte di:
  - costi computazionali superiori di tre ordini di grandezza (`> 10^5` secondi)
  - elevati requisiti di memoria RAM

---

## 🚀 Algoritmi Testati

Il framework integra ed estende i classificatori della libreria `sktime`, raggruppati per tipologia:

### 📚 Dictionary-based
- BOSS  
- WEASEL  

### ⚡ Convolution-based
- ROCKET  
- ARSENAL  

### 📏 Interval-based
- RandomIntervalClassifier (RIC)  
- DrCIF  

### 🧠 Deep Learning
- InceptionTime  
- ResNet  

### 🔗 Meta-Ensemble
- HIVE-COTE 2.0  

### 🔍 Shapelet-based
- ShapeletTransformClassifier (STC)  

---

## 📂 Struttura del Software

Il codice è progettato in modo modulare per gestire la complessità computazionale e la specificità di ogni algoritmo.

### 1. 🧩 Modulo Principale
`benchmark_all_algorithms_parallel.py`

Gestisce il workflow principale:

- Caricamento dataset UCR  
- Z-Normalizzazione (instance-wise)  
- Loop di esecuzione su algoritmi e seed  
- Gestione logiche specifiche:
  - skip dataset critici (es. HC2)
  - gestione dinamica dei `n_jobs`  
- Salvataggio incrementale dei risultati in CSV (fault-tolerant)

---

### 2. ⚙️ Moduli Algoritmi
`NomeAlgoritmo_benchmark.py`

Ogni algoritmo ha un modulo dedicato con gestione di inizializzazione e iperparametri.

**Esempio: logica adattiva per BOSS**

```python
# Esempio di logica adattiva per la gestione delle finestre in BOSS
if requested_min_win >= effective_max_win - 2:
    calc_min_window = max(6, int(effective_max_win * 0.25))
else:
    calc_min_window = requested_min_win

classifier = BOSSEnsemble(min_window=calc_min_window, ...)
```

---

### 3. 📊 Analisi Dati

**Cartella:** `/AnalisiDati`

Contiene il notebook:

- `analisiRisultatiBenchmark.ipynb`

Utilizzato per:

- analisi statistica  
- generazione di grafici  
- confronto delle performance  
- creazione di tabelle per il paper finale  

---

## 📊 Metodologia Sperimentale

- **Dataset:** 12 dataset dall'archivio UCR (formato TSV)  
- **Pre-processing:** Z-Normalizzazione per ogni serie (invarianza di scala)  

### Metriche

- Accuracy  
- F1-Score (Macro)  
- Fit Time  
- Predict Time  

---

## ⚠️ Hardware Note

Alcuni algoritmi (es. *HIVE-COTE 2.0*) sono stati:

- limitati  
- oppure esclusi su dataset specifici (es. *ElectricDevices*, *Crop*)  

a causa di:

- elevato consumo di RAM  
- tempi di esecuzione oltre i limiti accettabili  

---

## 💡 Conclusioni

Lo studio evidenzia che **non esiste un classificatore universalmente superiore**:

### HIVE-COTE 2.0
- massima accuratezza  
- ma costi computazionali estremi  

### ROCKET
- miglior compromesso  
- ideale per scenari *time-sensitive*  

### InceptionTime
- supera ResNet  
- efficace nel catturare pattern temporali complessi  

---

## 🎓 Documentazione

Il lavoro completo è disponibile in formato PDF:

📄 **Visualizza la Tesi**  
📄 **[Elborato di tesi Mattia Liberatore TSC (PDF)](./Tesi_Mattia_Liberatore_Benchmarking_TSC_Methods.pdf)**

---

## 🛠️ Come utilizzare il progetto

Questo progetto è stato sviluppato per scopi accademici.  

### Esecuzione

1. Inserisci i dataset UCR nella cartella configurata  
2. Esegui:

```bash
python benchmark_all_algorithms_parallel.py
```

---

## 📌 Note Finali

Il framework è progettato per essere:

- modulare  
- estendibile  
- adattabile a nuovi algoritmi e dataset  

Perfetto come base per ulteriori ricerche in ambito **Time Series Classification**.
