import pandas as pd
from cd_library import draw_cd_diagram

df = pd.read_csv('csvDatiManuale.csv')
# Crea la colonna total_time_seconds
df['total_time_seconds'] = df['fit_time'] + df['predict_time']
# Rinomina le colonne per la libreria hfawaz
# La libreria si aspetta: 'classifier_name', 'dataset_name', 'accuracy'
df_perf = df.rename(columns={
    'algorithm': 'classifier_name',
    'dataset': 'dataset_name',
    'accuracy': 'accuracy'
})
# Gestione dati mancanti
# righe senza accuracy (NaN) fanno fallire la statistica. Le rimuoviamo per il grafico
df_perf = df_perf.dropna(subset=['accuracy'])

# Se hai varianti dello stesso algoritmo, rendile uniche
#df_perf['classifier_name'] = df_perf['classifier_name'] + "_" + df_perf['variant']

draw_cd_diagram(df_perf=df_perf, title='Accuracy Comparison', labels=True)