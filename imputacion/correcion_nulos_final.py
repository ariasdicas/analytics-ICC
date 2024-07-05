import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import os
import warnings
from sklearn.impute import KNNImputer

warnings.filterwarnings("ignore")

# Ruta del directorio actual y de salida
directorio_actual = "/home/ciat/"
directorio_out = "/home/ciat/out_temporal/"

# Cargar parámetros desde el archivo YAML
ruta_yaml = os.path.join(directorio_actual, 'data.yaml')
with open(ruta_yaml, 'r') as f:
    parametros = yaml.load(f, Loader=yaml.FullLoader)

# Cargar datos de distancia y limpiar columnas
archivo_distancias = "distancias_vecinos.txt"
ruta_distancias = os.path.join(directorio_out, archivo_distancias)
distancias = pd.read_csv(ruta_distancias, sep=';')
distancias['estacion'] = distancias['estacion'].str.replace('-', '')
distancias['estaciones_cercanas'] = distancias['estaciones_cercanas'].str.replace('-', '')

# Cargar datos principales y limpiar columnas
archivo_df = 'base_fechas_completas.txt'
ruta_df = os.path.join(directorio_out, archivo_df)
df_final = pd.read_csv(ruta_df, sep=',')

# Seleccionar solo las columnas necesarias
columnas_necesarias = ['fecha', 'EstacionID', 'temperatura', 'radiacion', 'humedad_relativa', 
                       'precipitacion', 'velocidad_viento', 'mojadura', 'direccion_viento',
                       'estacion', 'Finca', 'cluster']
df_final = df_final[columnas_necesarias]

# Diccionario para mapear nombres de estaciones a IDs
estacion_dict = {
    'MAGTMG': 10, 'SAALMQ': 34, 'PANSRF': 19, 'SAAAMA': 20, 'SDTTRI': 22,
    'LUTTEH': 5, 'MAGBOU': 6, 'SAALGR': 26, 'ICCPBR': 41, 'ICCSOTZ': 45,
    'ICCPLT': 3, 'ICCYEPO': 37, 'CENCEN': 1, 'PANBAL': 17, 'SDTCBR': 21,
    'ICCCON': 33, 'PAGLOR': 14, 'PAGNRJ': 15, 'TULTLA': 25, 'MAGXOL': 11,
    'MAGPVD': 29, 'ICCCHI': 28, 'ICCALA': 30, 'PAGCOC': 31, 'MATPEO': 13,
    'MAGSNC': 9, 'TBUIRL': 23, 'TBUPUY': 24, 'LUTBON': 4, 'MAGSAV': 8,
    'ICCMTA': 44, 'ICCCDL': 27, 'ICCJOY': 46, 'ICCSTA': 36
}

# Reemplazar los valores en la columna 'EstacionID'
df_final['EstacionID'] = df_final['EstacionID'].replace(estacion_dict)

# Convertir 'fecha' a datetime
df_final['fecha'] = pd.to_datetime(df_final['fecha'])

def imputar_promedio_estaciones_cercanas(df, tabla_distancias, columna, umbral_nulos=0.5):
    df_copia = df.copy()
    nulos_totales = df_copia[columna].isnull().sum()
    total_filas = len(df_copia)
    porcentaje_nulos = nulos_totales / total_filas

    if porcentaje_nulos > umbral_nulos:
        estaciones_validas = df['EstacionID'].unique()
        filas_nulas = df_copia[df_copia[columna].isnull()]

        for indice, fila in filas_nulas.iterrows():
            estacion_id = fila['EstacionID']
            fecha_hora = fila['fecha']
            estaciones_cercanas = tabla_distancias[tabla_distancias['estacion'] == estacion_id].sort_values(by='distancia').head(2)['estaciones_cercanas']
            estaciones_escogidas = estaciones_cercanas.tolist()
            estaciones_escogidas_validas = [est for est in estaciones_escogidas if est in estaciones_validas]

            if not estaciones_escogidas_validas:
                continue

            datos_validos = df_copia[(df_copia['EstacionID'].isin(estaciones_escogidas_validas)) & (df_copia['fecha'] == fecha_hora)][columna]

            if len(datos_validos) == 0:
                promedio = np.nan
            else:
                promedio = np.nanmean(datos_validos)

            df_copia.at[indice, columna] = promedio

    return df_copia

def knn_impute(df, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

def preprocess_and_impute(df, distancias, climatic_columns, knn_imputer_k, umbral_nulos=0.5):
    result_df = df.copy()

    # Imputar usando el promedio de estaciones cercanas cuando los nulos superen el umbral
    for columna in climatic_columns:
        result_df = imputar_promedio_estaciones_cercanas(result_df, distancias, columna, umbral_nulos)

    # Imputar usando KNN para los valores que aún falten
    result_df[climatic_columns] = knn_impute(result_df[climatic_columns], n_neighbors=knn_imputer_k)

    return result_df

# Configuración de parámetros
knn_imputer_k = 5

# Convertir columnas a tipo numérico si es necesario
columnas_numericas = ['temperatura', 'radiacion', 'humedad_relativa', 'precipitacion', 
                      'velocidad_viento', 'mojadura', 'direccion_viento']
for columna in columnas_numericas:
    df_final[columna] = pd.to_numeric(df_final[columna], errors='coerce')

# Realizar la imputación
df_imputado = preprocess_and_impute(df_final, distancias, columnas_numericas, knn_imputer_k, umbral_nulos=0.5)

# Guardar el DataFrame imputado en un archivo txt
ruta_relativa_guardar = parametros['01-filtro']['base_final']
ruta_base_final = os.path.join(parametros['global']['pc'], ruta_relativa_guardar)
df_imputado.to_csv(ruta_base_final, index=False)