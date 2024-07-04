import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import os
from sklearn.impute import KNNImputer
import warnings
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

# Convertir 'fecha' a datetime y establecerlo como índice si es necesario
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

def imputar_vecinos_cercanos(df, columnas_a_imputar):
    imputer = KNNImputer(n_neighbors=5)
    df_copy = df[columnas_a_imputar].copy()
    df_imputed = imputer.fit_transform(df_copy)
    df[columnas_a_imputar] = df_imputed
    return df

def imputar_con_interpolacion_temporal(df, columna):
    df_copia = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df_copia['fecha']):
        df_copia['fecha'] = pd.to_datetime(df_copia['fecha'])

    df_copia.set_index('fecha', inplace=True)
    df_copia[columna] = df_copia.groupby('EstacionID')[columna].apply(lambda group: group.interpolate(method='time'))
    df_copia.reset_index(inplace=True)

    return df_copia

def imputar_datos(df, tabla_distancias, columnas_modificar, umbral_nulos=0.5):
    df_copia = df.copy()

    for estacion_id in df_copia['EstacionID'].unique():
        df_estacion = df_copia[df_copia['EstacionID'] == estacion_id]

        for columna in columnas_modificar:
            nulos_por_columna = df_estacion[columna].isnull().sum()

            if nulos_por_columna > 0:
                df_estacion = imputar_promedio_estaciones_cercanas(df_estacion, tabla_distancias, columna, umbral_nulos)

                if df_estacion[columna].isnull().sum() > 0:
                    df_estacion = imputar_con_interpolacion_temporal(df_estacion, columna)

        df_estacion = imputar_vecinos_cercanos(df_estacion, columnas_modificar)
        df_copia.loc[df_copia['EstacionID'] == estacion_id, columnas_modificar] = df_estacion[columnas_modificar]

    return df_copia

columnas_climaticas = ['temperatura', 'radiacion', 'humedad_relativa', 'precipitacion', 'velocidad_viento', 'mojadura', 'direccion_viento']
df_imputado = imputar_datos(df_final, distancias, columnas_climaticas, umbral_nulos=0.5)

print(df_imputado.head())

# Guardar el DataFrame imputado en un archivo txt
ruta_relativa_guardar = parametros['01-filtro']['base_final']
ruta_base_final = os.path.join(parametros['global']['pc'], ruta_relativa_guardar)
df_imputado.to_csv(ruta_base_final, index=False)
