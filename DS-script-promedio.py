import pandas as pd

# Paso 1: Leer los datos desde el archivo CSV
data = pd.read_csv('DS-Datos_Climaticos_Imputados.csv')

# Paso 2: Convertir la columna 'datetime' a formato de fecha
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d')

# Paso 3: Extraer el año de la columna 'datetime'
data['year'] = data['datetime'].dt.year

# Paso 4: Calcular los promedios de cada variable agrupando por año
promedios_por_anio = data.groupby('year').mean()

# Paso 5: Guardar los resultados en un nuevo archivo CSV
promedios_por_anio.to_csv('DS-Datos_Climaticos_Promedios.csv')

# Paso 6: Mostrar los primeros registros del nuevo DataFrame
print("Promedios anuales calculados:")
print(promedios_por_anio.head())
