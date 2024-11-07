import pandas as pd

# Leer el archivo CSV
df = pd.read_csv("Datos_Clima.csv")

# Convertir la columna datetime a formato de fecha
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y')

# Establecer 'datetime' como índice
df.set_index('datetime', inplace=True)

# Realizar la imputación por interpolación para las columnas seleccionadas
variables_a_imputar = ['tempmax', 'tempmin', 'humidity', 'cloudcover', 'sealevelpressure', 'windspeed', 'solarradiation']

# Interpolar valores faltantes
df[variables_a_imputar] = df[variables_a_imputar].interpolate(method='time')

# Opcional: Resetear el índice si lo necesitas en formato original
df.reset_index(inplace=True)

# Guardar el nuevo DataFrame a un nuevo archivo CSV
df.to_csv("Datos_Clima_Imputados.csv", index=False)
