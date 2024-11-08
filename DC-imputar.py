import pandas as pd
from sklearn.impute import SimpleImputer

# Leer el archivo CSV
df = pd.read_csv("DC-Datos-Climaticos-Originales.csv")

# Convertir la columna 'datetime' a formato de fecha para ordenar cronológicamente
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y')

# Establecer 'datetime' como índice para usar la interpolación basada en el tiempo
df.set_index('datetime', inplace=True)

# Listas de variables según el tipo de imputación
variables_numericas = [
    'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 
    'humidity', 'precip', 'precipprob', 'precipcover', 'snow', 'snowdepth', 'windgust', 
    'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation', 
    'solarenergy', 'uvindex', 'severerisk'
]

# Imputación para variables numéricas mediante interpolación temporal
df[variables_numericas] = df[variables_numericas].interpolate(method='time')

# Para las columnas categóricas, utilizamos la imputación del valor más frecuente
variables_categoricas = ['preciptype', 'conditions', 'description', 'icon', 'stations', 'name', 'sunrise', 'sunset', 'moonphase']

# Crear un imputador de SimpleImputer con la estrategia de moda (valor más frecuente)
imputer_categ = SimpleImputer(strategy='most_frequent')
df[variables_categoricas] = imputer_categ.fit_transform(df[variables_categoricas])

# Opcional: resetear el índice si prefieres tener 'datetime' como columna regular
df.reset_index(inplace=True)

# Guardar el DataFrame imputado en un nuevo archivo CSV
df.to_csv("DC-Datos_Clima_Imputados.csv", index=False)

print("Imputación completa y archivo guardado como 'Datos_Clima_Imputados.csv'")
