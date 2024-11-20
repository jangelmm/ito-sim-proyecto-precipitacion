import pandas as pd
from pmdarima import auto_arima
import numpy as np

# Leer los datos desde el archivo CSV
data = pd.read_csv('DS-Datos_Climaticos_Promedios.csv', index_col='year')
variables = data.columns.tolist()
variables.remove('precip')  # Excluir la columna 'precip' para la proyección

# Asegurarse de que las columnas sean numéricas
for variable in variables:
    data[variable] = pd.to_numeric(data[variable], errors='coerce')

# Crear un DataFrame vacío para almacenar las proyecciones
proyeccion = data.copy()

# Proyectar cada variable utilizando ARIMA
for variable in variables:
    # Asegurarse de trabajar con datos no nulos
    serie = data[variable].dropna()

    # Verificar si la serie tiene datos suficientes
    if len(serie) > 0:
        # Selección automática de parámetros ARIMA
        model = auto_arima(serie, seasonal=True, m=1, suppress_warnings=True)
        model_fit = model.fit(serie)
        
        # Hacer la predicción para 2025-2030
        forecast = model_fit.predict(n_periods=6)
        
        # Agregar la proyección con ruido a los valores
        forecast_with_noise = forecast + np.random.normal(0, forecast.std(), size=forecast.shape)
        
        # Agregar la proyección al DataFrame 'proyeccion'
        for i, value in enumerate(forecast_with_noise, start=2025):
            proyeccion.at[i, variable] = value
    else:
        pass

# Guardar los resultados completos en un nuevo archivo CSV
proyeccion.to_csv('DS-Datos_Climaticos_Proyectados.csv')

# Guardar solo las proyecciones de 2025 a 2030, excluyendo la columna "precip"
proyeccion_2025_2030 = proyeccion.loc[2025:2030].drop(columns=['precip'])
proyeccion_2025_2030.to_csv('DS-Datos_Climaticos_Proyeccion_2025_2030.csv')

# Mostrar los últimos registros del DataFrame proyectado para verificar
print("\nDatos completados hasta 2030:")
print(proyeccion.tail(10))
