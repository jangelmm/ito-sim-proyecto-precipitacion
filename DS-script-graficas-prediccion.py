# Grafica las predicciones de la precipitacion (variable dependiente) de acuerdo a la proyeccion de las variables independientes

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Paso 1: Leer los datos históricos y renombrar la primera columna
data_hist = pd.read_csv('DS-Datos_Climaticos_Promedios.csv')
data_hist.rename(columns={data_hist.columns[0]: 'year'}, inplace=True)

# Leer los datos de proyección 2025-2030 y renombrar la primera columna
data_proyeccion = pd.read_csv('DS-Datos_Climaticos_Proyeccion_2025_2030.csv')
data_proyeccion.rename(columns={data_proyeccion.columns[0]: 'year'}, inplace=True)

# Combinar los datos históricos y de proyección en un solo DataFrame
data_total = pd.concat([data_hist, data_proyeccion], ignore_index=True)

# Paso 2: Seleccionar las variables independientes (para MLR) y la variable dependiente
X_mlr = data_total[['tempmin', 'humidity', 'windspeed', 'sealevelpressure', 'cloudcover', 'solarradiation']]
y = data_hist['precip']  # Solo los valores de 'precip' reales hasta 2024 para entrenamiento

# Crear un modelo de regresión lineal múltiple (MLR)
mlr_model = LinearRegression()
mlr_model.fit(X_mlr[:len(data_hist)], y)  # Solo con datos históricos

# Calcular los valores predichos por el modelo MLR para todo el conjunto (2010-2030)
y_pred_mlr = mlr_model.predict(X_mlr)

# Evaluar el modelo MLR en los datos históricos
rmse_mlr = sqrt(mean_squared_error(y, y_pred_mlr[:len(data_hist)]))
r2_mlr = r2_score(y, y_pred_mlr[:len(data_hist)])
print(f"MLR - RMSE: {rmse_mlr}")
print(f"MLR - R^2: {r2_mlr}")

# Crear un modelo de regresión lineal simple (SLR)
X_slr = data_total[['humidity']]  # Usamos 'humidity' como única variable predictora para el modelo SLR
slr_model = LinearRegression()
slr_model.fit(X_slr[:len(data_hist)], y)  # Entrenar el modelo solo con datos históricos

# Calcular los valores predichos por el modelo SLR para todo el conjunto (2010-2030)
y_pred_slr = slr_model.predict(X_slr)

# Evaluar el modelo SLR en los datos históricos
rmse_slr = sqrt(mean_squared_error(y, y_pred_slr[:len(data_hist)]))
r2_slr = r2_score(y, y_pred_slr[:len(data_hist)])
print(f"SLR - RMSE: {rmse_slr}")
print(f"SLR - R^2: {r2_slr}")

# Paso 3: Extraer y mostrar predicciones de MLR y SLR para 2025-2030
years_proyeccion = data_proyeccion['year']
predicciones_mlr_2025_2030 = y_pred_mlr[len(data_hist):]
predicciones_slr_2025_2030 = y_pred_slr[len(data_hist):]

print("\nPredicciones de precipitación para los años 2025-2030 (en mm):")
for year, mlr, slr in zip(years_proyeccion, predicciones_mlr_2025_2030, predicciones_slr_2025_2030):
    print(f"Año {year}: MLR = {mlr:.2f} mm, SLR = {slr:.2f} mm")

# Paso 4: Graficar los resultados desde 2010 hasta 2030
plt.figure(figsize=(12, 11))

# Graficar los valores reales de precipitación de 2010 a 2024
plt.plot(data_hist['year'], y, label='Actual Precipitation (2010-2024)', marker='o', color='blue')

# Graficar las predicciones de MLR de 2010 a 2030
plt.plot(data_total['year'], y_pred_mlr, label='MLR Prediction (2010-2030)', marker='^', color='green')

# Graficar las predicciones de SLR de 2010 a 2030
plt.plot(data_total['year'], y_pred_slr, label='SLR Prediction (2010-2030)', marker='*', color='cyan')

# Configurar etiquetas y título
plt.xlabel('Year')
plt.ylabel('Precipitation (mm)')
plt.title('Comparison of Actual vs MLR and SLR Predictions for Precipitation (2010-2030)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
