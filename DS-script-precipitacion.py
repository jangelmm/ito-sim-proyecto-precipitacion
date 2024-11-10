import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Paso 1: Leer los datos desde el archivo CSV
data = pd.read_csv('DS-Datos_Climaticos_Promedios.csv')

# Paso 2: Seleccionar las variables independientes (para MLR) y la variable dependiente
# Vamos a predecir 'precip' usando otras variables como predictores
X_mlr = data[['tempmin', 'humidity', 'windspeed', 'sealevelpressure', 'cloudcover', 'solarradiation']]
y = data['precip']

# Crear un modelo de regresión lineal múltiple (MLR)
mlr_model = LinearRegression()
mlr_model.fit(X_mlr, y)

# Calcular los valores predichos por el modelo MLR
y_pred_mlr = mlr_model.predict(X_mlr)

# Evaluar el modelo MLR
rmse_mlr = sqrt(mean_squared_error(y, y_pred_mlr))
r2_mlr = r2_score(y, y_pred_mlr)
print(f"MLR - RMSE: {rmse_mlr}")
print(f"MLR - R^2: {r2_mlr}")

# Crear un modelo de regresión lineal simple (SLR)
# Usaremos 'humidity' como única variable predictora para el modelo SLR
X_slr = data[['humidity']]  # Convertimos a DataFrame para compatibilidad
slr_model = LinearRegression()
slr_model.fit(X_slr, y)

# Calcular los valores predichos por el modelo SLR
y_pred_slr = slr_model.predict(X_slr)

# Evaluar el modelo SLR
rmse_slr = sqrt(mean_squared_error(y, y_pred_slr))
r2_slr = r2_score(y, y_pred_slr)
print(f"SLR - RMSE: {rmse_slr}")
print(f"SLR - R^2: {r2_slr}")

# Paso 3: Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(data['year'], y, label='Actual Precipitation', marker='o', color='blue')
plt.plot(data['year'], y_pred_mlr, label='MLR Prediction', marker='^', color='green')
plt.plot(data['year'], y_pred_slr, label='SLR Prediction', marker='*', color='cyan')

# Configurar etiquetas y título
plt.xlabel('Year')
plt.ylabel('Precipitation (mm)')
plt.title('Comparison of Actual vs MLR and SLR Predictions for Precipitation')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
