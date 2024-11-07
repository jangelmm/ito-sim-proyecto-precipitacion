import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Cargar los datos desde un archivo CSV
df = pd.read_csv('Datos_Clima_Imputados.csv')

# Seleccionar las columnas relevantes para la predicción de la precipitación
columns = ['tempmax', 'tempmin', 'humidity', 'cloudcover', 'sealevelpressure', 'windspeed', 'solarradiation', 'precip']

# Filtrar los datos, eliminando cualquier fila con valores faltantes en las variables dependientes o independientes
df_filtered = df[columns].dropna()

# Imputación de valores faltantes (usamos la media de las columnas como estrategia)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df_filtered), columns=columns)

# Separar las variables predictoras (independientes) de la variable dependiente (precip)
X = df_imputed.drop('precip', axis=1)  # Eliminamos 'precip' de las variables independientes
y = df_imputed['precip']

# Dividir el conjunto de datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==================================================================
# 1. Regresión Polinómica
# ==================================================================
poly = PolynomialFeatures(degree=3)  # Usamos polinomios de grado 3
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Crear y entrenar el modelo de regresión lineal polinómica
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# Hacer predicciones
y_poly_pred = poly_model.predict(X_poly_test)

# Métricas de desempeño
mse_poly = mean_squared_error(y_test, y_poly_pred)
rmse_poly = np.sqrt(mse_poly)
r2_poly = poly_model.score(X_poly_test, y_test)

# Mostrar resultados
print(f'--- Regresión Polinómica ---')
print(f'Error Cuadrático Medio (MSE): {mse_poly}')
print(f'Raíz del Error Cuadrático Medio (RMSE): {rmse_poly}')
print(f'Coeficiente de determinación R²: {r2_poly}')

# ==================================================================
# 2. Random Forest
# ==================================================================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Hacer predicciones
y_rf_pred = rf_model.predict(X_test)

# Métricas de desempeño
mse_rf = mean_squared_error(y_test, y_rf_pred)
rmse_rf = np.sqrt(mse_rf)
r2_rf = rf_model.score(X_test, y_test)

# Mostrar resultados
print(f'--- Random Forest ---')
print(f'Error Cuadrático Medio (MSE): {mse_rf}')
print(f'Raíz del Error Cuadrático Medio (RMSE): {rmse_rf}')
print(f'Coeficiente de determinación R²: {r2_rf}')

# ==================================================================
# 3. XGBoost
# ==================================================================
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Hacer predicciones
y_xgb_pred = xgb_model.predict(X_test)

# Métricas de desempeño
mse_xgb = mean_squared_error(y_test, y_xgb_pred)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = xgb_model.score(X_test, y_test)

# Mostrar resultados
print(f'--- XGBoost ---')
print(f'Error Cuadrático Medio (MSE): {mse_xgb}')
print(f'Raíz del Error Cuadrático Medio (RMSE): {rmse_xgb}')
print(f'Coeficiente de determinación R²: {r2_xgb}')

# ==================================================================
# Comparar los resultados
# ==================================================================
models = ['Regresión Polinómica', 'Random Forest', 'XGBoost']
r2_values = [r2_poly, r2_rf, r2_xgb]

plt.figure(figsize=(10,6))
plt.bar(models, r2_values, color=['blue', 'green', 'red'])
plt.title('Comparación de Modelos: R²')
plt.xlabel('Modelos')
plt.ylabel('R²')
plt.show()

# Gráfico de comparación entre las predicciones y los valores reales (para el mejor modelo)
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_xgb_pred, label='Predicciones XGBoost')
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
plt.xlabel('Valor Real de Precipitación (mm)')
plt.ylabel('Predicción de Precipitación (mm)')
plt.title('Comparación entre Predicción y Realidad (XGBoost)')
plt.show()

# Histograma de los errores (diferencia entre predicción y realidad) para el mejor modelo
errors_xgb = y_xgb_pred - y_test
plt.figure(figsize=(10,6))
sns.histplot(errors_xgb, kde=True)
plt.title('Distribución de Errores de Predicción (XGBoost)')
plt.xlabel('Error (Predicción - Real)')
plt.ylabel('Frecuencia')
plt.show()
