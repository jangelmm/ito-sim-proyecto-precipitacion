import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from math import sqrt
import matplotlib.pyplot as plt

# Paso 1: Leer los datos desde un archivo CSV
data = pd.read_csv('DC-Datos_Clima_Imputados.csv')

# Paso 2: Convertir la columna 'datetime' a formato de fecha (formato YYYY-MM-DD)
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d', errors='coerce')

# Verificar si hay errores de conversión en la columna 'datetime'
if data['datetime'].isnull().any():
    print("Advertencia: Se encontraron fechas inválidas y se convirtieron a NaT (Not a Time).")
    data = data.dropna(subset=['datetime'])  # Eliminar filas con fechas inválidas

# **Paso 3: Guardar la columna 'datetime' antes de filtrar**
datetime_col = data['datetime']

# Paso 4: Filtrar las columnas no numéricas y manejar valores faltantes
data = data.select_dtypes(include=['float64', 'int64'])  # Mantener solo columnas numéricas
data = data.dropna(subset=['precip'])  # Eliminar filas donde 'precip' es NaN

# Paso 5: Seleccionar las variables independientes y la variable dependiente
X = data.drop(columns=['precip'], errors='ignore')  # Variables independientes
y = data['precip']  # Variable dependiente (precipitación)

# Paso 6: Imputar valores faltantes en las variables independientes (si los hay)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Paso 7: Crear el modelo de regresión lineal múltiple y ajustarlo a los datos
model = LinearRegression()
model.fit(X, y)

# Paso 8: Calcular los valores predichos
y_pred = model.predict(X)

# Paso 9: Calcular el RMSE (Error Cuadrático Medio)
rmse = sqrt(mean_squared_error(y, y_pred))
print(f"RMSE: {rmse}")

# Paso 10: Calcular el coeficiente de determinación R^2 para la precisión del modelo
r2 = r2_score(y, y_pred)
print(f"R^2: {r2}")

# Paso 11: Mostrar los coeficientes de regresión y el término constante
print("Coeficientes:", model.coef_)
print("Intercepción (constante):", model.intercept_)

# Paso 12: Generar una gráfica para visualizar los valores reales y predichos de precipitación

# Agregar la columna de predicciones al DataFrame original
data['precip_pred'] = y_pred

# Restaurar la columna 'datetime'
data['datetime'] = datetime_col

# Crear la gráfica
plt.figure(figsize=(12, 6))
plt.plot(data['datetime'], data['precip'], label='Precipitación Real', marker='o', linestyle='-', color='blue')
plt.plot(data['datetime'], data['precip_pred'], label='Precipitación Predicha', linestyle='--', color='red')
plt.xlabel('Fecha')
plt.ylabel('Precipitación (mm)')
plt.title('Comparación de Precipitación Real vs Predicha')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
