# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('Datos_Clima_Imputados.csv')

# Seleccionar las columnas relevantes
# Variables predictoras (independientes)
X = df[['tempmax', 'tempmin', 'humidity', 'cloudcover', 'sealevelpressure', 'windspeed', 'solarradiation']]
# Variable objetivo (dependiente)
y = df['precip']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Error cuadrático medio (MSE):", mse)
print("Coeficiente de determinación (R^2):", r2)

# Mostrar los coeficientes del modelo
print("Coeficientes del modelo:", model.coef_)
print("Intercepto del modelo:", model.intercept_)

# Generar la gráfica de valores reales vs predichos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="b", label="Predicciones")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Línea perfecta")
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Valores Reales vs Predichos de la Precipitación")
plt.legend()
plt.grid(True)
plt.show()

# Generar una gráfica de los residuos
residuos = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuos, bins=30, color="c", edgecolor="black", alpha=0.7)
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.title("Distribución de los Residuos")
plt.grid(True)
plt.show()

# Generar gráfica de barras para los coeficientes del modelo
plt.figure(figsize=(10, 6))
features = X.columns
coef_values = model.coef_
plt.bar(features, coef_values, color="skyblue", edgecolor="black")
plt.xlabel("Variables predictoras")
plt.ylabel("Coeficiente")
plt.title("Coeficientes de la Regresión Lineal")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
