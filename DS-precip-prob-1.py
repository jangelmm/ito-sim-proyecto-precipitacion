import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Cargar los datos desde un archivo CSV
df = pd.read_csv('DS-Datos_Climaticos_Imputados.csv')

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

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones con el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio (MSE) y la raíz del error cuadrático medio (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Mostrar las métricas de desempeño
print(f'Error Cuadrático Medio (MSE): {mse}')
print(f'Raíz del Error Cuadrático Medio (RMSE): {rmse}')
print(f'Coeficiente de determinación R²: {model.score(X_test, y_test)}')

# Gráfico de comparación entre las predicciones y los valores reales
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred)
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
plt.xlabel('Valor Real de Precipitación (mm)')
plt.ylabel('Predicción de Precipitación (mm)')
plt.title('Comparación entre Predicción y Realidad')
plt.show()

# Histograma de los errores (diferencia entre predicción y realidad)
errors = y_pred - y_test
plt.figure(figsize=(10,6))
sns.histplot(errors, kde=True)
plt.title('Distribución de Errores de Predicción')
plt.xlabel('Error (Predicción - Real)')
plt.ylabel('Frecuencia')
plt.show()
