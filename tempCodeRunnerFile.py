import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Paso 1: Leer los datos desde un archivo CSV
data = pd.read_csv('Datos_Clima_Imputados.csv') 

# Paso 2: Seleccionar las variables independientes y la variable dependiente
# Cambia los nombres de las columnas según tu archivo
X = data[['tempmin', 'tempmax', 'cloudcover', 'windspeed']]  # Variables independientes
y = data['precip']  # Variable dependiente (precipitación en este caso)

# Paso 3: Crear el modelo de regresión lineal múltiple y ajustarlo a los datos
model = LinearRegression()
model.fit(X, y)

# Paso 4: Calcular los valores predichos
y_pred = model.predict(X)

# Paso 5: Calcular el RMSE (Error Cuadrático Medio)
rmse = sqrt(mean_squared_error(y, y_pred))
print(f"RMSE: {rmse}")

# Paso 6: Calcular el coeficiente de determinación R^2 para la precisión del modelo
r2 = r2_score(y, y_pred)
print(f"R^2: {r2}")

# Paso 7: Mostrar los coeficientes de regresión y el término constante
print("Coeficientes:", model.coef_)
print("Intercepción (constante):", model.intercept_)
