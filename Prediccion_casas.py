# Importar librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Cargar el dataset
df = pd.read_csv('housing.csv')

# Eliminar columna 'Address'
df = df.drop('Address', axis=1)

# Definir X (variables independientes) y y (variable dependiente)
X = df.drop('Price', axis=1)
y = df['Price']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)


# Función para predecir el precio de una casa basada en sus características
def predecir_precio(avg_area_income, avg_area_house_age, avg_area_rooms, avg_area_bedrooms, area_population):
    # Crear un DataFrame con las características proporcionadas
    casa_nueva = pd.DataFrame({
        'Avg. Area Income': [avg_area_income],
        'Avg. Area House Age': [avg_area_house_age],
        'Avg. Area Number of Rooms': [avg_area_rooms],
        'Avg. Area Number of Bedrooms': [avg_area_bedrooms],
        'Area Population': [area_population]
    })

    # Hacer la predicción usando el modelo entrenado
    precio_predicho = model.predict(casa_nueva)

    return precio_predicho[0]


# Proporcionar las características de una nueva casa
nuevo_precio = predecir_precio(avg_area_income=75000, avg_area_house_age=6, avg_area_rooms=7, avg_area_bedrooms=4,
                               area_population=30000)

print(f"El precio estimado de la nueva casa es: ${nuevo_precio}")
