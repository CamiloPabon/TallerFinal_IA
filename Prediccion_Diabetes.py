# Importar librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Cargar el dataset
df = pd.read_csv('diabetes.csv')

# Definir X (variables independientes) y y (variable dependiente)
X = df.drop('Salida', axis=1)  # Eliminar la columna 'Salida' de las variables independientes
y = df['Salida']  # La columna 'Salida' es la variable dependiente (0 o 1)

# Escalar las variables (opcional pero recomendado)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Crear el modelo de regresión logística con class_weight para manejar el desbalance
model = LogisticRegression(class_weight='balanced')  # Añadimos 'balanced' para ajustar el modelo al desbalance

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Función para predecir si una paciente tiene diabetes
def predecir_diabetes(embarazos, glucosa, presion_arterial, grosor_piel, insulina, bmi, pedigree, edad):
    nueva_paciente = pd.DataFrame({
        'Embarazos': [embarazos],
        'Glucosa': [glucosa],
        'Presion': [presion_arterial],
        'EspesorDeLaPielpiel': [grosor_piel],
        'Insulina': [insulina],
        'IndiceMasaMuscular': [bmi],
        'FuncionArbolGenealogicoDiabetes': [pedigree],
        'Anios': [edad]
    })

    # Escalar los datos
    nueva_paciente_scaled = scaler.transform(nueva_paciente)

    # Realizar la predicción
    prediccion = model.predict(nueva_paciente_scaled)

    return prediccion[0]

# Ejemplo de predicción para una nueva paciente
resultado = predecir_diabetes(embarazos=0, glucosa=120, presion_arterial=70, grosor_piel=20, insulina=10, bmi=25.5,
                              pedigree=0.5, edad=30)
print(f"Resultado de la predicción: {resultado}")

