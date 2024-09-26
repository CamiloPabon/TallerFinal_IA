# Importar librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Cargar el dataset
df = pd.read_csv('diabetes.csv')

# Revisar el balance de clases
# print("Distribución de clases en el dataset:")
# print(df['Outcome'].value_counts())

# Definir X (variables independientes) y y (variable dependiente)
X = df.drop('Outcome', axis=1)  # Eliminar la columna 'Outcome' de las variables independientes
y = df['Outcome']  # La columna 'Outcome' es la variable dependiente (0 o 1)

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

# Evaluar el modelo con la matriz de confusión
# print("Matriz de confusión:")
# print(confusion_matrix(y_test, y_pred))

# Generar el reporte de clasificación con el parámetro zero_division
# print("Reporte de clasificación:")
# print(classification_report(y_test, y_pred, zero_division=1))


# Función para predecir si una paciente tiene diabetes
def predecir_diabetes(embarazos, glucosa, presion_arterial, grosor_piel, insulina, bmi, pedigree, edad):
    nueva_paciente = pd.DataFrame({
        'Pregnancies': [embarazos],
        'Glucose': [glucosa],
        'BloodPressure': [presion_arterial],
        'SkinThickness': [grosor_piel],
        'Insulin': [insulina],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [pedigree],
        'Age': [edad]
    })

    # Escalar los datos
    nueva_paciente_scaled = scaler.transform(nueva_paciente)

    # Realizar la predicción
    prediccion = model.predict(nueva_paciente_scaled)

    return prediccion[0]


# Ejemplo de predicción para una nueva paciente
resultado = predecir_diabetes(embarazos=3, glucosa=120, presion_arterial=70, grosor_piel=20, insulina=100, bmi=25.5,
                              pedigree=0.5, edad=30)
print(f"Resultado de la predicción: {resultado}")
