# Importar librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el dataset
df_heart = pd.read_csv('heart.csv')

# Definir X (variables independientes) y y (variable dependiente)
X = df_heart.drop('target', axis=1)  # Eliminar la columna 'target' de las variables independientes
y = df_heart['target']  # La columna 'target' es la variable dependiente (0 o 1)

# Escalar las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Crear el modelo de regresión logística con class_weight para manejar el desbalance
model = LogisticRegression(class_weight='balanced')

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo con la matriz de confusión y reporte de clasificación
# print("Matriz de confusión:")
# print(confusion_matrix(y_test, y_pred)) # Matriz de confusión

# print("\nReporte de clasificación:")
# print(classification_report(y_test, y_pred)) # Reporte de clasificación

# Función para predecir si una persona tiene enfermedad cardíaca
def predecir_enfermedad_cardiaca(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    nueva_persona = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    # Escalar los datos
    nueva_persona_scaled = scaler.transform(nueva_persona)

    # Realizar la predicción
    prediccion = model.predict(nueva_persona_scaled)

    return prediccion[0]

# Ejemplo de predicción para una nueva persona
resultado = predecir_enfermedad_cardiaca(
    age=45, sex=1, cp=3, trestbps=130, chol=250, fbs=0, restecg=1,
    thalach=150, exang=0, oldpeak=2.3, slope=2, ca=0, thal=2
)
print(f"\nResultado de la predicción: {'CHD' if resultado == 1 else 'Sano'}")
