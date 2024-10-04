# Importar librerías necesarias
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Cargar el archivo CSV generado previamente
df_logs = pd.read_csv('nginx_access_log.csv')

# Mostrar las primeras filas del DataFrame para verificar
print("Primeras filas del dataset:")
print(df_logs.head())

# Definir X (variables independientes) y y (variable dependiente)
X = df_logs[['requests_per_second', 'size', 'status']]  # Características relevantes
y = df_logs['is_dos_attack']  # Etiqueta: 1 si es ataque DOS, 0 si no

# Escalar las variables (recomendado para mejorar el rendimiento del modelo)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Crear el modelo AdaBoost con un árbol de decisión débil como estimador base
base_estimator = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Mostrar el reporte de clasificación para evaluar el rendimiento del modelo
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))


# Función para pedir datos del usuario y predecir si es ataque DOS
def predict_from_user_input():
    print("\nIngresa los siguientes datos para comprobar si la solicitud es una posible anomalía:")

    try:
        requests_per_second = 2
        size = 1000
        status = 10
    except ValueError:
        print("Entrada inválida. Asegúrate de ingresar números en los campos correspondientes.")
        return

    # Crear un DataFrame con los datos ingresados
    user_data = pd.DataFrame({
        'requests_per_second': [requests_per_second],
        'size': [size],
        'status': [status]
    })

    # Escalar los datos de entrada usando el mismo scaler que el entrenamiento
    user_data_scaled = scaler.transform(user_data)

    # Realizar la predicción
    prediction = model.predict(user_data_scaled)

    if prediction[0] == 1:
        print("La solicitud es considerada un ataque DOS.")
    else:
        print("La solicitud es normal.")


# Llamar a la función para ingresar datos y predecir
if __name__ == "__main__":
    predict_from_user_input()

