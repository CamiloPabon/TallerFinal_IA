# 1. Construir un modelo para predecir los precios de las casas

Dataset: housing.csv

### Caracteristicas:

- **Avg. Area Income**: Ingreso promedio de los residentes de la ciudad donde se encuentra la casa.
- **Avg. Area House Age**: Edad promedio de las casas en la misma ciudad
- **Avg. Area Number of Rooms**: Número promedio de habitaciones para casas en la misma ciudad
- **Avg. Area Number of Bedrooms:** edio de dormitorios en el área': Número promedio de dormitorios para casas en la misma ciudad
- **Area Population**: Población de la ciudad donde se encuentra la casa
- **Price**': Precio al que se vendió la casa
- **Address**': Dirección de la casa

## Variable objetivo:

El campo que interesa predecir es: **Price**

# 2. Predicción Diabetes

Este conjunto de datos proviene originalmente del Instituto Nacional de Diabetes y Enfermedades Digestivas y Renales. El objetivo es predecir, con base en medidas diagnósticas, si una paciente tiene diabetes. Se impusieron varias restricciones a la selección de estas instancias de una base de datos más grande. En particular, todas las pacientes aquí son mujeres.:

Dataset: diabetes.csv

## Características:

- **Embarazos**: Número de veces embarazadas
- **Glucosa**: concentración de glucosa en plasma a las 2 horas en una prueba de tolerancia oral a la glucosa
- **Presión arterial:** presión arterial diastólica (mm Hg)
- **Grosor de la piel:** Grosor del pliegue cutáneo del tríceps (mm)
- **Insulina:** insulina sérica de 2 horas (mu U/ml)
- **IMC:** Índice de masa corporal (peso en kg/(altura en m)^2)
- **DiabetesPedigreeFunction:** función de pedigrí de diabetes
- **Edad**: Edad (años)

## Variable objetivo:

- Resultado: variable de clase (0 o 1)

# 3. Predicción de enfermedades del corazón

Es un subconjunto de variables de un estudio en diferentes regiones del planeta para predecir el riesgo de sufrir una enfermedad relacionada con el corazón. El siguiente conjunto de datos es una muestra retrospectiva de personas en una región de alto riesgo de enfermedad cardíaca. Muchas de las personas con enfermedad coronaria positiva se han sometido a un tratamiento de reducción de la presión arterial y otros programas para reducir sus factores de riesgo después de su evento de enfermedad coronaria. En algunos casos, las mediciones se realizaron después de estos tratamientos. La base de datos está compuesta por 462 vectores de 9 características. Entre ellos, 302 corresponden a pacientes sanos mientras que 160 padecen enfermedad coronaria.

Dataset: heart.csv

## Características:

Las nueve características se definen de la siguiente manera:

- presión arterial sistólica sbp
- tabaco tabaco acumulativo (kg)
- LDL colesterol unido a lipoproteínas de baja densidad
- adiposidad
- antecedentes familiares familiares de enfermedad cardíaca
- comportamiento de tipo A
- obesidad
- alcohol consumo actual de alcohol
- edad edad de inicio

## Variable objetivo:

La clase asociada a cada vector:

- clase chd : CHD (1) o paciente sano (0