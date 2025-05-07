# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Carga el dataset
@st.cache
def cargar_datos():
    df = pd.read_csv('Reviews_reducido.csv')
    return df

df = cargar_datos()

# Muestra las primeras filas del dataframe en Streamlit
st.title("Análisis de Sentimientos")
st.subheader("Vista previa del dataset")
st.dataframe(df.head())

# 2. Seleccionar columnas importantes
df = df[['Text', 'Score']]

# 3. Crear columna de sentimiento
def sentimiento(score):
    if score >= 4:
        return 'Positivo'
    elif score == 3:
        return 'Neutro'
    else:
        return 'Negativo'
df['Sentimiento'] = df['Score'].apply(sentimiento)

df = df[['Text', 'Score']]
df['Sentimiento'] = df['Score'].apply(sentimiento)

# Balancear el dataset
positivas = df[df['Sentimiento'] == 'Positivo']
negativas = df[df['Sentimiento'] == 'Negativo']
minimo = min(len(positivas), len(negativas))

df = pd.concat([positivas.sample(minimo, random_state=42), negativas.sample(minimo, random_state=42)])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Mezclar


# Mostrar distribución de sentimientos
st.subheader("Distribución de Sentimientos")
fig, ax = plt.subplots()
sns.countplot(x='Sentimiento', data=df, ax=ax)
ax.set_title("Distribución de Sentimientos")
st.pyplot(fig)

# Mostrar proporción de sentimientos
st.subheader("Proporción de Sentimientos")
sentiment_proportion = df['Sentimiento'].value_counts(normalize=True)
st.write(sentiment_proportion)

# 4. Preprocesamiento y vectorización
st.subheader("Preprocesamiento y Vectorización")
X = df['Text']
y = df['Sentimiento']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizar el texto
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 5. Entrenamiento y evaluación del modelo
st.subheader("Entrenamiento y Evaluación del Modelo")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test_vectorized)
st.write("Precisión del modelo (accuracy):", accuracy_score(y_test, y_pred))
st.write("Reporte de clasificación:")
st.text(classification_report(y_test, y_pred))

# 6. Probar con texto real
st.subheader("Probar con texto real")
ejemplos = st.text_area("Ingresa una reseña para predecir el sentimiento:")
if ejemplos:
    ej_vector = vectorizer.transform([ejemplos])
    pred = model.predict(ej_vector)
    st.write(f"Sentimiento Predicho: {pred[0]}")
