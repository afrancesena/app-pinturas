import streamlit as st
import numpy as np
import joblib
import os

# Verificar modelo
modelo_filename = "modelo_pintura.pkl"
if not os.path.exists(modelo_filename):
    st.error("El modelo no se encontró. Sube 'modelo_pintura.pkl' al repositorio.")
    st.stop()

# Cargar modelo
modelo = joblib.load(modelo_filename)

st.title("Predicción de Propiedades de Pintura Decorativa")

# Función de predicción
def predecir(pigmento, resina, solvente, aditivos):
    datos = np.array([[pigmento, resina, solvente, aditivos]])
    return modelo.predict(datos)[0]

# Interfaz
pigmento = st.slider("Pigmento (%)", 10.0, 30.0, 20.0)
resina   = st.slider("Resina (%)",   40.0, 60.0, 50.0)
solvente = st.slider("Solvente (%)", 5.0, 15.0, 10.0)
aditivos = st.slider("Aditivos (%)", 1.0,  5.0,  3.0)

if st.button("Predecir Propiedades"):
    dureza, brillo, tiempo_secado = predecir(pigmento, resina, solvente, aditivos)
    st.write("### Resultados de la Predicción:")
    st.write(f"**Dureza:** {dureza:.2f}")
    st.write(f"**Brillo:** {brillo:.2f} GU")
    st.write(f"**Tiempo de Secado:** {tiempo_secado:.2f} minutos")
