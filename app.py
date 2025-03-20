import streamlit as st
import numpy as np
import joblib
import os

def main():
    st.title("Predicción de Propiedades de Pintura Decorativa")

    # Verificar modelo
    if not os.path.exists("modelo_pintura.pkl"):
        st.error("El modelo no se encontró. Primero entrena el modelo en Streamlit Cloud o sube el archivo!")
        return

    modelo = joblib.load("modelo_pintura.pkl")

    def predecir(pigmento, resina, solvente, aditivos):
        datos = np.array([[pigmento, resina, solvente, aditivos]])
        return modelo.predict(datos)[0]

    pigmento = st.slider("Pigmento (%)", 10.0, 30.0, 20.0)
    resina   = st.slider("Resina (%)",   40.0, 60.0, 50.0)
    solvente = st.slider("Solvente (%)", 5.0, 15.0, 10.0)
    aditivos = st.slider("Aditivos (%)", 1.0,  5.0,  3.0)

    if st.button("Predecir Propiedades"):
        dureza, brillo, secado = predecir(pigmento, resina, solvente, aditivos)
        st.write("### Resultados de la Predicción:")
        st.write(f"**Dureza:** {dureza:.2f}")
        st.write(f"**Brillo:** {brillo:.2f} GU")
        st.write(f"**Tiempo de Secado:** {secado:.2f} minutos")

if __name__ == "__main__":
    main()
