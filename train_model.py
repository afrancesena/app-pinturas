import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def main():
    np.random.seed(42)
    num_samples = 200
    df = pd.DataFrame({
        "Pigmento (%)": np.random.uniform(10, 30, num_samples),
        "Resina (%)": np.random.uniform(40, 60, num_samples),
        "Solvente (%)": np.random.uniform(5, 15, num_samples),
        "Aditivos (%)": np.random.uniform(1, 5, num_samples),
        "Dureza": np.random.uniform(2, 10, num_samples),
        "Brillo": np.random.uniform(20, 80, num_samples),
        "Tiempo de secado": np.random.uniform(30, 120, num_samples)
    })

    X = df[["Pigmento (%)", "Resina (%)", "Solvente (%)", "Aditivos (%)"]]
    y = df[["Dureza", "Brillo", "Tiempo de secado"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    joblib.dump(modelo, "modelo_pintura.pkl")
    print("Modelo entrenado y guardado como 'modelo_pintura.pkl'")

if __name__ == "__main__":
    main()
