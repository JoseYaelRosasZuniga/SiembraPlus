import pandas as pd
from sklearn.linear_model import LinearRegression

def predecir_junio_orizaba_lr():
    sm = pd.read_csv("DATASET_ORIZABA.csv")

    targets = [
        "LLUVMAX", "LLUVTOL", "TEMAXPROM", "EVAPMENS",
        "TEMAXEXT", "TEMINPRO", "TEMINEXT", "TEMEDMEN"
    ]

    predicciones_junio_2025 = {}

    # Índice para junio (asumiendo que el archivo está ordenado por meses)
    indice_junio = 5  # enero=0, ..., junio=5

    for palabra in targets:
        col_2024 = [col for col in sm.columns if palabra in col and "2024" in col]
        col_2025 = [col for col in sm.columns if palabra in col and "2025" in col]

        if not col_2024 or not col_2025:
            continue

        col_2024 = col_2024[0]
        col_2025 = col_2025[0]

        # X = todos los datos 2024 (para entrenar)
        X = sm[[col_2024]]
        # y = todos los datos reales 2025
        y = sm[col_2025]

        # Entrenar modelo
        model = LinearRegression()
        model.fit(X, y)

        # Tomar el valor de junio 2024 para predecir junio 2025
        valor_junio_2024 = sm.loc[indice_junio, col_2024]

        # Predecir junio 2025 usando un DataFrame con el nombre de columna correcto
        prediccion_junio_2025 = model.predict(pd.DataFrame([[valor_junio_2024]], columns=[col_2024]))

        predicciones_junio_2025[palabra] = float(prediccion_junio_2025[0])

    return predicciones_junio_2025