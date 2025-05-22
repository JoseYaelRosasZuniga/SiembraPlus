import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

def predecir_junio_orizaba():
    # Cargar los datos
    data = pd.read_csv('DATASET_ORIZABA.csv')

    # Filtrar solo los datos de junio (mes 6)
    junio_data = data[data['MES'] == 6]

    # Preparar los datos
    years = range(2019, 2026)  # 2019-2025
    X = []
    y = []

    # Determinar el número máximo de variables por año
    num_vars_por_anio = len(junio_data.filter(regex='2019').columns)

    # Crear características (años anteriores) y targets (año siguiente)
    for i in range(len(years)-2):  # Hasta 2024 porque 2025 será el último target
        features = []
        for year in years[:i+1]:
            year_data = junio_data.filter(regex=str(year)).values.flatten()
            if len(year_data) < num_vars_por_anio:
                year_data = np.pad(year_data, (0, num_vars_por_anio - len(year_data)))
            features.extend(year_data)
        target = junio_data.filter(regex=str(years[i+1])).values.flatten()
        if len(target) < num_vars_por_anio:
            target = np.pad(target, (0, num_vars_por_anio - len(target)))
        X.append(features)
        y.append(target)

    # Igualar longitudes
    lengths = [len(row) for row in X]
    min_length = min(lengths)
    X = [row[:min_length] if len(row) > min_length else np.pad(row, (0, min_length - len(row))) for row in X]
    X = np.array(X)
    y = np.array(y)

    # Entrenamiento y escalado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo XGBoost multi-output
    model = MultiOutputRegressor(XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
    model.fit(X_train_scaled, y_train)

    # Predicción para junio 2025 usando datos de 2024
    features_2025 = []
    for year in range(2019, 2025):
        year_data = junio_data.filter(regex=str(year)).values.flatten()
        if len(year_data) < num_vars_por_anio:
            year_data = np.pad(year_data, (0, num_vars_por_anio - len(year_data)))
        features_2025 = year_data  # Solo el último año

    prediccion_2025 = model.predict(scaler.transform([features_2025]))[0]

    # Nombres de las variables (ajusta según tu dataset real)
    variables = [
        "LLUVMAX", "LLUVTOL",
        "TEMAXPROM", "EVAPMENS",
        "TEMAXEXT",
        "TEMINPRO",
        "TEMINEXT",
        "TEMEDMEN"
    ]

    # Diccionario resultado
    resultado = {var: float(prediccion_2025[i]) for i, var in enumerate(variables)}
    return resultado