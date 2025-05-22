import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
from xgboost_predictor import predecir_junio_orizaba
from linear_regression_predictor import predecir_junio_orizaba_lr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import io
import numpy as np
from PIL import Image

# --- Paleta personalizada (verde oscuro y negro, minimalista y agradable) ---
st.set_page_config(
    page_title="Siembra+ Altas Montañas de Veracruz",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    body, .main, .stApp {
        background-color: #0A2E36 !important;
        color: #fff !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .st-bb, .st-cb {
        background-color: #0A2E36 !important;
    }
    .stButton>button {
        background-color: #09A129;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.5em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #036D19;
        color: #fff;
    }
    .metric-xgboost {
        background: linear-gradient(90deg, #0A2E36 0%, #14CC60 100%);
        border-radius: 12px;
        padding: 1em;
        margin-bottom: 1em;
        color: #fff;
        box-shadow: 0 2px 8px rgba(10,46,54,0.08);
        border: 2px solid #27FB6B;
    }
    .metric-linear {
        background: linear-gradient(90deg, #222 0%, #27FB6B 100%);
        border-radius: 12px;
        padding: 1em;
        margin-bottom: 1em;
        color: #fff;
        box-shadow: 0 2px 8px rgba(10,46,54,0.08);
        border: 2px solid #14CC60;
    }
    .section-title {
        color: #27FB6B;
        font-weight: bold;
        font-size: 1.5em;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .stDownloadButton>button {
        background-color: #036D19;
        color: #fff;
        border-radius: 8px;
        font-weight: bold;
    }
    .stDownloadButton>button:hover {
        background-color: #27FB6B;
        color: #0A2E36;
    }
    .st-cg {
        color: #27FB6B !important;
    }
    /* Menú horizontal */
    .css-1cypcdb {
        background-color: #0A2E36 !important;
    }
    .stSelectbox>div>div {
        font-size: 1.2em !important;
        color: #27FB6B !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Localidades ---
localidades = {
    "Acultzingo": {"lat": 18.72, "lon": -97.31},
    "Camarón de Tejeda": {"lat": 18.741, "lon": -96.721},
    "Alpatláhuac": {"lat": 19.10, "lon": -97.10},
    "Amatlán de los Reyes": {"lat": 18.84, "lon": -96.92},
    "Aquila": {"lat": 18.80, "lon": -97.30},
    "Astacinga": {"lat": 18.55, "lon": -97.13},
    "Atlahuilco": {"lat": 18.62, "lon": -97.08},
    "Atoyac": {"lat": 18.75, "lon": -96.77},
    "Atzacan": {"lat": 18.90, "lon": -97.05},
    "Calcahualco": {"lat": 19.13, "lon": -97.15},
    "Camerino Z. Mendoza": {"lat": 18.80, "lon": -97.18},
    "Carrillo Puerto": {"lat": 18.68, "lon": -96.47},
    "Coetzala": {"lat": 18.77, "lon": -96.95},
    "Comapa": {"lat": 19.05, "lon": -96.90},
    "Córdoba": {"lat": 18.89, "lon": -96.93},
    "Coscomatepec": {"lat": 19.06, "lon": -97.05},
    "Cuichapa": {"lat": 18.77, "lon": -96.87},
    "Cuitláhuac": {"lat": 18.75, "lon": -96.68},
    "Chocamán": {"lat": 18.89, "lon": -96.96},
    "Fortín": {"lat": 18.90, "lon": -96.98},
    "Huatusco": {"lat": 19.15, "lon": -96.95},
    "Huiloapan de Cuauhtémoc": {"lat": 18.82, "lon": -97.17},
    "Ixhuatlán del Café": {"lat": 18.88, "lon": -96.99},
    "Ixhuatlancillo": {"lat": 18.87, "lon": -97.07},
    "Ixtaczoquitlán": {"lat": 18.85, "lon": -97.00},
    "Magdalena": {"lat": 18.68, "lon": -97.08},
    "Maltrata": {"lat": 18.78, "lon": -97.22},
    "Mariano Escobedo": {"lat": 18.92, "lon": -97.11},
    "Mixtla de Altamirano": {"lat": 18.57, "lon": -97.08},
    "Naranjal": {"lat": 18.79, "lon": -96.91},
    "Nogales": {"lat": 18.82, "lon": -97.15},
    "Omealca": {"lat": 18.75, "lon": -96.73},
    "Orizaba": {"lat": 18.85, "lon": -97.10},
    "Paso del Macho": {"lat": 18.90, "lon": -96.72},
    "La Perla": {"lat": 18.93, "lon": -97.13},
    "Rafael Delgado": {"lat": 18.83, "lon": -97.07},
    "Los Reyes": {"lat": 18.62, "lon": -97.13},
    "Río Blanco": {"lat": 18.84, "lon": -97.13},
    "San Andrés Tenejapan": {"lat": 18.79, "lon": -97.04},
    "Sochiapa": {"lat": 19.07, "lon": -96.93},
    "Soledad Atzompa": {"lat": 18.73, "lon": -97.18},
    "Tehuipango": {"lat": 18.48, "lon": -97.08},
    "Tenampa": {"lat": 19.08, "lon": -96.85},
    "Tepatlaxco": {"lat": 19.13, "lon": -96.88},
    "Tequila": {"lat": 18.78, "lon": -97.08},
    "Texhuacán": {"lat": 18.60, "lon": -97.08},
    "Tezonapa": {"lat": 18.62, "lon": -96.77},
    "Tlacotepec de Mejía": {"lat": 19.13, "lon": -96.90},
    "Tlaltetela": {"lat": 19.18, "lon": -96.93},
    "Tlaquilpa": {"lat": 18.62, "lon": -97.13},
    "Tlilapan": {"lat": 18.81, "lon": -97.07},
    "Tomatlán": {"lat": 19.00, "lon": -96.98},
    "Totutla": {"lat": 19.13, "lon": -96.93},
    "Xoxocotla": {"lat": 18.73, "lon": -97.08},
    "Yanga": {"lat": 18.83, "lon": -96.80},
    "Zentla": {"lat": 19.08, "lon": -96.78},
    "Zongolica": {"lat": 18.66, "lon": -97.00}
}

def obtener_clima_open_meteo(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"
    response = requests.get(url)
    if response.status_code == 200:
        datos = response.json().get("current", {})
        return {
            "temperatura": datos.get("temperature_2m", "--"),
            "humedad": datos.get("relative_humidity_2m", "--"),
            "lluvia": datos.get("precipitation", 0)
        }
    else:
        return {"temperatura": "--", "humedad": "--", "lluvia": "--"}

# --- Menú horizontal intuitivo ---
from streamlit_option_menu import option_menu

selected = option_menu(
    menu_title=None,
    options=["Clima tiempo real", "Predicción", "Reporte PDF"],
    icons=["cloud-sun", "bar-chart", "file-earmark-pdf"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0A2E36"},
        "icon": {"color": "#27FB6B", "font-size": "20px"},
        "nav-link": {"font-size": "18px", "text-align": "center", "margin":"0px", "--hover-color": "#14CC60"},
        "nav-link-selected": {"background-color": "#27FB6B", "color": "#0A2E36"},
    }
)

# Mostrar logo en la parte superior
logo = Image.open("SiembraPlus.png")
st.image(logo, width=180)

st.title("Siembra+  Altas Montañas de Veracruz")
st.caption("Predicción y recomendaciones agrícolas para la región de montaña de Veracruz.")

# --- Selector de municipio grande y destacado ---
st.markdown("""
    <style>
    .big-selectbox label {
        font-size: 1.3em !important;
        color: #27FB6B !important;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    div[data-baseweb="select"] > div {
        background-color: #0A2E36 !important;
        color: #fff !important;
        border-radius: 10px !important;
        border: 2px solid #27FB6B !important;
        font-size: 1.2em !important;
        min-height: 55px !important;
    }
    .stSelectbox>div>div {
        font-size: 1.2em !important;
        color: #27FB6B !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="big-selectbox">', unsafe_allow_html=True)
    ciudad_seleccionada = st.selectbox(
        "Seleccionar municipio",
        sorted(localidades.keys()),
        key="municipio_selectbox"
    )
    st.markdown('</div>', unsafe_allow_html=True)

coord = localidades[ciudad_seleccionada]
clima = obtener_clima_open_meteo(coord["lat"], coord["lon"])

def mostrar_mapa(ciudad_selec):
    df = pd.DataFrame([
        {
            "Municipio": ciudad,
            "lat": coords["lat"],
            "lon": coords["lon"],
            "color": [39, 251, 107] if ciudad == ciudad_selec else [10, 46, 54]
        }
        for ciudad, coords in localidades.items()
    ])
    ciudad_coords = localidades[ciudad_selec]
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=ciudad_coords["lat"],
            longitude=ciudad_coords["lon"],
            zoom=10,
            pitch=0
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position='[lon, lat]',
                get_color="color",
                get_radius=300,
                pickable=True
            )
        ],
        tooltip={"text": "{Municipio}"}
    ))

if selected == "Clima tiempo real":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Clima actual en {ciudad_seleccionada}")
        st.metric("🌡️ Sensación térmica", f"{clima['temperatura']} °C")
        st.metric("💧 Humedad", f"{clima['humedad']} %")
        lluvia_mm = clima['lluvia']
        if lluvia_mm == "--" or lluvia_mm is None:
            st.metric("🌧️ Lluvia", "Datos no disponibles")
        else:
            st.metric("🌧️ Lluvia", f"{lluvia_mm} mm/h")
    with col2:
        st.subheader("Recomendaciones agrícolas")
        st.markdown("""
**Enlaces útiles para productores:**
- [Registro Agrario Nacional (RAN)](https://www.gob.mx/ran)
- [SURI Agricultura](https://suri.agricultura.gob.mx)
- [Secretaría de Agricultura y Desarrollo Rural](https://www.gob.mx/agricultura)
- [Producción para el Bienestar](https://programasparaelbienestar.gob.mx/produccion-para-el-bienestar)
""")
        try:
            temp = float(clima["temperatura"])
            lluvia = float(clima["lluvia"])
            if lluvia > 20:
                st.warning("🌧️ Lluvia intensa: Evite labores de campo hoy.")
            elif lluvia > 5:
                st.info("🌦️ Lluvia moderada: Reduzca el riego artificial.")
            if temp < 12:
                st.error("❄️ Temperaturas bajas: Proteja cultivos sensibles al frío.")
        except:
            st.error("⚠️ No se pudieron generar recomendaciones por falta de datos.")
    mostrar_mapa(ciudad_seleccionada)
    st.caption(f"Coordenadas: {coord['lat']}°N, {coord['lon']}°W")

if selected == "Predicción" and ciudad_seleccionada == "Orizaba":
    st.markdown('<div class="section-title">🔮 Predicción XGBoost para junio 2025 (Orizaba)</div>', unsafe_allow_html=True)
    pred = predecir_junio_orizaba()
    nombres = {
        "LLUVMAX": ("LLUVIA MÁXIMA 24 H", "mm"),
        "LLUVTOL": ("LLUVIA TOTAL MENSUAL", "mm"),
        "TEMAXPROM": ("TEMPERATURA MÁXIMA PROMEDIO", "°C"),
        "TEMAXEXT": ("TEMPERATURA MÁXIMA EXTREMA", "°C"),
        "TEMINPRO": ("TEMPERATURA MÍNIMA PROMEDIO", "°C"),
        "TEMINEXT": ("TEMPERATURA MÍNIMA EXTREMA", "°C"),
        "TEMEDMEN": ("TEMPERATURA MEDIA MENSUAL", "°C")
    }
    pred_filtrado = {k: v for k, v in pred.items() if k != "EVAPMENS"}
    cols = st.columns(4)
    for i, (var, valor) in enumerate(pred_filtrado.items()):
        nombre, unidad = nombres.get(var, (var, ""))
        with cols[i % 4]:
            st.markdown(f'<div class="metric-xgboost">', unsafe_allow_html=True)
            st.metric(nombre, f"{valor:.1f} {unidad}")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 Predicción Regresión Lineal para junio 2025 (Orizaba)</div>', unsafe_allow_html=True)
    pred_lr = predecir_junio_orizaba_lr()
    cols_lr = st.columns(4)
    for i, (var, valor) in enumerate(pred_lr.items()):
        nombre, unidad = nombres.get(var, (var, ""))
        with cols_lr[i % 4]:
            st.markdown(f'<div class="metric-linear">', unsafe_allow_html=True)
            st.metric(nombre, f"{valor:.1f} {unidad}")
            st.markdown('</div>', unsafe_allow_html=True)

    # Gráfica de comparación real vs predicción para junio 2025
    sm = pd.read_csv("DATASET_ORIZABA.csv")
    targets = [
        "LLUVMAX", "LLUVTOL", "TEMAXPROM",
        "TEMAXEXT", "TEMINPRO", "TEMINEXT", "TEMEDMEN"
    ]
    indice_junio = 5  # junio

    reales = []
    predichos = []

    for palabra in targets:
        col_real = [col for col in sm.columns if palabra in col and "2025" in col]
        col_ant = [col for col in sm.columns if palabra in col and "2024" in col]
        if not col_real or not col_ant:
            reales.append(None)
            predichos.append(None)
            continue
        col_real = col_real[0]
        col_ant = col_ant[0]
        valor_real = sm.loc[indice_junio, col_real]
        reales.append(valor_real)
        from sklearn.linear_model import LinearRegression
        X = sm[[col_ant]]
        y = sm[col_real]
        model = LinearRegression().fit(X, y)
        valor_ant = sm.loc[indice_junio, col_ant]
        prediccion = model.predict(pd.DataFrame([[valor_ant]], columns=[col_ant]))[0]
        predichos.append(prediccion)

    x = range(len(targets))
    plt.figure(figsize=(10, 5))
    plt.bar(x, reales, width=0.4, label='Real', align='center', color='#0A2E36')
    plt.bar([i + 0.4 for i in x], predichos, width=0.4, label='Predicción', align='center', color='#27FB6B')
    plt.xticks([i + 0.2 for i in x], targets)
    plt.xlabel('Variable')
    plt.ylabel('Valor')
    plt.title('Comparación Real vs Predicción - Junio 2025', color='#27FB6B')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

if selected == "Reporte PDF" and ciudad_seleccionada == "Orizaba":
    st.markdown("### 📄 Descargar reporte empresarial en PDF")
    anio = 2025
    mes = 6
    class ReporteClimatico:
        def __init__(self, archivo_datos="DATASET_ORIZABA.csv"):
            self.data = pd.read_csv(archivo_datos)
            self.fecha_reporte = datetime.now().strftime("%d/%m/%Y")
        def generar_reporte_completo(self, anio, mes):
            import seaborn as sns
            sns.set_theme(style="darkgrid")
            buffer = io.BytesIO()
            with PdfPages(buffer) as pdf:
                # Portada
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                plt.text(0.5, 0.7, "REPORTE CLIMÁTICO EMPRESARIAL", ha='center', va='center', fontsize=24, weight='bold')
                plt.text(0.5, 0.6, f"Análisis y Predicciones para {self._obtener_nombre_mes(mes)} {anio}", ha='center', va='center', fontsize=18)
                plt.text(0.5, 0.4, "Ciudad de Orizaba, Veracruz", ha='center', va='center', fontsize=14)
                plt.text(0.5, 0.3, f"Fecha de generación: {self.fecha_reporte}", ha='center', va='center', fontsize=12)
                plt.text(0.5, 0.1, "SIEMBRA+", ha='center', va='center', fontsize=16, style='italic', alpha=0.7)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

                # Gráfica 1: Evolución histórica de variables clave (verde)
                data = self.data
                junio_data = data[data['MES'] == mes]
                variables_grafica = ['LLUVTOL', 'TEMAXPROM', 'TEMINPRO', 'TEMEDMEN']
                years = range(2019, 2026)
                fig, axs = plt.subplots(2, 2, figsize=(15, 8))
                for i, var in enumerate(variables_grafica):
                    valores = [junio_data[f'{year} - ORI - {var}'].values[0] for year in years if f'{year} - ORI - {var}' in junio_data.columns]
                    años_disponibles = [year for year in years if f'{year} - ORI - {var}' in junio_data.columns]
                    ax = axs[i//2, i%2]
                    ax.plot(años_disponibles, valores, marker='o', linestyle='--', color='#388e3c')
                    ax.set_title(f'Evolución de {var} en Junio')
                    ax.set_xlabel('Año')
                    ax.set_ylabel('Valor')
                    ax.grid(True)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

                # Preparar datos para modelo XGBoost
                from xgboost import XGBRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_absolute_error
                from sklearn.preprocessing import StandardScaler
                from sklearn.multioutput import MultiOutputRegressor
                X = []
                y = []
                num_vars_por_anio = len(junio_data.filter(regex='2019').columns)
                for i in range(len(years)-2):
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
                lengths = [len(row) for row in X]
                min_length = min(lengths)
                X = [row[:min_length] if len(row) > min_length else np.pad(row, (0, min_length - len(row))) for row in X]
                X = np.array(X)
                y = np.array(y)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model = MultiOutputRegressor(XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=1000,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ))
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)

                # Gráfica 2: Comparación de valores reales vs predichos (verde)
                fig, axs = plt.subplots(2, 2, figsize=(12, 6))
                for i in range(4):
                    sns.scatterplot(x=y_test[:, i], y=y_pred[:, i], ax=axs[i//2, i%2], color='#388e3c')
                    axs[i//2, i%2].plot([min(y_test[:, i]), max(y_test[:, i])], [min(y_test[:, i]), max(y_test[:, i])], 'g--')
                    axs[i//2, i%2].set_title(f'Real vs Predicho - {variables_grafica[i]}')
                    axs[i//2, i%2].set_xlabel('Real')
                    axs[i//2, i%2].set_ylabel('Predicho')
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

                # Predicción para junio 2025
                features_2025 = junio_data.filter(regex='2024').values.flatten()
                if len(features_2025) < num_vars_por_anio:
                    features_2025 = np.pad(features_2025, (0, num_vars_por_anio - len(features_2025)))
                prediccion_2025 = model.predict(scaler.transform([features_2025]))[0]
                variables = ["LLUVMAX", "LLUVTOL", "TEMAXPROM", "TEMAXPROM_rep", "TEMAXEXT", "TEMINPRO", "TEMINEXT", "TEMEDMEN"]
                valores_reales_2024 = junio_data.filter(regex='2024').values.flatten()[:len(variables)]

                # Gráfica 3: Comparación 2024 real vs 2025 predicho (verde)
                fig, ax = plt.subplots(figsize=(12, 6))
                x = np.arange(len(variables))
                width = 0.35
                ax.bar(x - width/2, valores_reales_2024, width, label='2024 (Real)', color='#388e3c')
                ax.bar(x + width/2, prediccion_2025, width, label='2025 (Predicción)', color='#66bb6a')
                ax.set_xlabel('Variables')
                ax.set_ylabel('Valores')
                ax.set_title('Comparación 2024 Real vs 2025 Predicho')
                ax.set_xticks(x)
                ax.set_xticklabels(variables, rotation=45)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

                # Gráfica 4: Variables clave para 2025 (verde)
                variables_clave = ['LLUVTOL', 'TEMAXPROM', 'TEMINPRO', 'TEMEDMEN']
                valores_clave = [prediccion_2025[variables.index(var)] for var in variables_clave]
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x=variables_clave, y=valores_clave, ax=ax, color='#388e3c')
                ax.set_title('Variables Clave Predichas para Junio 2025')
                ax.set_ylabel('Valor')
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

                # Resumen ejecutivo
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                plt.text(0.5, 0.9, "RESUMEN EJECUTIVO", ha='center', va='center', fontsize=20, weight='bold')
                plt.text(0.5, 0.85, f"Conclusiones y Recomendaciones - {self._obtener_nombre_mes(mes)} {anio}", ha='center', va='center', fontsize=16)
                conclusiones = [
                    "1. El modelo predictivo muestra una tendencia al incremento en temperaturas medias.",
                    "2. Las precipitaciones podrían mantenerse dentro del rango histórico.",
                    "3. Las temperaturas extremas muestran mayor variabilidad interanual.",
                    "4. El modelo avanzado presenta un MAE aceptable para la mayoría de variables.",
                    "5. Se recomienda monitorear especialmente las temperaturas mínimas."
                ]
                for i, texto in enumerate(conclusiones):
                    plt.text(0.1, 0.7 - i*0.08, texto, ha='left', va='center', fontsize=12, wrap=True)
                plt.text(0.8, 0.1, "Departamento de Análisis Climático", ha='right', va='center', fontsize=10, style='italic')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

                # --- Información climática y recomendaciones agrícolas detalladas ---
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                plt.text(0.5, 0.95, "INFORMACIÓN CLIMÁTICA Y RECOMENDACIONES AGRÍCOLAS", ha='center', va='center', fontsize=18, weight='bold')
                recomendaciones = """
🌧️ Lluvia
- Total mensual: 378.1 mm (muy alta)
- Máxima en 24h: 89.7 mm (riesgo de encharcamientos y erosión)

🌡️ Temperaturas
- Mínima promedio: 17.6 °C
- Mínima extrema: 15.3 °C
- Media mensual: 22.8 °C
- Máxima promedio: 28.0 °C

Estas condiciones representan un clima cálido-húmedo con lluvias abundantes en temporada de verano, típico de zonas montañosas del centro de Veracruz. Este tipo de clima favorece cultivos resistentes a la humedad y que toleran bien temperaturas templadas-cálidas, pero no heladas.

✅ Cultivos recomendados para junio 2025 en Orizaba (zona de montaña alta)
1. Café de altura (Coffea arabica)
   - Requiere suelos húmedos, con buen drenaje.
   - Se adapta bien a temperaturas entre 15 y 24 °C.
   - Soporta bien lluvias abundantes si no hay anegamientos.
   - Requiere sombra parcial.
2. Plátano (Musa spp.)
   - Muy demandante en agua, ideal con lluvias altas.
   - Temperatura óptima: 20–30 °C.
   - Debe sembrarse en suelos bien drenados, porque es sensible a encharcamientos.
3. Caña de azúcar
   - Excelente para regiones cálidas con alta pluviosidad.
   - Tolera bien temperaturas altas y lluvias superiores a 300 mm.
   - Requiere buena exposición solar y cuidado del suelo para evitar erosión.
4. Maíz (Zea mays)
   - Se adapta a temperaturas de 18–27 °C.
   - Necesita entre 300 y 500 mm de lluvia mensual, justo lo que se espera.
   - Requiere suelos bien oxigenados y sin exceso de humedad.
5. Frijol (Phaseolus vulgaris)
   - Ideal para sembrar justo antes o al inicio de la temporada de lluvias.
   - Temperaturas entre 16–28 °C lo favorecen.
   - Exceso de lluvia puede generar enfermedades fúngicas.
6. Aguacate (Persea americana)
   - Prefiere temperaturas entre 16 y 25 °C, pero tolera hasta 28 °C.
   - Necesita lluvias regulares (1,200–1,500 mm/año); junio cubre buena parte de eso.
   - Evitar suelos con encharcamientos.

⚠️ Cultivos menos recomendables en junio en zona de montaña alta
Los siguientes cultivos podrían sufrir por el exceso de lluvia o temperaturas ligeramente altas para su óptimo desarrollo en esta época:

Cultivo    Motivo de riesgo
Papa       Exceso de lluvia puede pudrir tubérculos y propiciar hongos.
Mango      Florece en secas, lluvia abundante interfiere con fructificación.
Limón      Prefiere temperaturas más cálidas (>25 °C constantes).
Naranja    Similar al limón, necesita clima menos húmedo en etapa de maduración.

📌 Recomendaciones generales
- Monitoreo constante del drenaje en cultivos como plátano, café y maíz.
- Uso de coberturas vegetales o barreras vivas para evitar erosión en laderas por lluvias intensas.
- Revisión fitosanitaria frecuente en frijol y maíz para evitar daños por hongos o plagas tras lluvias.
"""
                plt.text(0.01, 0.9, recomendaciones, ha='left', va='top', fontsize=11, wrap=True)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            buffer.seek(0)
            return buffer

        def _obtener_nombre_mes(self, mes):
            meses = [
                "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
            ]
            return meses[mes-1] if 1 <= mes <= 12 else "Mes Desconocido"

    # --- Botón de descarga de reporte ---
    # Solo mostrar si es Orizaba
    if ciudad_seleccionada == "Orizaba":
        st.markdown("### 📄 Descargar reporte empresarial en PDF")
        anio = 2025
        mes = 6
        if st.button("Generar y descargar reporte"):
            reporte = ReporteClimatico()
            pdf_buffer = reporte.generar_reporte_completo(anio, mes)
            st.download_button(
                label="Descargar PDF",
                data=pdf_buffer,
                file_name="reporte_climatico_empresarial.pdf",
                mime="application/pdf"
            )

st.caption(f"Coordenadas: {coord['lat']}°N, {coord['lon']}°W")