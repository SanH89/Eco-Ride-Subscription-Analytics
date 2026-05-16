import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuración de la App Streamlit ---
st.set_page_config(page_title="Sistema de Alerta Temprana de Churn - Eco-Ride", layout="centered")

st.title("Sistema de Alerta Temprana de Churn - Eco-Ride")

# --- Carga de Modelos y Preprocesadores ---
# Asegúrate de que estas rutas sean correctas para tu entorno de ejecución.
# Si ejecutas localmente, considera copiar los .pkl al mismo directorio que app.py
# y usar rutas relativas como 'modelo_churn.pkl'.

# Para ejecutar en Colab o en un entorno con Google Drive montado:
# model_path = '/content/drive/MyDrive/Data Analytics IA/modelo_churn.pkl'
# preprocessor_path = '/content/drive/MyDrive/Data Analytics IA/pipeline_preproc.pkl'

# Asumiendo que los archivos .pkl están en el mismo directorio que app.py o una ruta accesible:
# Si el usuario sigue la instrucción de guardar app.py junto con los .pkl, estas rutas relativas funcionarán.
model_path = 'modelo_churn.pkl'
preprocessor_path = 'pipeline_preproc.pkl'

# Cargar el modelo
try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    st.sidebar.success("Modelos cargados exitosamente.")
except FileNotFoundError:
    st.sidebar.error(f"Error: Asegúrate de que los archivos '{model_path}' y '{preprocessor_path}' existan y estén en la ruta correcta. \n\nSi los moviste a Google Drive, monta la unidad y actualiza las rutas al directorio de Drive.")
    st.stop() # Detener la ejecución si no se encuentran los archivos
except Exception as e:
    st.sidebar.error(f"Error al cargar los modelos: {e}")
    st.stop()

# --- Interfaz de Usuario para la Entrada de Datos ---
st.header("Ingrese los datos del cliente:")

with st.form("churn_prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        edad = st.slider("Edad", min_value=18, max_value=80, value=30, step=1)
        uso_mensual_km = st.slider("Uso Mensual Km", min_value=0.0, max_value=200.0, value=75.0, step=0.1)
        gasto_promedio = st.slider("Gasto Promedio", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
    
    with col2:
        plan = st.selectbox("Plan", options=["básico", "premium", "elite"], index=0)
        soporte_tickets = st.slider("Soporte Tickets", min_value=0, max_value=10, value=1, step=1)
        region = st.selectbox("Región", options=["Centro", "Norte", "Sur"], index=0)
    
    # El botón de acción
    analizar_riesgo_button = st.form_submit_button("Analizar Riesgo")

# --- Lógica de Predicción ---
if analizar_riesgo_button:
    # Crear un DataFrame con los datos de entrada crudos
    input_data = pd.DataFrame({
        'Edad': [edad],
        'Plan': [plan],
        'Uso_Mensual_Km': [uso_mensual_km],
        'Soporte_Tickets': [soporte_tickets],
        'Gasto_Promedio': [gasto_promedio],
        'Region': [region],
        'Dias_Antiguedad': [365] # Valor placeholder, se transforma con el preprocesador
    })
    
    # En el preprocesador original, 'Dias_Antiguedad' se calculaba a partir de 'Fecha_Registro'.
    # Aquí, para una nueva inferencia, podemos pasar un valor razonable o el promedio/mediana.
    # El scaler manejará la estandarización. Si 'Dias_Antiguedad' fuera una entrada real, la pediríamos.
    # Para este ejemplo, como 'Dias_Antiguedad' se generó y no es una entrada directa del usuario,
    # la incluimos con un valor por defecto que el preprocesador escalará.
    # El preprocesador esperará todas las columnas que vio durante fit_transform en X_train.

    try:
        # Aplicar la transformación usando el preprocesador cargado
        # Asegurarse de que el orden de las columnas de input_data coincida con el esperado por el preprocesador
        # El ColumnTransformer maneja el orden, pero los nombres de las columnas deben ser los originales.

        # Para asegurar que ColumnTransformer reciba las columnas en el orden y con los nombres correctos
        # que esperaba durante el .fit(), reordenamos y renombramos si es necesario. (Aqui no hace falta rename)
        # Las columnas que el preprocessor espera son: 'Edad', 'Uso_Mensual_Km', 'Soporte_Tickets', 'Gasto_Promedio', 'Dias_Antiguedad', 'Plan', 'Region'
        
        # 'Dias_Antiguedad' es una característica ingenierizada, no una entrada de usuario directa.
        # La forma más segura de manejar esto en una app de despliegue es calcularla o usar un valor por defecto
        # que el modelo pueda interpretar (ej. la mediana de los datos de entrenamiento).
        # Para este ejemplo, ya se agregó con un valor de 365 días como placeholder.
        
        # El preprocesador espera las columnas que X_train tenía en el momento de fit_transform.
        # La lista completa de columnas originales (antes de Churn drop) era:
        # 'ID_Usuario', 'Edad', 'Fecha_Registro', 'Plan', 'Uso_Mensual_Km', 'Soporte_Tickets', 'Gasto_Promedio', 'Region'
        # Después de ingeniería de características, X tenía: 'Edad', 'Plan', 'Uso_Mensual_Km', 'Soporte_Tickets', 'Gasto_Promedio', 'Region', 'Dias_Antiguedad'

        # Asegurar el orden de las columnas para el preprocesador, si es necesario.
        # Las columnas en `input_data` deben ser las mismas y en el mismo orden que las `numerical_features`, `categorical_features_ordinal`, `categorical_features_onehot`.
        # num_features = ['Edad', 'Uso_Mensual_Km', 'Soporte_Tickets', 'Gasto_Promedio', 'Dias_Antiguedad']
        # cat_ord_features = ['Plan']
        # cat_ohe_features = ['Region']
        
        # Reordenamos input_data para que coincida con el orden esperado por el ColumnTransformer
        # Esto es crucial si el ColumnTransformer se entrenó en un orden específico.
        # Asumo que el orden de `numerical_features`, `categorical_features_ordinal`, `categorical_features_onehot` es el mismo que el orden original de las columnas en X_train.
        
        # Creamos el DataFrame de entrada con el orden de columnas que el `ColumnTransformer` espera
        # (numerical_features, categorical_features_ordinal, categorical_features_onehot)
        processed_input = preprocessor.transform(input_data)

        # Realizar la predicción
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0]

        st.subheader("Resultado del Análisis:")

        if prediction == 1:
            st.error(f"Alto Riesgo de Cancelación (Churn). Probabilidad de Churn: {prediction_proba[1]:.2%}")
        else:
            st.success(f"Cliente Estable. Probabilidad de No Churn: {prediction_proba[0]:.2%}")
            st.info(f"Probabilidad de Churn: {prediction_proba[1]:.2%}")
            
    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")
        st.warning("Por favor, verifica que los datos de entrada sean válidos y que el preprocesador y modelo se cargaron correctamente.")