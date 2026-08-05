import streamlit as st
import requests
import os
import plotly.graph_objects as go

# Configuración de la página (modo ancho para que luzca moderno)
st.set_page_config(page_title="Credit Risk AI", page_icon="🏦", layout="wide")

API_URL = os.getenv("API_URL", "http://fastapi-backend:8000/predict")

# Título y encabezado
st.title("🏦 Simulador de Riesgo de Crédito Inteligente")
st.markdown("Rellena el perfil del cliente. Nuestro modelo de IA analizará los datos en tiempo real para determinar la viabilidad del préstamo.")
st.divider()

# Diseño del formulario en 3 columnas
with st.form("loan_application_form"):
    st.subheader("Datos del Solicitante")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        person_age = st.number_input("Edad", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Ingresos anuales ($)", min_value=1.0, value=50000.0, step=1000.0)
        person_emp_exp = st.number_input("Experiencia laboral (años)", min_value=0, max_value=60, value=5)
        person_education = st.selectbox("Nivel educativo", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
        
    with col2:
        loan_amnt = st.number_input("Cantidad solicitada ($)", min_value=1.0, value=10000.0, step=500.0)
        loan_int_rate = st.number_input("Tipo de interés esperado (%)", min_value=0.0, value=10.5, step=0.1)
        loan_intent = st.selectbox("Propósito del préstamo", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        person_home_ownership = st.selectbox("Situación de la vivienda", ["RENT", "MORTGAGE", "OWN", "OTHER"])

    with col3:
        credit_score = st.number_input("Puntuación de crédito (Score)", min_value=300, max_value=850, value=650)
        cb_person_cred_hist_length = st.number_input("Historial crediticio (años)", min_value=0, value=5)
        previous_loan_defaults_on_file = st.selectbox("¿Tiene impagos previos?", ["No", "Yes"])
        
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Analizar Riesgo", use_container_width=True)

# Lógica al enviar el formulario
if submitted:
    # Cálculo automático del ratio de endeudamiento
    loan_percent_income = min(loan_amnt / person_income, 1.0) if person_income > 0 else 1.0

    payload = {
        "person_age": person_age,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }
    
    with st.spinner("Analizando perfil con Red Neuronal..."):
        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                status = data["loan_status"]
                prob = data["probability"]
                
                st.divider()
                st.subheader("Resultado de la Evaluación")
                
                # Columnas para mostrar el resultado visualmente
                res_col1, res_col2 = st.columns([1, 1.5])
                
                with res_col1:
                    # Mostrar el mensaje con el tick o la cruz
                    if status == 1:
                        st.success("### ✅ PRÉSTAMO APROBADO")
                        st.markdown(f"**El modelo estima una probabilidad de aprobación del {prob*100:.1f}%.** El perfil presenta un riesgo aceptable para la entidad.")
                    else:
                        st.error("### ❌ PRÉSTAMO DENEGADO")
                        st.markdown(f"**El modelo estima una probabilidad de aprobación de solo {prob*100:.1f}%.** El perfil presenta un riesgo de impago demasiado alto.")
                
                with res_col2:
                    # Crear el gráfico tipo velocímetro con Plotly
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob,
                        title={'text': "Probabilidad de Aprobación", 'font': {'size': 24}},
                        number={'valueformat': ".1%", 'font': {'size': 40}},
                        gauge={
                            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "rgba(0,0,0,0)"}, # Ocultamos la barra por defecto
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 0.33], 'color': "#ff4b4b"},     # Rojo (Riesgo alto / Denegado)
                                {'range': [0.33, 0.66], 'color': "#ffa600"},  # Naranja (Duda)
                                {'range': [0.66, 1.0], 'color': "#00c853"}    # Verde (Aprobado)
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 6}, # Esto actúa como la aguja del velocímetro
                                'thickness': 0.75,
                                'value': prob
                            }
                        }
                    ))
                    
                    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                st.error(f"Error de validación en los datos. Código {response.status_code}: {response.text}")
                
        except Exception as e:
            st.error(f"Error de conexión: No se pudo contactar con la API del modelo. Asegúrate de que el contenedor backend está encendido. Detalle: {e}")