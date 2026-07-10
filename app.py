import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import winsound
import plotly.graph_objects as go
import pandas as pd
import io
from predict import predict_image, predict_face, detect_faces

# Configuración de página
st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗", layout="wide")

# CSS personalizado
st.markdown("""
<style>
    .main-header { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem; }
    .status-alert { background-color: #ff4b4b; color: white; padding: 1rem; border-radius: 8px; text-align: center; font-size: 1.5rem; font-weight: bold; animation: pulse 1s infinite; }
    .status-ok { background-color: #00c853; color: white; padding: 1rem; border-radius: 8px; text-align: center; font-size: 1.5rem; font-weight: bold; }
    .alarm-active { background-color: #ff0000; color: white; padding: 1.5rem; border-radius: 10px; text-align: center; font-size: 2rem; font-weight: bold; animation: blink 0.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🚗 Driver Drowsiness Detection</h1><p>Sistema de detección de somnolencia con YOLO11</p></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuración")
mode = st.sidebar.radio("Modo de operación", ["📷 Subir imagen", "🎥 Cámara en tiempo real"])
confidence_threshold = st.sidebar.slider("Umbral de confianza", 0.5, 1.0, 0.7, 0.05)
alarm_enabled = st.sidebar.checkbox("🔊 Alarma sonora activada", value=True)
alarm_threshold = st.sidebar.slider("Frames consecutivos para alarma", 5, 30, 15, 1)

# Session State
if 'history' not in st.session_state: st.session_state.history = []
if 'drowsy_count' not in st.session_state: st.session_state.drowsy_count = 0
if 'non_drowsy_count' not in st.session_state: st.session_state.non_drowsy_count = 0
if 'total_frames' not in st.session_state: st.session_state.total_frames = 0
if 'consecutive_drowsy' not in st.session_state: st.session_state.consecutive_drowsy = 0
if 'alarm_triggered' not in st.session_state: st.session_state.alarm_triggered = False

# ==================== MODO IMAGEN ====================
if mode == "📷 Subir imagen":
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        faces = detect_faces(image_array)
        
        img_with_faces = image_array.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        with col1: st.image(img_with_faces, caption="Rostros detectados", use_container_width=True)
        
        with col2:
            st.subheader("📊 Resultados")
            if len(faces) == 0:
                st.warning("⚠️ No se detectaron rostros")
            else:
                for i, (x, y, w, h) in enumerate(faces):
                    face_crop = image_array[y:y+h, x:x+w]
                    resultado, probabilidad = predict_face(face_crop)
                    
                    if resultado == "Drowsy":
                        st.error(f"**Rostro {i+1}:** 😴 {resultado} ({probabilidad*100:.1f}%)")
                        if probabilidad >= confidence_threshold:
                            st.markdown('<div class="status-alert">⚠️ ALERTA: CONDUCTOR SOMNOLIENTO</div>', unsafe_allow_html=True)
                            if alarm_enabled: winsound.Beep(1000, 500)
                    else:
                        st.success(f"**Rostro {i+1}:** 😊 {resultado} ({probabilidad*100:.1f}%)")

# ==================== MODO CÁMARA ====================
else:
    st.subheader("🎥 Detección en Tiempo Real")
    
    col_controls = st.columns([1, 1, 1])
    with col_controls[0]: 
        run = st.checkbox("▶️ Iniciar cámara", value=False)
    with col_controls[1]:
        if st.button("🔄 Reiniciar estadísticas"):
            st.session_state.history = []
            st.session_state.drowsy_count = 0
            st.session_state.non_drowsy_count = 0
            st.session_state.total_frames = 0
            st.session_state.consecutive_drowsy = 0
            st.session_state.alarm_triggered = False
            if 'last_alarm_time' in st.session_state:
                del st.session_state.last_alarm_time
            st.rerun()
    with col_controls[2]:
        if st.button("🔕 Silenciar alarma"): 
            st.session_state.alarm_triggered = False
    
    col_metrics = st.columns(4)
    with col_metrics[0]: st.metric("📊 Frames totales", st.session_state.total_frames)
    with col_metrics[1]: st.metric("😴 Drowsy", st.session_state.drowsy_count)
    with col_metrics[2]: st.metric("😊 Non Drowsy", st.session_state.non_drowsy_count)
    with col_metrics[3]: st.metric("⚠️ Consecutivos", st.session_state.consecutive_drowsy)
    
    col_video, col_charts = st.columns([2, 3])
    
    with col_video:
        if st.session_state.alarm_triggered:
            st.markdown('<div class="alarm-active">🚨 ¡ALARMA ACTIVADA! CONDUCTOR SOMNOLIENTO</div>', unsafe_allow_html=True)
        
        frame_placeholder = st.empty()
        
        if run:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ No se pudo acceder a la cámara. Verifica permisos o si otra app la está usando.")
            else:
                st.info("⏳ Cámara activa. Procesando frames continuamente...")
                
                start_time = time.time()
                
                # BUCLE CONTINUO - Sin límite de frames
                while run and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Error al capturar frame")
                        break
                    
                    faces = detect_faces(frame)
                    current_time = time.time() - start_time
                    
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_crop = frame[y:y+h, x:x+w]
                        resultado, probabilidad = predict_face(face_crop)
                        
                        color = (0, 0, 255) if resultado == "Drowsy" else (0, 255, 0)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        label = f"{resultado} {probabilidad*100:.1f}%"
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        st.session_state.total_frames += 1
                        
                        if resultado == "Drowsy":
                            st.session_state.drowsy_count += 1
                            st.session_state.consecutive_drowsy += 1
                            
                            # Alarma continua cada 0.5 segundos
                            if st.session_state.consecutive_drowsy >= alarm_threshold:
                                if 'last_alarm_time' not in st.session_state:
                                    st.session_state.last_alarm_time = 0
                                
                                current_alarm_time = time.time()
                                time_since_last = current_alarm_time - st.session_state.last_alarm_time
                                
                                if time_since_last >= 0.5:
                                    if alarm_enabled:
                                        for _ in range(3):
                                            winsound.Beep(1500, 200)
                                            time.sleep(0.05)
                                    
                                    st.session_state.last_alarm_time = current_alarm_time
                                    st.session_state.alarm_triggered = True
                        else:
                            st.session_state.non_drowsy_count += 1
                            st.session_state.consecutive_drowsy = 0
                            st.session_state.alarm_triggered = False
                            if 'last_alarm_time' in st.session_state:
                                del st.session_state.last_alarm_time
                        
                        st.session_state.history.append({
                            'time': round(current_time, 2),
                            'result': resultado,
                            'probability': round(probabilidad, 4)
                        })
                        if len(st.session_state.history) > 200:
                            st.session_state.history = st.session_state.history[-200:]
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Control de velocidad (30 FPS)
                    time.sleep(0.033)
                
                cap.release()
                st.success("✅ Cámara detenida.")
        else:
            st.info("📹 Haz clic en '▶️ Iniciar cámara' para comenzar")
            frame_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Cámara detenida", use_container_width=True)
    
    with col_charts:
        st.subheader("📈 Análisis en Tiempo Real")
        
        fig1 = go.Figure()
        if st.session_state.history:
            timestamps = [h['time'] for h in st.session_state.history]
            probabilities = [h['probability'] for h in st.session_state.history]
            colors = ['red' if h['result'] == 'Drowsy' else 'green' for h in st.session_state.history]
            fig1.add_trace(go.Scatter(x=timestamps, y=probabilities, mode='lines+markers', line=dict(color='blue'), marker=dict(color=colors, size=8)))
            fig1.add_hline(y=confidence_threshold, line_dash="dash", line_color="red", annotation_text="Umbral")
        fig1.update_layout(title="Probabilidad de Somnolencia", xaxis_title="Tiempo (s)", yaxis_title="Probabilidad", yaxis_range=[0, 1], height=300)
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = go.Figure(data=[
            go.Bar(name='Drowsy 😴', x=['Detecciones'], y=[st.session_state.drowsy_count], marker_color='red'),
            go.Bar(name='Non Drowsy 😊', x=['Detecciones'], y=[st.session_state.non_drowsy_count], marker_color='green')
        ])
        fig2.update_layout(title="Distribución de Detecciones", barmode='group', height=300)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("📥 Exportar Datos y Gráficas")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📄 Descargar Historial (CSV)", data=csv, file_name='historial_somnolencia.csv', mime='text/csv', use_container_width=True)
    with col_exp2:
        if st.session_state.history:
            html_buffer = io.StringIO()
            fig1.write_html(html_buffer, include_plotlyjs='cdn')
            fig2.write_html(html_buffer, include_plotlyjs=False, full_html=False)
            st.download_button(label="🌐 Descargar Gráficas (HTML)", data=html_buffer.getvalue(), file_name='graficas.html', mime='text/html', use_container_width=True)