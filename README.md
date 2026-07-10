# 🚗 Driver Drowsiness Detection - YOLO11

Sistema de detección de somnolencia del conductor en tiempo real utilizando Inteligencia Artificial (YOLO11) y Streamlit.

## 📋 Descripción

Este proyecto implementa un sistema completo de monitoreo de seguridad vial que analiza el estado facial del conductor. Utiliza un modelo de clasificación YOLO11 entrenado para detectar si el conductor está alerta o somnoliento. Incluye una interfaz web interactiva, alarmas sonoras, gráficas en tiempo real y exportación de datos para su posterior análisis.

## ✨ Características Principales

- **Detección de rostros:** Localiza automáticamente el rostro del conductor usando YOLO.
- **Clasificación en tiempo real:** Analiza el estado (Drowsy / Non Drowsy) con alta precisión.
- **Modo Imagen:** Permite subir fotografías para analizar el estado del conductor.
- **Modo Cámara en Vivo:** Monitoreo continuo a través de la webcam.
- **Sistema de Alarmas:** Alertas sonoras configurables si se detecta somnolencia consecutiva.
- **Dashboard Interactivo:** Gráficas en tiempo real con Plotly (probabilidad y distribución).
- **Exportación de Datos:** Descarga del historial de detecciones en formato CSV y gráficas en HTML.

## 🛠️ Tecnologías Utilizadas

- **Python 3.11+**
- **YOLO11 (Ultralytics):** Para detección de rostros y clasificación de somnolencia.
- **Streamlit:** Para la creación de la interfaz web.
- **OpenCV:** Para el procesamiento de video y captura de cámara.
- **Plotly & Pandas:** Para el análisis y visualización de datos en tiempo real.

## 📁 Estructura del Proyecto

```text
DriverDrowsinessProject/
│
├── app.py                  # Interfaz principal de Streamlit
├── predict.py              # Lógica de predicción y detección con YOLO
├── test_predict.py         # Script de pruebas unitarias
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Documentación del proyecto
│
├── models/
│   └── best.pt             # Modelo YOLO entrenado (Descargar por separado)
│
└── data/
    └── Dataset/            # Dataset de entrenamiento (Descargar por separado)
        ├── train/
        └── val/
```

## ⚙️ Instalación y Configuración

Sigue estos pasos para ejecutar el proyecto en tu máquina local:

### 1. Clonar o descargar el repositorio
```bash
git clone https://github.com/LUIS0709-eng/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

### 2. Crear y activar un entorno virtual (Recomendado)
```bash
# En Windows
python -m venv .venv
.venv\Scripts\activate

# En Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar las dependencias
```bash
pip install -r requirements.txt
```

### 4. Descargar el Modelo Entrenado
Por su peso, el modelo no está incluido en el repositorio.
1. Descarga el archivo `best.pt` desde el siguiente enlace:
   👉 [Descargar modelo desde Google Drive](https://drive.google.com/file/d/1GpIsyYeMshf2-YH0aYuFFBkiudK3RNeo/view?usp=drive_link)
2. Coloca el archivo descargado dentro de la carpeta `models/`.

### 5. Descargar el Dataset (Opcional)
Si deseas reentrenar el modelo o probar con los datos originales, descarga el *Driver Drowsiness Dataset (DDD)* desde Kaggle:
👉 https://www.kaggle.com/datasets/ismailnasri21/driver-drowsiness-dataset-ddd
Extrae las carpetas `train` y `val` dentro de `data/Dataset/`.

## 🚀 Uso

Para iniciar la aplicación web, ejecuta el siguiente comando en la terminal:

```bash
streamlit run app.py
```

Se abrirá automáticamente una pestaña en tu navegador (usualmente en `http://localhost:8501`).

### Modos de operación:
1. **📷 Subir imagen:** Carga una foto (JPG, PNG) para analizar si el conductor está somnoliento.
2. **🎥 Cámara en tiempo real:** Inicia tu webcam. El sistema dibujará un cuadro alrededor de tu rostro y mostrará las gráficas y estadísticas en vivo.

### Configuración de la Alarma:
Desde el panel lateral (Sidebar), puedes:
- Ajustar el **Umbral de confianza** (0.5 a 1.0).
- Activar o desactivar la **Alarma sonora**.
- Configurar los **Frames consecutivos** necesarios para que salte la alarma (evita falsos positivos).

## 📊 Dataset y Entrenamiento

El modelo fue entrenado utilizando el **Driver Drowsiness Dataset (DDD)**, que contiene más de 41,000 imágenes de rostros de conductores en condiciones de alerta y somnolencia.

- **Arquitectura:** YOLO11m-cls
- **Épocas:** 79 (Early Stopping)
- **Precisión (Top-1 Accuracy):** 100% en el conjunto de validación.

## 👨‍💻 Autor

- **Nombre:** Luis Fernando Carrillo Morales
- **Universidad / Institución:** Universidad de Lima
- **Correo de contacto:** 20250607@aloe.ulima.edu.pe

## 📄 Licencia

Este proyecto fue desarrollado con fines académicos y educativos. El uso del modelo y el dataset está sujeto a sus respectivas licencias de origen.