import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# Rutas a los modelos
MODEL_PATH = "models/best.pt"       # Tu modelo de clasificación entrenado
FACE_MODEL_PATH = "yolo11n.pt"      # Modelo ligero para detectar personas/rostros

# Cargar modelos
model = YOLO(MODEL_PATH)
face_model = YOLO(FACE_MODEL_PATH)

def detect_faces(image_array):
    """Detecta rostros usando YOLO (detecta personas y recorta la zona de la cara)"""
    results = face_model(image_array, verbose=False)
    
    faces = []
    if results and len(results) > 0:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Clase 0 = persona
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Tomar la región superior (donde está el rostro)
                height = y2 - y1
                face_height = int(height * 0.4)
                faces.append((x1, y1, x2 - x1, face_height))
    
    return faces

def predict_image(image):
    """Predice somnolencia en una imagen PIL completa"""
    image_np = np.array(image.convert("RGB"))
    results = model(image_np, verbose=False)
    
    if results and len(results) > 0:
        probs = results[0].probs
        top1_class = probs.top1
        top1_conf = probs.top1conf.item()
        class_name = results[0].names[top1_class]
        return class_name, float(top1_conf)
    
    return "Non Drowsy", 0.0

def predict_face(image_array):
    """Predice somnolencia en el rostro detectado (array OpenCV BGR)"""
    face_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    results = model(face_rgb, verbose=False)
    
    if results and len(results) > 0:
        probs = results[0].probs
        top1_class = probs.top1
        top1_conf = probs.top1conf.item()
        class_name = results[0].names[top1_class]
        return class_name, float(top1_conf)
    
    return "Non Drowsy", 0.0