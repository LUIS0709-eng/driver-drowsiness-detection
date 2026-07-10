import os
import random
from PIL import Image
from predict import predict_image

# Asegúrate de que esta ruta exista
ruta = "data/Dataset/val/Drowsy"
archivo = random.choice(os.listdir(ruta))
ruta_imagen = os.path.join(ruta, archivo)
imagen = Image.open(ruta_imagen)

resultado, probabilidad = predict_image(imagen)
print("Imagen:", archivo)
print("Resultado:", resultado)
print("Probabilidad:", probabilidad)