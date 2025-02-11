import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def configuracion_modelo(model: models.resnet18, out_features: int):
        for param in model.parameters():
            param.require_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid()  # Añadir la capa sigmoide aquí
        )
        return model

# Cargar el modelo previamente entrenado
model = models.resnet18(weights=None)  # No usa pesos preentrenados
model = configuracion_modelo(model, 1)  # Asegúrate de que la arquitectura coincida con la entrenada

# Cargar los pesos
model.load_state_dict(torch.load(r"Scripts\Rama1\EjericiosPrácticos\5.TodoJunto\CNN\Pesos\PIMD.pth", map_location=device))
model.to(device)
model.eval()

# Transformaciones para la imagen (mismo Data Augmentation que en entrenamiento)
transform = A.Compose([
    A.Resize(224, 224),  # Redimensionar la imagen
    A.RandomCrop(width=150, height=200),  # Recortar
    A.Rotate(limit=40, p=0.5, border_mode=cv2.BORDER_CONSTANT),  # Rotación
    A.HorizontalFlip(p=0.6),  # Giro horizontal
    A.VerticalFlip(p=0.3),  # Giro vertical
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.3),  # Cambio de colores
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  # Normalización
    ToTensorV2(),  # Convertir a tensor
])

# Aplicar la transformación a la imagen de la cámara
def preprocess_image(frame):
    imagen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a RGB
    imagen = transform(image=imagen)["image"]  # Aplicar transformaciones
    return imagen.unsqueeze(0)  # Agregar dimensión de batch

# Etiquetas de clasificación
clases = ["No Estresado", "Estresado"]

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # No convertir a PIL, solo pasar la imagen OpenCV a preprocess_image
    imagen_tensor = preprocess_image(frame)

    # Pasar la imagen al modelo
    imagen_tensor = imagen_tensor.to(device)  
    with torch.no_grad():
        salida = model(imagen_tensor)
        print(salida)
        prediccion = round(salida.item())

    # Mostrar la etiqueta en la imagen
    cv2.putText(frame, clases[prediccion], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen en una ventana
    cv2.imshow("Estas estresaos", frame)

    # Presiona 'q' para salir
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()