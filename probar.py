import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256, 6)
)
model.load_state_dict(torch.load('trained_resnet_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Transformaciones para la imagen de prueba
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Lista de nombres de las clases
class_names = ['estrellita', 'gerald', 'helen', 'jhon', 'julio', 'roberto'] 

# Función para predecir una imagen desde archivo
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    print(f"Predicción: {class_names[predicted.item()]}")

# Función para predecir usando la cámara de la laptop
def predict_from_camera():
    cap = cv2.VideoCapture(0)  # 0 es el ID de la cámara predeterminada

    if not cap.isOpened():
        print("No se puede acceder a la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame.")
            break

        # Convertir el frame a RGB y aplicarle las transformaciones
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        transformed_image = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(transformed_image)
            _, predicted = torch.max(output, 1)

        # Mostrar la predicción en la ventana de la cámara
        predicted_label = class_names[predicted.item()]
        cv2.putText(frame, f"Predicción: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)

        # Presiona 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Función para abrir el explorador de archivos y seleccionar una imagen
def select_image():
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de tkinter
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        predict_image(file_path)
    else:
        print("No se seleccionó ningún archivo.")

# Menú para seleccionar el modo de predicción
def main():
    print("Selecciona el modo de predicción:")
    print("1. Predecir desde una imagen")
    print("2. Predecir desde la cámara de la laptop")
    
    choice = input("Ingresa 1 o 2: ")
    
    if choice == '1':
        select_image()
    elif choice == '2':
        print("Presiona 'q' para salir de la cámara.")
        predict_from_camera()
    else:
        print("Opción no válida. Por favor, ingresa 1 o 2.")

# Ejecutar el menú principal
if __name__ == "__main__":
    main()
