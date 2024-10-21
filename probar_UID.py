import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button
import numpy as np
import torch.nn as nn
from torchvision.models import ResNet50_Weights

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo
# model = models.resnet50(pretrained=False)
# num_features = model.fc.in_features
# model.fc = torch.nn.Sequential(
#     torch.nn.Linear(num_features, 256),
#     torch.nn.ReLU(),
#     torch.nn.Dropout(0.5),
#     torch.nn.Linear(256, 6)
# )
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 6),
)



model.load_state_dict(torch.load('improved_trained_resnet_model_15.pth', map_location=device))
model = model.to(device)
model.eval()

# Transformaciones para la imagen de prueba
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Lista de nombres de las clases
class_names = ['estrellita', 'gerald', 'helen', 'jhon', 'julio', 'roberto']  # Ajusta los nombres

# Función para predecir una imagen desde archivo
def predict_image(image_path, label):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    label.config(text=f"Predicción: {class_names[predicted.item()]}")
    image_display = Image.open(image_path).resize((300, 300))
    photo = ImageTk.PhotoImage(image_display)
    image_label.config(image=photo)
    image_label.image = photo

# Función para predecir usando la cámara de la laptop
def predict_from_camera(label):
    cap = cv2.VideoCapture(0)  # 0 es el ID de la cámara predeterminada

    if not cap.isOpened():
        label.config(text="No se puede acceder a la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            label.config(text="No se pudo capturar el frame.")
            break

        # Convertir el frame a RGB y aplicarle las transformaciones
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        transformed_image = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(transformed_image)
            _, predicted = torch.max(output, 1)

        # Mostrar la predicción en tiempo real
        predicted_label = class_names[predicted.item()]
        label.config(text=f"Predicción: {predicted_label}")

        cv2.putText(frame, f"Predicción: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", frame)

        # Presiona 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Función para abrir el explorador de archivos y seleccionar una imagen
def select_image(label):
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        predict_image(file_path, label)
    else:
        label.config(text="No se seleccionó ningún archivo.")

# Función para crear la interfaz gráfica
def create_gui():
    window = tk.Tk()
    window.title("Reconocimiento de Rostros")

    # Etiqueta para mostrar la predicción
    prediction_label = Label(window, text="Selecciona un modo para empezar", font=("Helvetica", 14))
    prediction_label.pack(pady=10)

    # Botones para seleccionar el modo
    Button(window, text="Cargar Imagen", command=lambda: select_image(prediction_label)).pack(pady=5)
    Button(window, text="Usar Cámara", command=lambda: predict_from_camera(prediction_label)).pack(pady=5)

    # Etiqueta para mostrar la imagen seleccionada
    global image_label
    image_label = Label(window)
    image_label.pack(pady=10)

    # Correr la interfaz
    window.mainloop()

# Ejecutar la interfaz gráfica
if __name__ == "__main__":
    create_gui()
