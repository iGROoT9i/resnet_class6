# torch: Es la librería principal de PyTorch, que nos permite manejar tensores, crear modelos de redes neuronales y entrenarlas.
# torch.nn: Contiene componentes para construir redes neuronales, como capas (e.g., Linear), funciones de activación (e.g., ReLU), y funciones de pérdida (e.g., CrossEntropyLoss).
# torch.optim: Incluye algoritmos de optimización para entrenar modelos, como Adam y SGD.
# torchvision: Biblioteca que facilita el manejo de imágenes y modelos preentrenados, como ResNet.
# sklearn.metrics: Utilizamos classification_report para generar un resumen de las métricas de clasificación.
# StepLR: Es un planificador de tasa de aprendizaje que ajusta la tasa de aprendizaje en función de los epoch.
# Configuración del dispositivo

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import StepLR


# Esto selecciona si usar GPU (cuda) o CPU para entrenar el modelo. El uso de GPU acelera significativamente el entrenamiento.
# Estas líneas imprimen información sobre el entorno PyTorch, incluyendo si CUDA (la tecnología que permite usar GPU) está disponible.
# Transformaciones y Data Augmentation
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
print('-------------------------------------')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device count: {torch.cuda.device_count()}')


# Transformaciones con Data Augmentation
# Data Augmentation: Se aplican transformaciones aleatorias para aumentar la variabilidad de las imágenes, ayudando al modelo a generalizar mejor.
# RandomResizedCrop: Recorta aleatoriamente un área de la imagen y la redimensiona a 224x224.
# RandomHorizontalFlip: Invierte horizontalmente la imagen con probabilidad de 0.5.
# RandomRotation: Rota la imagen aleatoriamente hasta 30 grados.
# ColorJitter: Modifica el brillo, contraste, saturación y tono para crear más variedad.
# ToTensor: Convierte la imagen en un tensor.
# Normalize: Normaliza los valores de los píxeles en base a las medias y desviaciones estándar de ImageNet.
 
img_size = 224
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Las imágenes de validación y prueba se redimensionan sin aplicar aumentación para evaluar el rendimiento real del modelo.
# Carga de datos y DataLoader
val_test_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# ImageFolder: Carga imágenes de directorios organizados por clases (carpetas).
# DataLoader: Organiza los datos en lotes (batch_size) y permite iterar sobre ellos durante el entrenamiento.
# Modelo ResNet50 Preentrenado

batch_size = 32
train_dataset = datasets.ImageFolder('dataset/train', transform=train_transforms)
val_dataset = datasets.ImageFolder('dataset/val', transform=val_test_transforms)
test_dataset = datasets.ImageFolder('dataset/test', transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Carga el modelo ResNet50 preentrenado en ImageNet.
# Congelar y descongelar capas

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Congelar capas base, pero descongelar las últimas capas
# Se congelan las capas base para evitar actualizar sus pesos, lo cual acelera el entrenamiento y evita sobreajuste.
# Se descongelan las últimas capas (layer4) para permitir que se ajusten a nuestro problema específico.
# Modificación de la capa fully connected
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

# Modificar la capa fully connected
# Modifica la última capa para adaptarse a nuestro problema de clasificación de 6 clases.
# Se agregan capas fully connected, activaciones ReLU, y capas Dropout para prevenir sobreajuste.
# Configuración del modelo en el dispositivo
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


# Función de pérdida y optimizador
model = model.to(device)

# CrossEntropyLoss: Función de pérdida para clasificación multiclase.
# AdamW: Optimizador que ajusta los pesos del modelo durante el entrenamiento.
# StepLR: Reduce la tasa de aprendizaje cada 7 épocas.
# Entrenamiento del modelo
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)


# El modelo se entrena por 15 épocas, actualizando los pesos y ajustando la tasa de aprendizaje.
# Evaluación del modelo
epochs = 15

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")
    
    scheduler.step()



# Se evalúa el rendimiento en el conjunto de prueba para calcular la precisión.
# Informe de clasificación

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.2f}")


# Calcula métricas de clasificación como precisión, recall y f1-score para cada clase.
# Guardar el modelo
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

class_labels = test_dataset.classes
print(classification_report(y_true, y_pred, target_names=class_labels))

# Guarda los pesos del modelo entrenado para su reutilización.
model_save_path = 'improved_trained_resnet_model_15.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Modelo guardado en {model_save_path}")
