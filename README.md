# DeteccionOjosPerros
Investigacion sobre como usar Pytorch y Label Studio

# Como levantarlo:
1. Nos posicionamos en una ruta deseada con el CMD y hacemos: git clone https://github.com/AgustinSaturni/DeteccionOjosPerros
2. Con el comando: cd DeteccionOjosPerros ingresamos a la carpeta
3. Creamos el entorno virtual con el comando: python -m venv .venv         
4. Levantamos el entorno virtual con: source .venv/bin/activate    (entorno en Linux/Mac) o .venv\Scripts\activate (entorno en Windows)
6. Instalamos las dependencias con el comando: pip install -r requirements.txt
7. Corremos el codigo con el comando: python dataset.py


# 🐶 KeypointDogDataset - Detección de Ojos de Perros con Puntos Clave

Este script define una clase personalizada de PyTorch `Dataset` que carga imágenes de perros y sus puntos clave (keypoints), como los ojos, desde un archivo JSON y una carpeta de imágenes.

## 📦 Librerías necesarias

```python
import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
```

- `os` y `json`: Para manejo de rutas y lectura de archivos JSON.
- `cv2`: OpenCV para cargar y visualizar imágenes.
- `torch`: PyTorch para representar los datos como tensores.
- `matplotlib`: Para mostrar las imágenes con los puntos.

---

## 🧠 Clase `KeypointDogDataset`

```python
class KeypointDogDataset(Dataset):
```

Esta clase extiende `torch.utils.data.Dataset` y permite usar PyTorch para entrenar modelos con imágenes anotadas con puntos clave.

### 🔧 `__init__`

```python
def __init__(self, images_dir, labels_path, transform=None):
```

- `images_dir`: Carpeta con las imágenes.
- `labels_path`: Archivo `.json` con anotaciones.
- `transform`: Transformaciones opcionales a aplicar al sample.

#### 📥 Lectura del archivo JSON

```python
with open(labels_path, 'r') as f:
    self.labels_data = json.load(f)
```

Se abre y carga el archivo de etiquetas.

#### 🔁 Procesamiento de cada entrada

```python
for item in self.labels_data:
    ...
    for point in annotations:
        ...
```

- Cada entrada tiene:
  - `file_upload`: nombre del archivo de imagen.
  - `annotations`: contiene puntos con coordenadas en porcentaje.
- Convierte esos porcentajes en píxeles multiplicando por el ancho/alto original de la imagen.

Los puntos clave se guardan como listas de tuplas `(x, y)` en la variable `samples`.

---

### 📏 `__len__`

```python
def __len__(self):
    return len(self.samples)
```

Devuelve cuántos ejemplos tiene el dataset.

---

### 🧱 `__getitem__`

```python
def __getitem__(self, idx):
    ...
```

- Usa el índice para recuperar una imagen y sus keypoints.
- Carga la imagen con `cv2.imread`.
- Convierte la imagen a formato RGB con `cv2.cvtColor`.
- Convierte los puntos a un tensor de `float32`.

---

## 🧪 Visualización de un ejemplo

```python
# Rutas
images_path = r"D:\...\images"
labels_path = r"D:\...\labels.json"

# Dataset
dataset = KeypointDogDataset(images_path, labels_path)

# Visualizar la primera imagen
sample = dataset[0]
image = sample["image"]
keypoints = sample["keypoints"]
```

Se crea el dataset y se selecciona el primer ejemplo.

### 🎯 Dibujo de los keypoints

```python
for (x, y) in keypoints:
    cv2.circle(image, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)
```

- Dibuja un círculo rojo en cada punto clave sobre la imagen.

### 🖼️ Mostrar la imagen

```python
plt.imshow(image)
plt.axis("off")
plt.title("Ojos del perro")
plt.show()
```

Muestra la imagen anotada con los puntos clave usando `matplotlib`.

---


### 📦 `ResizeGrayNormalize`

Esta clase es una transformación personalizada que aplica los siguientes pasos a cada imagen del dataset:

1. **Convierte a escala de grises.**
2. **Redimensiona la imagen** a un tamaño fijo (`output_size`).
3. **Normaliza los píxeles** dividiendo por 255 para que estén entre 0 y 1.
4. **Redimensiona los keypoints** (puntos clave) proporcionalmente al nuevo tamaño de imagen.

```python
class ResizeGrayNormalize:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        # Procesamiento de imagen y ajuste de puntos clave
        ...
```

---


---


## 📁 Estructura esperada del dataset

```
DeteccionOjosPerros/
├── images/
│   ├── perro1.jpg
│   └── perro2.jpg
├── labels.json
└── dataset.py
```


