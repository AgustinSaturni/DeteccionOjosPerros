import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.nn.functional as F


class ResizeGrayNormalize:
    def __init__(self, output_size):
        self.output_size = output_size  # (height, width)

    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]

        # Convertir a escala de grises
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Guardar tamaño original
        original_h, original_w = image.shape[:2]
        new_h, new_w = self.output_size

        # Redimensionar imagen
        image = cv2.resize(image, (new_w, new_h))

        # Normalizar la imagen (valores entre 0 y 1)
        image = image / 255.0

        # Redimensionar keypoints
        scale_x = new_w / original_w
        scale_y = new_h / original_h
        keypoints = keypoints.clone()
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        return {
            "image": torch.tensor(image, dtype=torch.float32).unsqueeze(0),  # (1, H, W) canal único
            "keypoints": keypoints
        }

class KeypointDogDataset(Dataset):
    def __init__(self, images_dir, labels_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        with open(labels_path, 'r') as f:
            self.labels_data = json.load(f)

        self.samples = []
        for item in self.labels_data:
            filename = item["file_upload"]
            annotations = item["annotations"][0]["result"]
            keypoints = []
            width = annotations[0]["original_width"]
            height = annotations[0]["original_height"]
            for point in annotations:
                x_pct = point["value"]["x"]
                y_pct = point["value"]["y"]
                x_px = int(x_pct * width / 100)
                y_px = int(y_pct * height / 100)
                keypoints.append((x_px, y_px))
            self.samples.append((filename, keypoints))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, keypoints = self.samples[idx]
        img_path = os.path.join(self.images_dir, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sample = {
            "image": image,
            "keypoints": torch.tensor(keypoints, dtype=torch.float32)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class KeypointNet(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # reduce to 128x128

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # reduce to 64x64

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # reduce to 32x32

        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_keypoints * 2)  # 2 coords por punto

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(x.size(0), -1, 2)  # (batch, num_keypoints, 2)


def predict_keypoints(image_path, model, transform, num_keypoints=2):
    # Cargar la imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crear sample dummy
    dummy_keypoints = torch.zeros((num_keypoints, 2), dtype=torch.float32)
    sample = {"image": image, "keypoints": dummy_keypoints}

    # Aplicar la misma transformación que en entrenamiento
    transformed = transform(sample)
    input_tensor = transformed["image"].unsqueeze(0)  # shape: (1, 1, H, W)

    # Predecir con el modelo
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_keypoints = output.squeeze(0).numpy()

    # Visualizar los keypoints sobre la imagen
    image_gray = transformed["image"].squeeze(0).numpy()  # (H, W)
    image_bgr = cv2.cvtColor((image_gray * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)

    for (x, y) in predicted_keypoints:
        cv2.circle(image_bgr, (int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-1)

    # Mostrar
    plt.imshow(image_bgr)
    plt.title("Predicción de keypoints")
    plt.axis("off")
    plt.show()

    return predicted_keypoints

# Rutas
images_path = r"D:\Usuarios\Usuario\Desktop\IA\DeteccionOjosPerros\images"
labels_path = r"D:\Usuarios\Usuario\Desktop\IA\DeteccionOjosPerros\labels.json"

# Crear transformación
transform = ResizeGrayNormalize(output_size=(256, 256))

# Pasarla al dataset
dataset = KeypointDogDataset(images_path, labels_path, transform=transform)

# Visualizar la primera imagen con los puntos
sample = dataset[9]
image = sample["image"].squeeze().numpy()  # Eliminar canal adicional
keypoints = sample["keypoints"]

# Dibujar keypoints sobre la imagen en escala de grises
image_bgr = cv2.cvtColor((image * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
for (x, y) in keypoints:
    cv2.circle(image_bgr, (int(x), int(y)), radius=4, color=(0, 255, 0), thickness=-1)

plt.imshow(image_bgr)
plt.axis("off")
plt.title("Imagen transformada en escala de grises con keypoints")
plt.show()


# Crear el DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Suponiendo que cada imagen tiene 2 puntos (modificá según tu dataset)
num_keypoints = 2
num_epochs=10
model = KeypointNet(num_keypoints=num_keypoints)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_loss = float("inf")

for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch["image"]  # (B, 1, H, W)
        keypoints = batch["keypoints"]  # (B, N, 2)

        preds = model(images)  # (B, N, 2)
        loss = criterion(preds, keypoints)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), "keypoint_model_best.pth")
        print(f"✅ Modelo guardado con loss = {best_loss:.4f}")


model = KeypointNet(num_keypoints=2)  # Usá el mismo valor que entrenaste
model.load_state_dict(torch.load("keypoint_model_best.pth"))
model.eval()  # Muy importante para desactivar dropout, batchnorm, etc.

# Ruta a la imagen nueva
ruta_nueva_img = r"D:\Usuarios\Usuario\Desktop\IA\DeteccionOjosPerros\images\perro5.jpg"

# Hacer la predicción
preds = predict_keypoints(ruta_nueva_img, model, transform, num_keypoints=2)
print("Coordenadas predichas:", preds)