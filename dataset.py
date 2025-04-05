import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

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

# Rutas
images_path = r"D:\Usuarios\Usuario\Downloads\Perritos\images"
labels_path = r"D:\Usuarios\Usuario\Downloads\Perritos\labels.json"

# Dataset
dataset = KeypointDogDataset(images_path, labels_path)

# Visualizar la primera imagen con los puntos
sample = dataset[0]
image = sample["image"]
keypoints = sample["keypoints"]

for (x, y) in keypoints:
    cv2.circle(image, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)

plt.imshow(image)
plt.axis("off")
plt.title("Ojos del perro")
plt.show()
