import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import KeypointDogDataset, ResizeGrayNormalize

class KeypointNet(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_keypoints * 2)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(x.size(0), -1, 2)

# Configuración general
images_path = r"D:\Usuarios\Usuario\Desktop\IA\DeteccionOjosPerros\Train_Images"
labels_path = r"D:\Usuarios\Usuario\Desktop\IA\DeteccionOjosPerros\labels.json"
transform = ResizeGrayNormalize(output_size=(256, 256))

# Dataset y modelo (solo definición, sin entrenamiento)
dataset = KeypointDogDataset(images_path, labels_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
num_keypoints = 2
model = KeypointNet(num_keypoints=num_keypoints)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Función de entrenamiento
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    best_loss = float("inf")
    for epoch in range(num_epochs):
        for batch in dataloader:
            images = batch["image"]
            keypoints = batch["keypoints"]

            preds = model(images)
            loss = criterion(preds, keypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "keypoint_model_best.pth")
            print(f"✅ Modelo guardado con loss = {best_loss:.4f}")

# Ejecutar entrenamiento solo si este archivo es el principal
if __name__ == "__main__":
    train_model(model, dataloader, criterion, optimizer, num_epochs=10)
