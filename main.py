import argparse
from train import KeypointNet
from train import dataset, dataloader, criterion, optimizer
from dataset import ResizeGrayNormalize
from predict import predict_keypoints
import torch

def train_model(model, dataloader, num_epochs=10, save_path="keypoint_model_best.pth"):
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

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), save_path)
            print(f"✅ Modelo guardado con loss = {best_loss:.4f}")

def main(mode):
    num_keypoints = 2
    model = KeypointNet(num_keypoints)

    if mode == "train":
        train_model(model, dataloader)

    elif mode == "predict":
        model.load_state_dict(torch.load("keypoint_model_best.pth"))
        model.eval()

        # Ruta de imagen para predecir
        test_img = r"D:\Usuarios\Usuario\Desktop\IA\DeteccionOjosPerros\perro11.jpg"
        transform = ResizeGrayNormalize(output_size=(256, 256))
        preds = predict_keypoints(test_img, model, transform, num_keypoints=num_keypoints)
        print("🔍 Coordenadas predichas:", preds)

    else:
        print("❌ Modo no válido. Usá 'train' o 'predict'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento o predicción de keypoints.")
    parser.add_argument("mode", choices=["train", "predict"], help="Modo de ejecución: train o predict")
    args = parser.parse_args()

    main(args.mode)
