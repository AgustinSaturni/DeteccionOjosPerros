import torch
import cv2
import matplotlib.pyplot as plt
from train import KeypointNet
from dataset import ResizeGrayNormalize

def predict_keypoints(image_path, model, transform, num_keypoints=2):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    dummy_keypoints = torch.zeros((num_keypoints, 2), dtype=torch.float32)
    sample = {"image": image, "keypoints": dummy_keypoints}

    transformed = transform(sample)
    input_tensor = transformed["image"].unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_keypoints = output.squeeze(0).numpy()

    image_gray = transformed["image"].squeeze(0).numpy()
    image_bgr = cv2.cvtColor((image_gray * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)

    for (x, y) in predicted_keypoints:
        cv2.circle(image_bgr, (int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-1)

    plt.imshow(image_bgr)
    plt.title("Predicción de keypoints")
    plt.axis("off")
    plt.show()

    return predicted_keypoints

# 🛑 Solo ejecutar si este archivo es el principal
if __name__ == "__main__":
    ruta_nueva_img = r"D:\Usuarios\Usuario\Desktop\IA\DeteccionOjosPerros\perro11.jpg"
    model = KeypointNet(num_keypoints=2)
    model.load_state_dict(torch.load("keypoint_model_best.pth"))
    transform = ResizeGrayNormalize(output_size=(256, 256))

    preds = predict_keypoints(ruta_nueva_img, model, transform, num_keypoints=2)
    print("Coordenadas predichas:", preds)
