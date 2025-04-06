import os
import json
import cv2
import torch
from torch.utils.data import Dataset

class ResizeGrayNormalize:
    def __init__(self, output_size):
        self.output_size = output_size  # (height, width)

    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        original_h, original_w = image.shape[:2]
        new_h, new_w = self.output_size

        image = cv2.resize(image, (new_w, new_h))
        image = image / 255.0

        scale_x = new_w / original_w
        scale_y = new_h / original_h
        keypoints = keypoints.clone()
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        return {
            "image": torch.tensor(image, dtype=torch.float32).unsqueeze(0),
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
