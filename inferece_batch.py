import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dust_detection import UNetClassifier   # import your trained model class

# ==========================================================
# 1. Setup
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load trained model
model = UNetClassifier().to(device)
model.load_state_dict(torch.load("unet_dust_classifier_final.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully.\n")

# Transform for input images
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2(),
])

# ==========================================================
# 2. Batch Prediction Function
# ==========================================================
def predict_folder(folder_path):
    print("📂 Scanning folder:", folder_path)
    valid_ext = (".jpg", ".jpeg", ".png")
    results = []

    for file in os.listdir(folder_path):
        if not file.lower().endswith(valid_ext):
            continue

        img_path = os.path.join(folder_path, file)
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            img_t = transform(image=img)["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(img_t).argmax(1).item()

            label = "Dusty" if pred == 1 else "Clean"
            results.append((file, label))
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")

    # Print results
    print("\n🧹 Classification Results:")
    for file, label in results:
        print(f"{file}: {label}")

# ==========================================================
# 3. Run Batch Inference
# ==========================================================
predict_folder("test_images")   # 👈 folder containing all images
