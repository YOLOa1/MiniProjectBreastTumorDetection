import os
import torch
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- Dataset Class ---

class ImageTumorDataset(Dataset):
    def __init__(self, image_paths, annotation_dict, transform=None):
        self.image_paths = image_paths
        self.annotation_dict = annotation_dict  # dict: {filename: [[x1, y1, x2, y2], ...]}
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")

        boxes = self.annotation_dict.get(filename, [])
        if not boxes:
            boxes = [[0, 0, 1, 1]]  # dummy box if no annotation

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # 1 = tumor class

        target = {
            "boxes": boxes_tensor,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.image_paths)

# --- Load Annotations from CSV ---

def load_annotations_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    annotation_dict = {}
    for _, row in df.iterrows():
        filename = row['filename']
        box = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
        if filename not in annotation_dict:
            annotation_dict[filename] = []
        annotation_dict[filename].append(box)
    return annotation_dict

# --- Paths (Update these) ---

image_dir = "Path/to/images"
csv_path = "Path/to/annotations.csv"

# --- Prepare Data ---

all_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
annotations = load_annotations_from_csv(csv_path)

# Filter only images that have annotations
filtered_images = [img for img in all_images if os.path.basename(img) in annotations]

# Split into train/test
random.seed(42)
random.shuffle(filtered_images)

split_idx = int(0.8 * len(filtered_images))
train_imgs = filtered_images[:split_idx]
test_imgs = filtered_images[split_idx:]

train_dataset = ImageTumorDataset(train_imgs, annotations, transform=F.to_tensor)
test_dataset = ImageTumorDataset(test_imgs, annotations, transform=F.to_tensor)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# --- Model Setup ---

model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # background + tumor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# --- Training Loop ---

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    lr_scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

# --- Save Model ---
torch.save(model.state_dict(), "fasterrcnn_tumor_png.pth")
print("✅ Model saved as 'fasterrcnn_tumor_png.pth'")
