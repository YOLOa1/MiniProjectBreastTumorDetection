import os
import torch
import random
import pydicom
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- Helper Functions ---

def load_dcm_image(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    return img

def get_bbox_from_mask_dcm(mask_path):
    dcm = pydicom.dcmread(mask_path)
    mask = dcm.pixel_array
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return []
    x_min, y_min, x_max, y_max = xs.min(), ys.min(), xs.max(), ys.max()
    return [[x_min, y_min, x_max, y_max]]

# --- Dataset Class ---

class DicomTumorDataset(Dataset):
    def __init__(self, slice_paths, mask_paths):
        self.slice_paths = slice_paths
        self.mask_paths = mask_paths

    def __getitem__(self, idx):
        img_path = self.slice_paths[idx]
        mask_path = self.mask_paths[idx]

        image = load_dcm_image(img_path)
        image = torch.tensor(image).unsqueeze(0).repeat(3, 1, 1)  # Convert to 3-channel

        boxes = get_bbox_from_mask_dcm(mask_path)
        if not boxes:
            boxes = [[0, 0, 1, 1]]  # dummy box

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        return image, target

    def __len__(self):
        return len(self.slice_paths)

# --- Data Preparation ---

def gather_dicom_pairs(slice_root, mask_root):
    slice_paths, mask_paths = [], []
    for patient_id in os.listdir(slice_root):
        slice_folder = os.path.join(slice_root, patient_id)
        mask_folder = os.path.join(mask_root, patient_id)

        if not os.path.isdir(slice_folder) or not os.path.isdir(mask_folder):
            continue

        for fname in os.listdir(slice_folder):
            if not fname.endswith(".dcm"):
                continue
            slice_path = os.path.join(slice_folder, fname)
            mask_path = os.path.join(mask_folder, fname.replace("slice", "mask"))
            if os.path.exists(mask_path):
                slice_paths.append(slice_path)
                mask_paths.append(mask_path)

    return slice_paths, mask_paths

# Set your actual paths here
slice_dir = "/content/drive/MyDrive/PM/New folder (1)/Slices"
mask_dir = "/content/drive/MyDrive/PM/New folder (1)/Masks"

all_slices, all_masks = gather_dicom_pairs(slice_dir, mask_dir)

# Split into train/test
combined = list(zip(all_slices, all_masks))
random.seed(42)
random.shuffle(combined)

split_idx = int(0.8 * len(combined))
train_data = combined[:split_idx]
test_data = combined[split_idx:]

train_slices, train_masks = zip(*train_data)
test_slices, test_masks = zip(*test_data)

train_dataset = DicomTumorDataset(train_slices, train_masks)
test_dataset = DicomTumorDataset(test_slices, test_masks)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# --- Model Setup ---

model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# --- Training Loop ---

num_epochs = 40
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
torch.save(model.state_dict(), "/content/drive/MyDrive/PM/New folder (1)/fasterrcnn_dicom_tumor_2.pth")
print("✅ Model saved as 'fasterrcnn_dicom_tumor.pth'")
