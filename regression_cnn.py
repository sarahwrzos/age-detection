import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision.models as models


# Dataset
class AgeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples      # list of (path, age)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, age = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor([age], dtype=torch.float32)


# Transforms + Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# ResNet Model
class AgeResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.model(x)


# Train
def train_model(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for images, ages in loader:
        images, ages = images.to(device), ages.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, ages)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Validate
def validate_model(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, ages in loader:
            images, ages = images.to(device), ages.to(device)
            preds = model(images)
            loss = loss_fn(preds, ages)
            total_loss += loss.item()
    return total_loss / len(loader)

# Compute MAE (Mean Absolute Error)
def compute_mae(model, loader, device):
    model.eval()
    total_abs_error = 0
    count = 0
    with torch.no_grad():
        for images, ages in loader:
            images, ages = images.to(device), ages.to(device)
            preds = model(images).squeeze()
            abs_err = torch.abs(preds - ages.squeeze())
            total_abs_error += abs_err.sum().item()
            count += ages.size(0)
    return total_abs_error / count

# Binned Confusion Matrix
def age_to_bin(age):
    return min(int(age) // 10, 11)

def binned_confusion_matrix(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, ages in loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            outputs = torch.clamp(outputs, 0, 120)

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(ages.squeeze().cpu().numpy())

    all_preds_binned = [age_to_bin(a) for a in all_preds]
    all_labels_binned = [age_to_bin(a) for a in all_labels]

    cm = confusion_matrix(all_labels_binned, all_preds_binned)
    n= cm.shape[0]
    df = pd.DataFrame(cm,
        index=[f"{i*10}-{i*10+9}" for i in range(n)],
        columns=[f"{i*10}-{i*10+9}" for i in range(n)],

    )
    print("Binned Confusion Matrix:")
    print(df)

# Save / Load
def save_model(model, path="resnet_regression.pth"):
    torch.save(model.state_dict(), path)

def load_model(path="resnet_regression.pth", device="cpu"):
    model = AgeResNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main():
    folder = "faces_02/part3"
    samples = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                age = int(filename.split("_")[0])
                samples.append((os.path.join(folder, filename), age))
            except:
                pass

    print("Total samples:", len(samples))

    # Train-val split
    idx_train, idx_val = train_test_split(
        np.arange(len(samples)), test_size=0.2, random_state=42
    )

    train_ds = AgeDataset([samples[i] for i in idx_train], transform=train_transform)
    val_ds   = AgeDataset([samples[i] for i in idx_val],   transform=val_transform)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    model_path = "resnet_regression.pth"

    # Load or train
    if os.path.exists(model_path):
        print("Loading saved model")
        model = load_model(model_path, device)
    else:
        print("Training new model")
        model = AgeResNet().to(device)

        loss_fn = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        early_stopper = EarlyStopping(patience=5, min_delta=0.1)

        epochs = 20
        for e in range(epochs):
            train_loss = train_model(model, train_loader, optimizer, loss_fn, device)
            val_loss = validate_model(model, val_loader, loss_fn, device)
            val_mae = compute_mae(model, val_loader, device)

            print(f"Epoch {e+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_mae:.2f}")
             
            # Early stopping check
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

        save_model(model, model_path)
        print("Model saved")

    # Final validation
    val_mae = compute_mae(model, val_loader, device)
    print(f"Final Validation MAE: {val_mae:.2f}")

    # Confusion matrix
    binned_confusion_matrix(model, val_loader, device)

    # Show sample predictions (from validation set)
    print("Sample Predictions")
    model.eval()
    for i in range(5):
        img, true_age = val_ds[i]
        pred = model(img.unsqueeze(0).to(device)).item()
        print(f"True: {true_age.item()} | Predicted: {pred:.1f}")


if __name__ == "__main__":
    main()


