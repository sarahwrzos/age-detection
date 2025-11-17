#imports
#data
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
#model
import torch
import timm
import torch.nn as nn
import torch.optim as optim

def load_data():
    # Base folder of your project (the folder where your script is)
    project_root = Path(__file__).resolve().parent

    # Local dataset folder
    extract_path = project_root

    train_csv_path = extract_path / "faces" / "train.csv"
    train_images_path = extract_path / "faces" / "Train"
    extra_images_path = extract_path / "faces_02" / "part3"

    df = pd.read_csv(train_csv_path)
    print(df.head())

    print("CSV:", train_csv_path)
    print("Train images:", train_images_path)
    print("Extra images:", extra_images_path)
    return df, train_images_path

def print_images(df, train_images_path):
    num_images_to_show = 5
    for i in range(num_images_to_show):
        image_id = df.loc[i, 'ID']
        image_class = df.loc[i, 'Class']

        image_path = train_images_path / image_id  # <-- Path version

        try:
            img = mpimg.imread(str(image_path))     # mpimg requires a string
            plt.imshow(img)
            plt.title(f"ID: {image_id}, Class: {image_class}")
            plt.axis('off')
            plt.show()

        except FileNotFoundError:
            print(f"Image not found: {image_path}")
        except Exception as e:
            print(f"Error displaying image {image_id}: {e}")

def map_labels(df):
    # Map textual labels to integers
    df['Class'] = df['Class'].str.strip().str.upper()

    class_mapping = {'YOUNG': 0, 'MIDDLE': 1, 'OLD': 2}
    df['Class'] = df['Class'].map(class_mapping)

    df.head()


class AgeDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['ID']   # change if your column is different
        age = self.dataframe.iloc[idx]['Class']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, age
    
def resize():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),        # Resize to ViT input size
        transforms.ToTensor(),                # Convert PIL image to tensor
        transforms.Normalize([0.5,0.5,0.5],   # Normalize to [-1,1]
                            [0.5,0.5,0.5])
    ])
    return transform



def split_data(transform, train_images_path, df):
    #split data

    # Split
    train_df_split, val_df_split = train_test_split(df, test_size=0.2, random_state=42)

    # Create Dataset instances
    train_dataset = AgeDataset(train_df_split, train_images_path, transform=transform)
    val_dataset = AgeDataset(val_df_split, train_images_path, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)#was 32
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)#was 32



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    return device, train_loader



def load_resnet18(device):
    # Load pretrained ResNet18
    model = timm.create_model('resnet18', pretrained=True)

    # Replace the final classification layer (3 classes)
    model.fc = nn.Linear(model.fc.in_features, 3)

    # Move model to device
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    return model, optimizer, criterion

def train_loop(model, optimizer, criterion, train_loader, device):
    num_epochs = 5

    print("starting loop")
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return model

def save_model(model):
    torch.save(model.state_dict(), "resnet18.pth")

def load_model():
    # Make sure to define the model architecture the same way
    model = AgeDataset()  # Replace MyCNN with your class
    model.load_state_dict(torch.load("resnet18.pth"))
    model.eval()  # Set to evaluation mode
    return model


def main():
    df, train_images_path = load_data()
    #print_images(df, train_images_path)
    map_labels(df)
    transform = resize()
    device, train_loader = split_data(transform, train_images_path, df)
    if not os.path.exists("resnet18.pth"):
        model, optimizer, critereon = load_resnet18(device)
        trained_model = train_loop(model, optimizer, critereon, train_loader, device)
        save_model(model)
    else:
        trained_model = load_model()


main()