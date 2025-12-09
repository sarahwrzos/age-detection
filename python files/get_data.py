from pathlib import Path
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch

def load_data():
    # Base folder of your project (the folder where your script is)
    project_root = Path(__file__).resolve().parent

    # Local dataset folder
    extract_path = project_root.parent

    train_csv_path = extract_path / "faces" / "train.csv"
    train_images_path = extract_path / "faces" / "Train"
    extra_images_path = extract_path / "faces_02" / "part3"

    df = pd.read_csv(train_csv_path)
    print(df.head())

    print("CSV:", train_csv_path)
    print("Train images:", train_images_path)
    print("Extra images:", extra_images_path)
    return df, train_images_path

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


    
def resizeCNN():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),        # Resize to CNN input size
        transforms.ToTensor(),                # Convert PIL image to tensor
        transforms.Normalize([0.5,0.5,0.5],   # Normalize to [-1,1]
                            [0.5,0.5,0.5])
    ])
    return transform

def resizeVit():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),        # Resize to ViT input size
        transforms.ToTensor(),                # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485,0.456,0.406],   # Normalize to [-1,1]
                            std=[0.229,0.224,0.225])
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    return device, train_loader, val_loader, train_df_split, val_df_split
