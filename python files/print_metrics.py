import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from sklearn.metrics import confusion_matrix

def print_confusion_matrix(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    confusion_matrix_df = pd.DataFrame(cm, index=['Young', 'Middle', 'Old'], columns=['Young', 'Middle', 'Old'])
    print("Confusion Matrix (True on Rows, Predicted on Columns):")
    print(confusion_matrix_df)

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
    