import torch.nn as nn

from get_data import *
from store_model import *
from train_loops import *
from print_metrics import *

MODEL_DIR = Path(__file__).resolve().parent.parent / "trained_models"
filename = "resnet18_better.pth"
#filename = "vit.pth"
# filename = "vit_10e.pth"
model_pth = MODEL_DIR / filename
model_name = "resnet18"
# model_name = 'vit_tiny_patch16_224'

def main():
    df, train_images_path = load_data()
    #print_images(df, train_images_path)
    map_labels(df)
    transform = resizeCNN()
    # transform = resizeVit()
    device, train_loader, val_loader, train_df_split, val_df_split = split_data(transform, train_images_path, df)
    if not model_pth.exists():
        model, optimizer, criterion = load_pretrained_cnn_model(device, model_name)
        # model, optimizer, criterion = load_pretrained_vit_model(device, "vit_tiny_patch16_224")
        trained_model = train_loop(model, optimizer, criterion, train_loader, val_loader, device, 25)
        save_model(trained_model, model_pth)
    else:
        #umcomment one or the other for resnet/vit
        trained_model = load_trained_resnet_model(3, model_pth, model_name)
        # trained_model = load_trained_vit_model(3, model_pth, model_name)
        print(f"{model_name} Model loaded")

        #validation and training accuracies
        criterion = nn.CrossEntropyLoss()
        val_acc, val_loss = validate(trained_model, val_loader, criterion, device)
        print(f"Loaded Model Validation Acc: {val_acc*100:.2f}%")
        train_acc, train_loss = evaluate(trained_model, train_loader, criterion, device)
        print(f"Training Accuracy: {train_acc*100:.2f}%")

        #confusion matrix
        print_confusion_matrix(trained_model, val_loader, device)


if __name__ == "__main__":
    main()