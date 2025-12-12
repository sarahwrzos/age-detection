from pathlib import Path
import torch
import timm
import torch.nn as nn

def save_model(model, name):
    MODEL_DIR = Path(__file__).resolve().parent.parent / "trained_models"
    MODEL_DIR.mkdir(exist_ok=True)
    save_path = MODEL_DIR / name

    torch.save(model.state_dict(), save_path)
    print(f"Saved model to: {save_path}")

def load_trained_resnet_model(num_classes, model_path, model_name, device="cpu"):
    # Create model (pretrained=False, because we load saved weights)
    model = timm.create_model(model_name, pretrained=False)

    # Replace classifier with same structure as during training
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    # Load saved weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move to device and set evaluation mode
    model.to(device)
    model.eval()

    return model


def load_trained_vit_model(num_classes, model_path, model_name, device="cpu"):
    # Recreate model architecture
    model = timm.create_model(model_name, pretrained=False)

    # Replace ViT classifier head
    if hasattr(model, "head"):
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise RuntimeError("Model does not have a .head attribute. Cannot load classifier.")

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model