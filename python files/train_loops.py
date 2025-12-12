import torch
import timm
import torch.nn as nn
import torch.optim as optim

#retrains all layers. nothing is explicitly frozen
def train_loop(model, optimizer, criterion, train_loader, val_loader, device, epochs):
    num_epochs = epochs
    best_val_acc = 0
    patience = 5  # number of epochs to wait for improvement
    counter = 0

    print("starting training loop")
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

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += batch_size

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- VALIDATION -----
        val_acc, val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}%")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model




def validate(model, val_loader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = val_loss / len(val_loader)

    return accuracy, avg_loss

#training accuracy
def evaluate(model, data_loader, criterion, device):
    model.eval()  # set model to evaluation mode
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad(): 
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0) 

            _, preds = torch.max(outputs, 1) 
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss

def load_pretrained_cnn_model(device, model_name):
    # Load pretrained ResNet18
    model = timm.create_model(model_name, pretrained=True)

    # Replace the final classification layer with dropout
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # dropout for regularization
        nn.Linear(model.fc.in_features, 3)
    )

    for param in model.parameters():
        param.requires_grad = False  # freeze everything

    #unfreeze certain layers
    for param in model.fc.parameters():
        param.requires_grad = True

    if hasattr(model, "layer4"):
        for param in model.layer4.parameters():
            param.requires_grad = True
    else:
        print("Warning: model has no layer4 block")

    if hasattr(model, "layer3"):
        for param in model.layer3.parameters():
            param.requires_grad = True
    else:
        print("Warning: model has no layer3 block")

    # Move model to device
    model = model.to(device)


    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-5,   # slightly higher LR for fine-tuning
        weight_decay=1e-2
    )


    return model, optimizer, criterion

def load_pretrained_vit_model(device, model_name):
    # Load pretrained ViT
    model = timm.create_model(model_name, pretrained=True)

    # Replace the final classification layer (3 classes)
    if hasattr(model, "head"):
        model.head = nn.Linear(model.head.in_features, 3)
    else:
        raise RuntimeError("Model does not have a .head attribute. Cannot replace classifier.")
    
    #Freeze all layers except classifier head and last transformer block
    for name, param in model.named_parameters():
        if "head" not in name and "blocks.11" not in name:
            param.requires_grad = False

    # Move model to device
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    return model, optimizer, criterion