import torch
import timm
import torch.nn as nn
import torch.optim as optim

#retrains all layers. nothing is explicitly frozen
def train_loop(model, optimizer, criterion, train_loader, val_loader, device):
    num_epochs = 50

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

def load_pretrained_model(device, model_name):
    # Load pretrained ResNet18
    model = timm.create_model(model_name, pretrained=True) #'vit_tiny_patch16_224'

    # Replace the final classification layer (3 classes)
    model.fc = nn.Linear(model.fc.in_features, 3)

    # Move model to device
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    return model, optimizer, criterion
