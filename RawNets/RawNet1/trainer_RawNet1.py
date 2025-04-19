import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_rawnet1_with_loaders(model, train_loader, val_loader=None, device="cuda", epochs=40, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)
            elif inputs.dim() == 3 and inputs.shape[1] != 1:
                inputs = inputs.permute(0, 2, 1)

            # Feed raw input to both audio and pathology branches
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f}")

        if val_loader:
            val_loss, val_acc = validate_rawnet1(model, val_loader, device)
            print(f"              --> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    print("Training completed.")

def validate_rawnet1(model, val_loader, device="cuda"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)
            elif inputs.dim() == 3 and inputs.shape[1] != 1:
                inputs = inputs.permute(0, 2, 1)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(val_loader.dataset)
    accuracy = 100.0 * correct / total
    model.train()
    return avg_loss, accuracy

def test_rawnet1(model, test_loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)
            elif inputs.dim() == 3 and inputs.shape[1] != 1:
                inputs = inputs.permute(0, 2, 1)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(targets, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Synthetic", "Real"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    return predictions, targets

def save_model_rawnet1(model, path="pretrained_weights/rawnet1.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model RawNet1 saved to {path}")