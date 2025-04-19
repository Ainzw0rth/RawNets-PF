import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_rawnet3_with_loaders(model, train_loader, val_loader=None, test_loader=None, class_labels=None, device="cuda", epochs=40, lr=0.001):
    torch.autograd.set_detect_anomaly(True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # RawNet3 expects (B, T)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f}")

        if val_loader:
            val_loss, val_acc = validate_rawnet3(model, val_loader, device)
            print(f"              --> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
        torch.cuda.empty_cache()

    print("Training completed.")

    if test_loader:
        test_rawnet3(model, test_loader, class_labels, device)

def validate_rawnet3(model, val_loader, device="cuda"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

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

def test_rawnet3(model, test_loader, class_labels=None, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    if class_labels is None:
        class_labels = ["Class 0", "Class 1"]

    cm = confusion_matrix(targets, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    return predictions, targets

def save_model_rawnet3(model, path="pretrained_weights/rawnet3.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model RawNet3 saved to {path}")