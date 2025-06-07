import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_rawnet2_with_loaders(model, train_loader, val_loader=None, device="cuda", epochs=20, lr=0.001, patience=5):
    torch.autograd.set_detect_anomaly(True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()
    scaler = GradScaler()

    best_val_loss = float("inf")
    patience_counter = 0
    total_start_time = time.time()

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(enabled=False, dtype=torch.float16, cache_enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f}")
        print(f"              --> Time: {time.time() - start_time:.2f} seconds")

        if val_loader:
            val_loss, val_acc = validate_rawnet2(model, val_loader, device)
            print(f"              --> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"              --> Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        torch.cuda.empty_cache()

    print("Training completed.")
    total_time = time.time() - total_start_time
    print(f"\nTotal training time for RawNet2: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    
def validate_rawnet2(model, val_loader, device="cuda"):
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

def test_rawnet2(model, test_loader, device="cuda"):
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

    cm = confusion_matrix(targets, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Synthetic", "Real"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    return predictions, targets

def save_model_rawnet2(model, path="pretrained_weights/rawnet2.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model RawNet2 saved to {path}")