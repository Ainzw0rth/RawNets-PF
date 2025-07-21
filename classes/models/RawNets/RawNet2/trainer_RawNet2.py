import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from metrics.CLLR import CLLR
from metrics.DCF import actDCF, minDCF
from metrics.EER import EER

def train_rawnet2_with_loaders(model, train_loader, val_loader=None, device="cuda", epochs=100, lr=0.0001, start_epoch=0, variation="combined"):
    torch.autograd.set_detect_anomaly(True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()
    scaler = GradScaler()

    total_start_time = time.time()

    for epoch in range(start_epoch, epochs):
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
            metrics = validate_rawnet2(model, val_loader, device)
            print(f"              --> Val Loss: {metrics['loss']:.4f} | Val Acc: {metrics['accuracy']:.2f}%")
            print(f"              --> Balanced Acc: {metrics['balanced_accuracy']:.4f} | Precision: {metrics['precision']:.4f}")
            print(f"              --> Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | F2: {metrics['f2']:.4f}")
            print(f"              --> EER: {metrics['eer']:.4f} | actDCF: {metrics['actDCF']:.4f} | minDCF: {metrics['minDCF']:.4f}")
            print(f"              --> CLLR: {metrics['cllr']:.4f}")

        torch.cuda.empty_cache()

        save_model_rawnet2(model, optimizer, scaler, epoch, path=f"pretrained_weights/RawNet2/rawnet2_{variation}-ep_{epoch+1}-bs_{train_loader.batch_size}-lr_{lr}.pth")

    print("Training completed.")
    total_time = time.time() - total_start_time
    print(f"\nTotal training time for RawNet2: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    
def validate_rawnet2(model, val_loader, device="cuda"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []


    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # assuming class 1 is "real"
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = torch.tensor(all_labels).numpy()
    y_pred = torch.tensor(all_preds).numpy()
    y_prob = torch.tensor(all_probs).numpy()

    metrics = {
        "loss": running_loss / len(val_loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "eer": EER(y_true, y_prob),
        "actDCF": actDCF(y_true, y_prob),
        "minDCF": minDCF(y_true, y_prob),
        "cllr": CLLR(y_true, y_prob)
    }
    model.train()
    return metrics

def test_rawnet2(model, test_loader, device="cuda"):
    model.eval()
    predictions = []
    targets = []
    probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            prob = torch.softmax(outputs, dim=1)[:, 1]  # prob for class 1
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    y_true = torch.tensor(targets).numpy()
    y_pred = torch.tensor(predictions).numpy()
    y_prob = torch.tensor(probs).numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "eer": EER(y_true, y_prob),
        "actDCF": actDCF(y_true, y_prob),
        "minDCF": minDCF(y_true, y_prob),
        "cllr": CLLR(y_true, y_prob)
    }

    print(f"              --> Val Acc: {metrics['accuracy']:.2f}%")
    print(f"              --> Balanced Acc: {metrics['balanced_accuracy']:.4f} | Precision: {metrics['precision']:.4f}")
    print(f"              --> Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | F2: {metrics['f2']:.4f}")
    print(f"              --> EER: {metrics['eer']:.4f} | actDCF: {metrics['actDCF']:.4f} | minDCF: {metrics['minDCF']:.4f}")
    print(f"              --> CLLR: {metrics['cllr']:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Synthetic", "Real"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    return predictions, targets, metrics

def save_model_rawnet2(model, optimizer, scaler, epoch, path="pretrained_weights/rawnet2.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict()
    }, path)

    print(f"Model RawNet2 saved to {path}")

def load_model_rawnet2(model, optimizer=None, scaler=None, path=None, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    print(f"Loaded model from {path} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"]