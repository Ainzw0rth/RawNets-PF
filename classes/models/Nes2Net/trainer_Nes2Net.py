from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
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
from tqdm import tqdm

# ================================================================================
# WavLM-Nes2Net Trainer Functions
# ================================================================================
# This trainer module supports both:
# 1. WavLMNes2Net (model_Nes2Net.py) - Regular WavLM-Nes2Net with waveform input only
# 2. WavLMNes2NetDiffPipeline (model_Nes2Net_diff_pipeline.py) - WavLM-Nes2Net with 
#    pathological features using different pipeline approach
#
# All functions are model-agnostic and work with both architectures.
# ================================================================================

# Assuming metrics are available in these paths - adjust as needed
try:
    from metrics.CLLR import CLLR
    from metrics.DCF import actDCF, minDCF
    from metrics.EER import EER
except ImportError:
    print("Warning: Metrics modules not found. Using placeholder metrics.")
    def CLLR(y_true, y_prob): return 0.0
    def actDCF(y_true, y_prob): return 0.0
    def minDCF(y_true, y_prob): return 0.0
    def EER(y_true, y_prob): return 0.0


def train_nes2net_with_loaders(model, train_loader, val_loader=None, device="cuda", epochs=100, 
                                lr=0.0001, start_epoch=0, variation="combined", weight_decay=0.0001):
    """
    Train WavLM-Nes2Net model with data loaders (works with both regular and diff_pipeline versions)
    
    Args:
        model: The WavLM-Nes2Net model (WavLMNes2Net or WavLMNes2NetDiffPipeline)
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of epochs to train
        lr: Learning rate
        start_epoch: Starting epoch (for resuming training)
        variation: Variation name for saving models
        weight_decay: Weight decay for optimizer
    """
    torch.autograd.set_detect_anomaly(True)
    
    # Weighted Cross Entropy Loss (0.1 for spoof, 0.9 for bonafide)
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)
    model.train()
    scaler = GradScaler()

    total_start_time = time.time()

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        running_loss = 0.0
        num_total = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            num_total += batch_size

            with autocast(enabled=False, dtype=torch.float16, cache_enabled=True):
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * batch_size
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / num_total
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f}")
        print(f"              --> Time: {time.time() - start_time:.2f} seconds")

        if val_loader:
            metrics = validate_nes2net(model, val_loader, device)
            print(f"              --> Val Loss: {metrics['loss']:.4f} | Val Acc: {metrics['accuracy']:.2f}%")
            print(f"              --> Balanced Acc: {metrics['balanced_accuracy']:.4f} | Precision: {metrics['precision']:.4f}")
            print(f"              --> Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | F2: {metrics['f2']:.4f}")
            print(f"              --> EER: {metrics['eer']:.4f} | actDCF: {metrics['actDCF']:.4f} | minDCF: {metrics['minDCF']:.4f}")
            print(f"              --> CLLR: {metrics['cllr']:.4f}")

        torch.cuda.empty_cache()

        save_model_nes2net(
            model, optimizer, scaler, epoch, 
            path=f"pretrained_weights/{variation}/Nes2Net/nes2net_{variation}-ep_{epoch+1}-bs_{train_loader.batch_size}-lr_{lr}.pth"
        )

    print("Training completed.")
    total_time = time.time() - total_start_time
    print(f"\nTotal training time for WavLM-Nes2Net: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")


def validate_nes2net(model, val_loader, device="cuda"):
    """
    Validate WavLM-Nes2Net model (works with both regular and diff_pipeline versions)
    
    Args:
        model: The WavLM-Nes2Net model (WavLMNes2Net or WavLMNes2NetDiffPipeline)
        val_loader: Validation data loader
        device: Device to validate on
    
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    running_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

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


def test_nes2net(model, test_loader, class_labels=None, device="cuda"):
    """
    Test WavLM-Nes2Net model and display results (works with both regular and diff_pipeline versions)
    
    Args:
        model: The WavLM-Nes2Net model (WavLMNes2Net or WavLMNes2NetDiffPipeline)
        test_loader: Test data loader
        class_labels: Labels for confusion matrix display
        device: Device to test on
    
    Returns:
        Tuple of (predictions, targets, metrics)
    """
    model.eval()
    predictions = []
    targets = []
    probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)

            prob = torch.softmax(outputs, dim=1)[:, 1]
            pred = torch.argmax(outputs, dim=1)

            predictions.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    y_true = np.array(targets)
    y_pred = np.array(predictions)
    y_prob = np.array(probs)

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

    print(f"              --> Test Acc: {metrics['accuracy']:.2f}%")
    print(f"              --> Balanced Acc: {metrics['balanced_accuracy']:.4f} | Precision: {metrics['precision']:.4f}")
    print(f"              --> Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | F2: {metrics['f2']:.4f}")
    print(f"              --> EER: {metrics['eer']:.4f} | actDCF: {metrics['actDCF']:.4f} | minDCF: {metrics['minDCF']:.4f}")
    print(f"              --> CLLR: {metrics['cllr']:.4f}")
    
    # Confusion Matrix
    if class_labels is None:
        class_labels = ["Synthetic", "Real"]
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - WavLM-Nes2Net")
    plt.show()

    return predictions, targets, metrics


def test_nes2net_with_loaders(model, test_loader, device="cuda"):
    """
    Test WavLM-Nes2Net model (returns only predictions) - works with both regular and diff_pipeline versions
    
    Args:
        model: The WavLM-Nes2Net model (WavLMNes2Net or WavLMNes2NetDiffPipeline)
        test_loader: Test data loader
        device: Device to test on
    
    Returns:
        Tuple of (predictions, targets, probabilities)
    """
    model.eval()
    predictions = []
    targets = []
    probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)

            prob = torch.softmax(outputs, dim=1)[:, 1]
            pred = torch.argmax(outputs, dim=1)

            predictions.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    y_true = np.array(targets)
    y_pred = np.array(predictions)
    y_prob = np.array(probs)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Synthetic", "Real"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - WavLM-Nes2Net")
    plt.show()
    
    torch.cuda.empty_cache()
    
    return predictions, targets, probs


def save_model_nes2net(model, optimizer, scaler, epoch, path="pretrained_weights/nes2net.pth"):
    """
    Save WavLM-Nes2Net model checkpoint
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        scaler: The GradScaler state
        epoch: Current epoch number
        path: Path to save the checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict()
    }, path)

    print(f"Model WavLM-Nes2Net saved to {path}")


def load_model_nes2net(model, optimizer=None, scaler=None, path=None, device="cuda"):
    """
    Load WavLM-Nes2Net model checkpoint
    
    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into (optional)
        scaler: The GradScaler to load state into (optional)
        path: Path to the checkpoint file
        device: Device to load the model on
    
    Returns:
        The epoch number from the checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    print(f"Loaded model from {path} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"]


def produce_evaluation_file(model, dataset, device, save_path):
    """
    Generate evaluation scores file for ASVspoof evaluation
    
    Args:
        model: The trained model
        dataset: Evaluation dataset
        device: Device to run inference on
        save_path: Path to save the scores file
    """
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()
    fname_list = []
    score_list = []
    text_list = []

    with torch.no_grad():
        for batch_x, utt_id in data_loader:
            batch_x = batch_x.to(device)
            batch_out, _ = model(batch_x)
            
            batch_score = torch.softmax(batch_out, dim=1)[:, 1]
            batch_score = batch_score.data.cpu().numpy().ravel()
            
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())

    for f, cm in zip(fname_list, score_list):
        text_list.append('{} {}'.format(f, cm))
    
    del fname_list
    del score_list
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'a+') as fh:
        for i in range(0, len(text_list), 500):
            fh.write('\n'.join(text_list[i:i+500]))
            fh.write('\n')
    
    del text_list
    fh.close()
    print('Scores saved to {}'.format(save_path))
