import os
import sys
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import model and trainer
from classes.models.XLSR_Conformer_TCM.model_XLSR_Conformer_TCM_diff_pipeline import XLSRConformerTCMDiffPipeline
from classes.models.XLSR_Conformer_TCM.trainer_XLSR_Conformer_TCM import (
    train_xlsr_conformer_tcm_with_loaders,
    test_xlsr_conformer_tcm_with_loaders
)

# Import dataset
from classes.FeatureDataset.CombinedFeatureDataset import CombinedFeatureDataset

# Import utilities
from utils.Seed import set_seed
from utils.Logger import setup_logger


def main():
    # ================================================================================
    # Configuration
    # ================================================================================
    
    # Set random seed for reproducibility
    SEED = 42
    set_seed(SEED)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model configuration
    model_config = {
        'emb_size': 144,           # Embedding size for Conformer
        'num_encoders': 4,         # Number of Conformer encoder blocks
        'heads': 4,                # Number of attention heads
        'kernel_size': 31,         # Kernel size for convolutional modules
        'cp_path': 'xlsr2_300m.pt', # Path to XLSR checkpoint
        'fine_tune_ssl': True,     # Whether to fine-tune SSL model
        'nb_patho_features': 24    # Number of pathological features
    }
    
    # Training configuration
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-6
    weight_decay = 0.0001
    
    # Data paths
    data_root = "preprocessed_data/combined"  # Combined waveform + pathological features
    train_path = os.path.join(data_root, "train")
    val_path = os.path.join(data_root, "val")
    test_path = os.path.join(data_root, "test")
    
    # Model save path
    variation = "diff_pipeline"
    save_dir = f"new_pretrained_weights/{variation}/XLSR_Conformer_TCM"
    os.makedirs(save_dir, exist_ok=True)
    
    # Logging
    logger = setup_logger(
        name="XLSR_Conformer_TCM_DiffPipeline",
        log_file=f"logs/train/train_log_xlsr_conformer_tcm_{variation}.txt"
    )
    
    logger.info("="*80)
    logger.info("XLSR-Conformer-TCM with Different Pipeline Training")
    logger.info("="*80)
    logger.info(f"Model Config: {model_config}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Learning Rate: {learning_rate}")
    logger.info(f"Weight Decay: {weight_decay}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Device: {device}")
    logger.info("="*80)
    
    # ================================================================================
    # Data Loading
    # ================================================================================
    
    logger.info("Loading datasets...")
    
    # Create datasets
    train_dataset = CombinedFeatureDataset(train_path)
    val_dataset = CombinedFeatureDataset(val_path)
    test_dataset = CombinedFeatureDataset(test_path)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # ================================================================================
    # Model Creation
    # ================================================================================
    
    logger.info("Creating XLSR-Conformer-TCM (Different Pipeline) model...")
    model = XLSRConformerTCMDiffPipeline(model_config, device)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # ================================================================================
    # Training
    # ================================================================================
    
    logger.info("Starting training...")
    
    train_xlsr_conformer_tcm_with_loaders(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=num_epochs,
        lr=learning_rate,
        start_epoch=0,
        variation=variation,
        weight_decay=weight_decay
    )
    
    logger.info("Training completed!")
    
    # ================================================================================
    # Testing
    # ================================================================================
    
    logger.info("Starting testing on test set...")
    
    predictions, targets, probs = test_xlsr_conformer_tcm_with_loaders(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    # Calculate and log test metrics
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    try:
        from metrics.EER import EER
        eer = EER(targets, probs)
    except:
        eer = 0.0
    
    accuracy = accuracy_score(targets, predictions) * 100
    balanced_acc = balanced_accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    
    logger.info("="*80)
    logger.info("FINAL TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"EER: {eer:.4f}")
    logger.info("="*80)
    
    logger.info("Training and testing completed successfully!")


if __name__ == "__main__":
    main()
