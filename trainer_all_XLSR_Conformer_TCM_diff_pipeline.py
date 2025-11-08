import os
import sys
import time
import torch
import numpy as np
import random
from datetime import datetime
from torch.utils.data import DataLoader

# Import utilities
from utils.Logger import Logger
from utils.Seed import set_seed
from utils.Splitter import stratified_split

# Import dataset
from classes.FeatureDataset.CombinedFeatureDataset import CombinedFeatureDataset
from classes.FeatureDataset.ListDataset import ListDataset

# Import model and trainer
from classes.models.XLSR_Conformer_TCM.model_XLSR_Conformer_TCM_diff_pipeline import XLSRConformerTCMDiffPipeline
from classes.models.XLSR_Conformer_TCM.trainer_XLSR_Conformer_TCM import (
    train_xlsr_conformer_tcm_with_loaders,
    test_xlsr_conformer_tcm
)

# -----------------------------
# Main Training Script
# -----------------------------
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

    # Logger setup
    os.makedirs("logs/train/", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/train/train_log_xlsr_conformer_tcm_diff_pipeline_{timestamp}.txt"
    log_file = open(log_filename, "w")
    sys.stdout = Logger(sys.stdout, log_file)
    sys.stderr = Logger(sys.stderr, log_file)

    seed = 42
    set_seed(seed)

    # Load full dataset
    print("==================== LOADING DATASET ====================\n")

    spoof_dirs = [
        "preprocessed_data/combined/Spoof/Converted/FacebookMMS",
        "preprocessed_data/combined/Spoof/Converted/GoogleTTS",
        "preprocessed_data/combined/Spoof/Converted/VITS",
        "preprocessed_data/combined/Spoof/TTS/FacebookMMS",
        "preprocessed_data/combined/Spoof/TTS/GoogleTTS",
        "preprocessed_data/combined/Spoof/TTS/VITS"
    ]
    
    bonafide_dirs = [
        "preprocessed_data/combined/Bonafide/CommonVoice",
        "preprocessed_data/combined/Bonafide/Prosa"
    ]
    bonafide_dirs = [
        "preprocessed_data/combined/Bonafide/CommonVoice",
        "preprocessed_data/combined/Bonafide/Prosa"
    ]

    # Split each dataset individually, then combine corresponding splits
    train_samples = []
    val_samples = []
    test_samples = []

    # Process spoof datasets
    for spoof_dir in spoof_dirs:
        if os.path.exists(spoof_dir):
            dataset = CombinedFeatureDataset(spoof_dir, force_label=0)
            spoof_samples = [(features, 0) for features, _ in dataset.samples]
            spoof_dataset = ListDataset(spoof_samples)
            t, v, te = stratified_split(spoof_dataset, splits=(0.7, 0.15, 0.15), seed=seed)
            train_samples.extend([spoof_dataset[i] for i in range(len(t))])
            val_samples.extend([spoof_dataset[i] for i in range(len(v))])
            test_samples.extend([spoof_dataset[i] for i in range(len(te))])
        else:
            print(f"Warning: Directory not found: {spoof_dir}")

    # Process bonafide datasets
    for bonafide_dir in bonafide_dirs:
        if os.path.exists(bonafide_dir):
            dataset = CombinedFeatureDataset(bonafide_dir, force_label=1)
            bonafide_samples = [(features, 1) for features, _ in dataset.samples]
            bonafide_dataset = ListDataset(bonafide_samples)
            t, v, te = stratified_split(bonafide_dataset, splits=(0.7, 0.15, 0.15), seed=seed)
            train_samples.extend([bonafide_dataset[i] for i in range(len(t))])
            val_samples.extend([bonafide_dataset[i] for i in range(len(v))])
            test_samples.extend([bonafide_dataset[i] for i in range(len(te))])
        else:
            print(f"Warning: Directory not found: {bonafide_dir}")

    train_dataset = ListDataset(train_samples)
    val_dataset = ListDataset(val_samples)
    test_dataset = ListDataset(test_samples)
    full_dataset = ListDataset(train_samples + val_samples + test_samples)

    print(f"Loaded {len(full_dataset)} samples from spoof and bonafide directories.")
    print("\n==================== DATASET LOADED ====================\n")

    # Print dataset sizes
    print(f"Total samples: {len(full_dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("\n==================== DATASET SPLITTED ====================\n")

    # Training parameters
    batch_sizes = [32]
    learning_rates = [0.000001]
    epochs = 30
    weight_decay = 0.0001

    start_time = time.time()
    print("\n==================== TRAINING STARTED ====================\n")

    try:
        for batch_size in batch_sizes:
            # DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            print(f"\n=========== DATA LOADERS ===========")
            print(f"Train batches: {len(train_loader)}")
            print(f"Validation batches: {len(val_loader)}")
            print(f"Test batches: {len(test_loader)}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            for lr_idx, learning_rate in enumerate(learning_rates):
                print(f"\n===== Training with Batch Size: {batch_size}, Learning Rate: {learning_rate} =====")
                
                # -----------------------------
                # XLSR-Conformer-TCM Diff Pipeline (with Pathological Features)
                # -----------------------------
                
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

                print(f"Device: {device}")
                model = XLSRConformerTCMDiffPipeline(model_config, device).to(device)
                
                # Count parameters
                nb_params = sum([param.view(-1).size()[0] for param in model.parameters() if param.requires_grad])
                print(f'Number of trainable parameters: {nb_params:,}')

                # Train XLSR-Conformer-TCM Diff Pipeline
                print("\n=== Training XLSR-Conformer-TCM Diff Pipeline ===")
                train_xlsr_conformer_tcm_with_loaders(
                    model, 
                    train_loader, 
                    val_loader, 
                    device=device, 
                    epochs=epochs, 
                    lr=learning_rate, 
                    start_epoch=0, 
                    variation="diff_pipeline",
                    weight_decay=weight_decay
                )

                # Test XLSR-Conformer-TCM Diff Pipeline
                print("\n--- Testing XLSR-Conformer-TCM Diff Pipeline ---")
                predictions, targets, metrics = test_xlsr_conformer_tcm(model, test_loader, device=device)

                # Clear CUDA memory
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time

        print("\n==================== TRAINING COMPLETED ====================\n")
        print(f"Total time taken: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

        print("\n===============================================================")
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
