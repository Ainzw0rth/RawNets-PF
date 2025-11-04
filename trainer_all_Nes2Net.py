import os
import sys
import time
import torch
from datetime import datetime
from torch.utils.data import DataLoader

# Import utils - adjust paths as needed based on your project structure
try:
    from utils.Logger import Logger
    from utils.Seed import set_seed
    from utils.Splitter import stratified_split
    from classes.FeatureDataset.WaveformFeatureDataset import WaveformFeatureDataset
    from classes.FeatureDataset.ListDataset import ListDataset
except ImportError:
    print("Warning: Some utility modules not found. Using fallbacks.")
    # Fallback implementations if needed
    class Logger:
        def __init__(self, stdout, file):
            self.stdout = stdout
            self.file = file
        def write(self, text):
            self.stdout.write(text)
            self.file.write(text)
        def flush(self):
            self.stdout.flush()
            self.file.flush()
    
    def set_seed(seed):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def stratified_split(dataset, splits=(0.7, 0.15, 0.15), seed=42):
        # Simple implementation
        set_seed(seed)
        indices = list(range(len(dataset)))
        import random
        random.shuffle(indices)
        
        n = len(indices)
        n_train = int(n * splits[0])
        n_val = int(n * splits[1])
        
        return indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]

# Import Nes2Net components
from classes.models.Nes2Net.model_Nes2Net import WavLMNes2Net
from classes.models.Nes2Net.trainer_Nes2Net import (
    train_nes2net_with_loaders,
    test_nes2net,
    save_model_nes2net
)

# -----------------------------
# Main Training Script
# -----------------------------
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

    # Logger setup
    os.makedirs("logs/train/", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/train/train_log_nes2net_{timestamp}.txt"
    log_file = open(log_filename, "w")
    sys.stdout = Logger(sys.stdout, log_file)
    sys.stderr = Logger(sys.stderr, log_file)

    seed = 42
    set_seed(seed)

    # Load full dataset
    print("==================== LOADING DATASET ====================\n")

    spoof_dirs = [
        "preprocessed_data/waveform/Spoof/Converted/FacebookMMS",
        "preprocessed_data/waveform/Spoof/Converted/GoogleTTS",
        "preprocessed_data/waveform/Spoof/Converted/VITS",
        "preprocessed_data/waveform/Spoof/TTS/FacebookMMS",
        "preprocessed_data/waveform/Spoof/TTS/GoogleTTS",
        "preprocessed_data/waveform/Spoof/TTS/VITS"
    ]
    
    bonafide_dirs = [
        "preprocessed_data/waveform/Bonafide/CommonVoice",
        "preprocessed_data/waveform/Bonafide/Prosa"
    ]

    # Split each dataset individually, then combine corresponding splits
    train_samples = []
    val_samples = []
    test_samples = []

    # Process spoof datasets
    for spoof_dir in spoof_dirs:
        if os.path.exists(spoof_dir):
            dataset = WaveformFeatureDataset(spoof_dir, force_label=0)
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
            dataset = WaveformFeatureDataset(bonafide_dir, force_label=1)
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
    learning_rates = [0.000001]  # Same as in main.py
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
                # WavLM-Nes2Net
                # -----------------------------
                
                # Model configuration
                model_config = {
                    'agg': 'SEA',              # Aggregation method: 'SEA', 'WeightedSum', 'AttM'
                    'Nes_ratio': [8, 8],       # Nested Res2Net ratio
                    'dilation': 2,             # Dilation factor
                    'pool_func': 'mean',       # Pooling function: 'mean' or 'ASTP'
                    'SE_ratio': [8],           # SE module ratio
                    'cp_path': 'wavlm_large.pt',  # Path to pretrained WavLM model (not used with s3prl)
                    'fine_tune_ssl': True      # Whether to fine-tune SSL model
                }

                print(f"Device: {device}")
                model = WavLMNes2Net(model_config, device).to(device)
                
                # Count parameters
                nb_params = sum([param.view(-1).size()[0] for param in model.parameters() if param.requires_grad])
                print(f'Number of trainable parameters: {nb_params:,}')

                # Count SSL and backend parameters separately
                ssl_params = sum([param.view(-1).size()[0] for param in model.ssl_model.parameters() if param.requires_grad])
                backend_params = sum([param.view(-1).size()[0] for param in model.Nested_Res2Net_TDNN.parameters() if param.requires_grad])
                print(f'SSL parameters (with aggregation): {ssl_params:,}')
                print(f'Backend parameters: {backend_params:,}')

                # Train WavLM-Nes2Net
                print("\n=== Training WavLM-Nes2Net ===")
                train_nes2net_with_loaders(
                    model, 
                    train_loader, 
                    val_loader, 
                    device=device, 
                    epochs=epochs, 
                    lr=learning_rate, 
                    start_epoch=0, 
                    variation="waveform",
                    weight_decay=weight_decay
                )

                # Test WavLM-Nes2Net
                print("\n--- Testing WavLM-Nes2Net ---")
                predictions, targets, metrics = test_nes2net(model, test_loader, device=device)

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
