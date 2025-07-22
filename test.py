import os
import sys
import time
import torch
from datetime import datetime
from torch.utils.data import DataLoader

# Import utils
from utils.Logger import Logger
from utils.Seed import set_seed
from utils.Splitter import stratified_split
from classes.FeatureDataset.TestDataset import TestDataset
from classes.FeatureDataset.ListDataset import ListDataset

# Import RawNet1 components
from classes.models.RawNets.RawNet1.model_RawNet1_preprocessed_diff_pipeline import RawNet
from classes.models.RawNets.RawNet1.trainer_RawNet1 import train_rawnet1_with_loaders, test_rawnet1, save_model_rawnet1

# Import RawNet2 components
from classes.models.RawNets.RawNet2.model_RawNet2_preprocessed_diff_pipeline import RawNet2
from classes.models.RawNets.RawNet2.trainer_RawNet2 import train_rawnet2_with_loaders, test_rawnet2, save_model_rawnet2

# Import RawNet3 components
from classes.models.RawNets.RawNet3.model_RawNet3_preprocessed_diff_pipeline import RawNet3
from classes.models.RawNets.RawNet3.trainer_RawNet3 import train_rawnet3_with_loaders, test_rawnet3, save_model_rawnet3

# -----------------------------
# Main Training Script
# -----------------------------
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

    # Logger setup
    os.makedirs("logs/train/", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/train/train_log_combined_{timestamp}.txt"
    log_file = open(log_filename, "w")
    sys.stdout = Logger(sys.stdout, log_file)
    sys.stderr = Logger(sys.stderr, log_file)

    seed = 42
    set_seed(42)

    # Load spoof and bonafide datasets individually, label spoof as 0 and bonafide as 1
    print("==================== LOADING DATASET ====================\n")

    spoof_dirs = [
        "preprocessed_data/combined/Spoof/Converted/ElevenMultilingualV2",
        "preprocessed_data/combined/Spoof/Converted/FacebookMMS",
        "preprocessed_data/combined/Spoof/Converted/GoogleTTS",
        "preprocessed_data/combined/Spoof/Converted/VITS",
        "preprocessed_data/combined/Spoof/TTS/ElevenMultilingualV2",
        "preprocessed_data/combined/Spoof/TTS/FacebookMMS",
        "preprocessed_data/combined/Spoof/TTS/GoogleTTS",
        "preprocessed_data/combined/Spoof/TTS/VITS"
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
            dataset = TestDataset(spoof_dir, force_label=0)
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
            dataset = TestDataset(bonafide_dir, force_label=1)
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

    # Datasets already split and combined above
    print("\n==================== DATASET SPLITTED ====================\n")

    # Looping to do some variations on the models' parameters
    batch_sizes = [32]
    learning_rates = [0.0001]
    epochs = 100

    # Print dataset sizes
    print(f"Total samples: {len(full_dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("\n==================== DATASET SPLITTED ====================\n")

    start_time = time.time()
    print("\n==================== TRAINING STARTED ====================\n")

    try:
        for batch_size in batch_sizes:
            # DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            print(f"\n=========== TRAIN LOADER ===========")
            print(f"Train batches: {len(train_loader)}")
            print(f"Validation batches: {len(val_loader)}")
            print(f"Test batches: {len(test_loader)}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            for lr_idx, learning_rate in enumerate(learning_rates):
                print(f"\n===== Training with Batch Size: {batch_size}, Learning Rate: {learning_rate} =====")
                
                parameter_format = f"ep_{epochs}-bs_{batch_size}-lr_{learning_rate}"
                # -----------------------------
                # RAWNET1
                # -----------------------------
                
                # Model config
                model_config = {
                    'in_channels': 1,
                    'first_conv': 3,
                    'filts': [128, [128, 128], [128, 256], [256, 256]],
                    'blocks': [2, 4],
                    'gru_node': 1024,
                    'nb_gru_layer': 1,
                    'nb_fc_node': 1024,
                    'nb_classes': 2,
                    'input_length': 16000 * 4 + 24
                }

                print(f"device: {device}")
                model = RawNet(model_config, device).to(device)

                # Train RawNet1
                print("\n=== Training RawNet1 ===")
                train_rawnet1_with_loaders(model, train_loader, val_loader, 
                                        device=device, epochs=epochs, lr=learning_rate, start_epoch=0, variation="diff_pipeline")

                # # Test RawNet1
                # print("\n--- Testing RawNet1 ---")
                # predictions, targets = test_rawnet1(model, test_loader, device=device)

                # Clear CUDA memory
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()


                # -----------------------------
                # RAWNET2
                # -----------------------------
                # Model config for RawNet2
                model_config2 = {
                    'in_channels': 1,
                    'first_conv': 3,
                    'filts': [128, [128, 128], [128, 256], [256, 256]],
                    'blocks': [2, 4],
                    'gru_node': 1024,
                    'nb_gru_layer': 1,
                    'nb_fc_node': 1024,
                    'nb_classes': 2,
                    'nb_samp': 16000 * 4
                }

                model2 = RawNet2(model_config2).to(device)

                # Train RawNet2
                print("\n=== Training RawNet2 ===")
                train_rawnet2_with_loaders(model2, train_loader, val_loader, 
                                        device=device, epochs=epochs, lr=learning_rate, start_epoch=0, variation="diff_pipeline")

                # # Test RawNet2
                # print("\n--- Testing RawNet2 ---")
                # predictions2, targets2 = test_rawnet2(model2, test_loader, device=device)

                # Clear CUDA memory
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()


                # -----------------------------
                # RAWNET3
                # -----------------------------
                # Model config for RawNet3
                model_config3 = {
                    "nOut": 2,
                    "sinc_stride": 10,
                    "encoder_type": "ECA",
                    "log_sinc": True,
                    "norm_sinc": "mean_std",
                    "out_bn": True
                }

                model3 = RawNet3(**model_config3).to(device)

                # Train RawNet3
                print("\n=== Training RawNet3 ===")
                train_rawnet3_with_loaders(model3, train_loader, val_loader,
                                        device=device, epochs=epochs, lr=learning_rate, start_epoch=0, variation="diff_pipeline")

                # # Test RawNet3
                # print("\n--- Testing RawNet3 ---")
                # predictions3, targets3 = test_rawnet3(model3, test_loader, device=device)
 
                # Clear CUDA memory
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()   
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time

        print("\n==================== PREPROCESSING COMPLETED ====================\n")
        print(f"Total time taken: {elapsed_time:.2f} seconds")

        print("\n===============================================================")
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()