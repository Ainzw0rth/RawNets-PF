import os
import sys
import numpy as np
import random
import torch
from datetime import datetime
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset

# Import RawNet1 components
from RawNets.RawNet1.model_RawNet1 import RawNet
from RawNets.RawNet1.trainer_RawNet1 import train_rawnet1_with_loaders, test_rawnet1, save_model_rawnet1

# Import RawNet2 components
from RawNets.RawNet2.model_RawNet2 import RawNet2
from RawNets.RawNet2.trainer_RawNet2 import train_rawnet2_with_loaders, test_rawnet2, save_model_rawnet2

# Import RawNet3 components
from RawNets.RawNet3.model_RawNet3 import RawNet3
from RawNets.RawNet3.trainer_RawNet3 import train_rawnet3_with_loaders, test_rawnet3, save_model_rawnet3

# -----------------------------
# Logger Setup
# -----------------------------
class Logger:
    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for f in self.files:
            f.write(message)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

# -----------------------------
# Reproducibility Setup
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Custom Dataset
# -----------------------------
class SegmentedAudioDataset(Dataset):
    def __init__(self, base_dir="preprocessed_data/"):
        self.data = []
        self.labels = {"synthetic voice": 0, "real voice": 1}

        for label_name, label in self.labels.items():
            folder_path = os.path.join(base_dir, label_name)
            if not os.path.exists(folder_path):
                continue

            for file_name in os.listdir(folder_path):
                if file_name.endswith(".npy"):
                    file_path = os.path.join(folder_path, file_name)
                    self.data.append((file_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        audio = np.load(file_path)
        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# -----------------------------
# Stratified Split Function
# -----------------------------
def stratified_split(dataset, splits=(0.7, 0.15, 0.15), seed=42):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label.item()].append(idx)

    train_indices, val_indices, test_indices = [], [], []
    random.seed(seed)

    for label, indices in class_indices.items():
        random.shuffle(indices)
        total = len(indices)
        n_train = int(splits[0] * total)
        n_val = int(splits[1] * total)

        train_indices += indices[:n_train]
        val_indices += indices[n_train:n_train + n_val]
        test_indices += indices[n_train + n_val:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, val_subset, test_subset

# -----------------------------
# Main Training Script
# -----------------------------
if __name__ == "__main__":
    # Logger setup
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/train_log_{timestamp}.txt"
    log_file = open(log_filename, "w")
    sys.stdout = Logger(sys.stdout, log_file)
    sys.stderr = Logger(sys.stderr, log_file)

    seed = 42
    set_seed(42)

    # Load full dataset
    full_dataset = SegmentedAudioDataset("preprocessed_data/")

    # Stratified dataset split
    train_dataset, val_dataset, test_dataset = stratified_split(full_dataset, splits=(0.7, 0.15, 0.15), seed=seed)

    # Looping to do some variations on the models' parameters
    batch_sizes = [8, 16, 32]
    learning_rates = [0.001, 0.0005, 0.0001]
    epochs = 20
    patience = 5

    for batch_size in batch_sizes:
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for lr_idx, learning_rate in enumerate(learning_rates):
            print(f"\n===== Training with Batch Size: {batch_size}, Learning Rate: {learning_rate} =====")
            
            parameter_format = f"ep_{epochs}-bs_{batch_size}-lr_{learning_rate}-pa_{patience}"
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
                'nb_classes': 2
            }
            model = RawNet(model_config, device).to(device)

            # Train RawNet1
            print("\n=== Training RawNet1 ===")
            train_rawnet1_with_loaders(model, train_loader, val_loader, 
                                    device=device, epochs=epochs, lr=learning_rate, patience=patience)

            # # Test RawNet1
            # print("\n--- Testing RawNet1 ---")
            # predictions, targets = test_rawnet1(model, test_loader, device=device)

            # Save RawNet1 model
            print("\n=== Saving RawNet1 Model ===")
            save_model_rawnet1(model, f"./RawNets/RawNet1/pretrained_weights/rawnet1-{parameter_format}.pth")


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
                'nb_samp': 16000
            }

            model2 = RawNet2(model_config2).to(device)

            # Train RawNet2
            print("\n=== Training RawNet2 ===")
            train_rawnet2_with_loaders(model2, train_loader, val_loader, 
                                    device=device, epochs=epochs, lr=learning_rate, patience=patience)

            # # Test RawNet2
            # print("\n--- Testing RawNet2 ---")
            # predictions2, targets2 = test_rawnet2(model2, test_loader, device=device)

            # Save RawNet2 model
            print("\n=== Saving RawNet2 Model ===")
            save_model_rawnet2(model, f"./RawNets/RawNet2/pretrained_weights/rawnet2-{parameter_format}.pth")


            # -----------------------------
            # RAWNET3
            # -----------------------------
            # Model config for RawNet3
            model_config3 = {
                "nOut": 512,
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
                                    device=device, epochs=epochs, lr=learning_rate, patience=patience)

            # # Test RawNet3
            # print("\n--- Testing RawNet3 ---")
            # predictions3, targets3 = test_rawnet3(model3, test_loader, device=device)

            # Save RawNet3 model
            print("\n=== Saving RawNet3 Model ===")
            save_model_rawnet3(model, f"./RawNets/RawNet3/pretrained_weights/rawnet3-{parameter_format}.pth")

    log_file.close()