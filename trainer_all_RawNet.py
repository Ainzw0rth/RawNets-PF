import os
import numpy as np
import random
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset

# Import RawNet1 components
from RawNets.RawNet1.model_RawNet1 import RawNet
from RawNets.RawNet1.trainer_RawNet1 import train_rawnet1_with_loaders, test_rawnet1

# Import RawNet2 components
from RawNets.RawNet2.model_RawNet2 import RawNet2
from RawNets.RawNet2.trainer_RawNet2 import train_rawnet2_with_loaders, test_rawnet2

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
    set_seed(42)

    # Load full dataset
    full_dataset = SegmentedAudioDataset("preprocessed_data/")

    # Stratified dataset split
    train_dataset, val_dataset, test_dataset = stratified_split(full_dataset, splits=(0.7, 0.15, 0.15), seed=42)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=120, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=120, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=120, shuffle=False)

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    train_rawnet1_with_loaders(model, train_loader, val_loader, device=device, epochs=40, lr=0.001)

    # Test RawNet1
    print("\n--- Testing RawNet1 ---")
    predictions, targets = test_rawnet1(model, test_loader, device=device)


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
        'nb_samp': 16000,
        'pathology_dim': 128
    }

    model2 = RawNet2(model_config2, pathology_dim=model_config2['pathology_dim']).to(device)

    # Train RawNet2
    print("\n=== Training RawNet2 ===")
    train_rawnet2_with_loaders(model2, train_loader, val_loader, class_labels=["synthetic voice", "real voice"], device=device, epochs=40, lr=0.001)

    # Test RawNet2
    print("\n--- Testing RawNet2 ---")
    predictions2, targets2 = test_rawnet2(model2, test_loader, device=device)