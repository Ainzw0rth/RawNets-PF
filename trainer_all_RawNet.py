import os
import numpy as np
import torch
from torch.utils.data import Dataset

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
        audio = np.load(file_path)  # Load preprocessed npy file
        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Example usage
dataset = SegmentedAudioDataset()
print(len(dataset))  # Number of segments