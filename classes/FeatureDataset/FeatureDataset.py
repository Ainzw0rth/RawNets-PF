import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, root_dir, label_map={"Bonafide": 1, "Spoof": 0}):
        self.samples = []
        self.label_map = label_map

        for label_name in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue

            for root, _, files in os.walk(label_dir):
                for file in files:
                    if not file.endswith(".npy"):
                        continue  # skip non-npy files

                    file_path = os.path.join(root, file)
                    try:
                        feature_data = np.load(file_path)
                        # Ensure feature_data is 1D
                        feature_data = feature_data.flatten()
                        # Pad or truncate to length 16000
                        if feature_data.shape[0] < 16000:
                            pad_width = 16000 - feature_data.shape[0]
                            feature_data = np.pad(feature_data, (0, pad_width), mode='constant')
                        else:
                            feature_data = feature_data[:16000]
                        features = torch.tensor(feature_data, dtype=torch.float32)
                        label = label_map["Bonafide" if "bonafide" in label_name.lower() else "Spoof"]
                        self.samples.append((features, label))
                    except Exception as e:
                        print(f"Skipping file {file_path} due to error: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]