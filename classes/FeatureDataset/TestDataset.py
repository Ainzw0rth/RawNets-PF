import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, root_dir, label_map={"Bonafide": 1, "Spoof": 0}, force_label=None):
        self.samples = []
        self.label_map = label_map
        self.skipped_files = 0

        # If force_label is set, assign all samples this label
        if force_label is not None:
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if not file.endswith(".npy"):
                        continue
                    file_path = os.path.join(root, file)
                    try:
                        feature_data = np.load(file_path)
                        if not np.isfinite(feature_data).all():
                            print(f"Skipping file {file_path} due to NaN or Inf values")
                            self.skipped_files += 1
                            continue
                        feature_data = feature_data.flatten()
                        if feature_data.shape[0] < 16000 * 4 + 24:
                            pad_width = 16000 * 4 + 24 - feature_data.shape[0]
                            feature_data = np.pad(feature_data, (0, pad_width), mode='constant')
                        else:
                            feature_data = feature_data[:16000 * 4 + 24]
                        features = torch.tensor(feature_data, dtype=torch.float32)
                        self.samples.append((features, force_label))
                    except Exception as e:
                        print(f"Skipping file {file_path} due to error: {e}")
        else:
            # Original behavior: expects subfolders for each class
            for label_name in os.listdir(root_dir):
                label_dir = os.path.join(root_dir, label_name)
                if not os.path.isdir(label_dir):
                    continue
                for root, _, files in os.walk(label_dir):
                    for file in files:
                        if not file.endswith(".npy"):
                            continue
                        file_path = os.path.join(root, file)
                        try:
                            feature_data = np.load(file_path)
                            if not np.isfinite(feature_data).all():
                                print(f"Skipping file {file_path} due to NaN or Inf values")
                                self.skipped_files += 1
                                continue
                            feature_data = feature_data.flatten()
                            if feature_data.shape[0] < 16000 * 4 + 24:
                                pad_width = 16000 * 4 + 24 - feature_data.shape[0]
                                feature_data = np.pad(feature_data, (0, pad_width), mode='constant')
                            else:
                                feature_data = feature_data[:16000 * 4 + 24]
                            features = torch.tensor(feature_data, dtype=torch.float32)
                            label = label_map["Bonafide" if "bonafide" in label_name.lower() else "Spoof"]
                            self.samples.append((features, label))
                        except Exception as e:
                            print(f"Skipping file {file_path} due to error: {e}")
        print(f"Skipped {self.skipped_files} files due to NaNs/Infs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]