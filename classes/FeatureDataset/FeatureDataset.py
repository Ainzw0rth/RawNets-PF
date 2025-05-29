import csv
import sys
import ast
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, tsv_path, label_map={"Bonafide": 1, "Spoof": 0}):
        self.samples = []
        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter='\,')
            for row in reader:
                features = torch.tensor(ast.literal_eval(row["features"]), dtype=torch.float32)
                label_str = "Bonafide" if "bonafide" in row["path"].lower() else "Spoof"
                label = label_map[label_str]
                self.samples.append((features, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
