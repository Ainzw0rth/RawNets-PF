import random
from collections import defaultdict
from torch.utils.data import Subset

# -----------------------------
# Stratified Split Function
# -----------------------------
def stratified_split(dataset, splits=(0.7, 0.15, 0.15), seed=42):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

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