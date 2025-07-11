import numpy as np

def EER(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    false_accepts = np.logical_and(y_pred == 1, y_true == 0).sum()
    false_rejects = np.logical_and(y_pred == 0, y_true == 1).sum()

    total_spoof = (y_true == 0).sum()
    total_bonafide = (y_true == 1).sum()

    # Avoid division by zero
    if total_spoof == 0 or total_bonafide == 0:
        raise ValueError("Both bonafide and spoofed samples are required.")

    FAR = false_accepts / total_spoof
    FRR = false_rejects / total_bonafide

    eer = (FAR + FRR) / 2
    return eer