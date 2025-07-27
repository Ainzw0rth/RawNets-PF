import numpy as np
from sklearn.metrics import roc_curve

def EER(y_true, y_scores):
    # Compute ROC curve
    FAR, TPR, thresholds = roc_curve(y_true, y_scores)
    FRR = 1 - TPR  # FRR = 1 - TPR

    # Find threshold where FAR and FRR are closest
    eer_threshold_idx = np.nanargmin(np.abs(FAR - FRR))
    eer = (FAR[eer_threshold_idx] + FRR[eer_threshold_idx]) / 2

    return eer