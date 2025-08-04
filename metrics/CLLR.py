import numpy as np
from scipy.special import expit

def CLLR(y_true, y_scores, eps=1e-10):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    y_prob = expit(y_scores)
    # Clip probabilities to avoid log(0)
    y_prob = np.clip(y_prob, eps, 1 - eps)

    bonafide_prob = y_prob[y_true == 1]
    spoof_prob = y_prob[y_true == 0]

    c1 = -np.mean(np.log2(bonafide_prob))
    c2 = -np.mean(np.log2(1 - spoof_prob))

    cllr = 0.5 * (c1 + c2)
    return cllr