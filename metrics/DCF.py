import numpy as np

def DCF(y_true, y_pred, C_miss=1, C_fa=1, pi_spoof=0.5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fa = np.logical_and(y_pred == 1, y_true == 0).sum()
    fr = np.logical_and(y_pred == 0, y_true == 1).sum()
    n_spoof = (y_true == 0).sum()
    n_bonafide = (y_true == 1).sum()

    if n_spoof == 0 or n_bonafide == 0:
        raise ValueError("Both bonafide and spoofed samples are required.")

    P_miss = fr / n_bonafide
    P_fa = fa / n_spoof
    beta = (C_miss / C_fa) * ((1 - pi_spoof) / pi_spoof)
    return beta * P_miss + P_fa

def actDCF(y_true, y_scores, C_miss=1, C_fa=1, pi_spoof=0.5):
    beta = (C_miss / C_fa) * ((1 - pi_spoof) / pi_spoof)
    tau = -np.log(beta)
    y_pred = (y_scores >= tau).astype(int)
    return DCF(y_true, y_pred, C_miss, C_fa, pi_spoof), tau

def minDCF(y_true, y_scores, C_miss=1, C_fa=1, pi_spoof=0.5):
    min_dcf = float("inf")
    best_thresh = None

    for t in np.unique(y_scores):
        y_pred = (y_scores >= t).astype(int)
        dcf = DCF(y_true, y_pred, C_miss, C_fa, pi_spoof)
        if dcf < min_dcf:
            min_dcf = dcf
            best_thresh = t

    return min_dcf, best_thresh
