import numpy as np


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((3, 3))
    for p,t in zip(y_pred, y_true):
        if p[0] == 1:
            if t[0] == 1: cm[0][0] = cm[0][0] + 1
            if t[1] == 1: cm[0][1] = cm[0][1] + 1
            if t[2] == 1: cm[0][2] = cm[0][2] + 1
        if p[1] == 1:
            if t[0] == 1: cm[1][0] = cm[1][0] + 1
            if t[1] == 1: cm[1][1] = cm[1][1] + 1
            if t[2] == 1: cm[1][2] = cm[1][2] + 1
        if p[2] == 1:
            if t[0] == 1: cm[2][0] = cm[2][0] + 1
            if t[1] == 1: cm[2][1] = cm[2][1] + 1
            if t[2] == 1: cm[2][2] = cm[2][2] + 1
    return cm