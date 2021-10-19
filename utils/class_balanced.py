import numpy as np
import torch
import torch.nn as nn

def class_balanced_weights(labels, nc=80, beta=.9999):
    labels = np.concatenate(labels, 0) # labels.shape = ? (866643, 5) (849942, 5)
    #print(labels.shape)
    classes = labels[:, 0].astype(np.int) # labels = [class xywh]
    #print(classes)
    occ_per_class = np.bincount(classes, minlength=nc) # occurences per class. 
    #print(occ_per_class)

    effective_num = 1.0 - np.power(beta, occ_per_class)
    #print(effective_num)
    weights = (1.0 - beta) / np.array(effective_num)
    #print(weights)
    weights = weights / np.sum(weights) * nc
    #print(weights)
    return torch.from_numpy(weights)
