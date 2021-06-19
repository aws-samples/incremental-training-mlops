import torch.nn as nn


def CrossEntropyLoss(output, target):
    criterion = nn.CrossEntropyLoss()
    return criterion(output, target)
