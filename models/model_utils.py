import numpy as np

def get_accuracy(y1, y2):
    return (y1 == y2).mean()