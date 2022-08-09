import numpy as np
from numpy.lib.scimath import log10

def get_lin_edges(min_val, max_val, no_of_classes):
    edges = np.linspace(min_val, max_val, no_of_classes)
    return edges

def get_log_edges(min_val, max_val, no_of_classes):
    edges = 10**np.linspace(log10(min_val), log10(max_val), no_of_classes)
    return edges
