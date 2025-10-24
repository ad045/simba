import numpy as np 
import networkx as nx
import pandas as pd
import os
from pathlib import Path


def get_eta_and_gamma_from_filename(filename):
    """
    Extract eta and gamma values from the network filename.
    
    Expected filename format:
    net_eta-{eta}_gamma{gamma}_rule{generative_rule}.npy
    """
    import re
    
    eta_match = re.search(r'eta([-+]?\d*\.\d+|\d+)', filename)
    gamma_match = re.search(r'gamma([-+]?\d*\.\d+|\d+)', filename)
    
    eta = float(eta_match.group(1)) if eta_match else None
    gamma = float(gamma_match.group(1)) if gamma_match else None
    
    return eta, gamma



def sorted_listing_by_creation_time(directory):
    def get_creation_time(item):
        item_path = os.path.join(directory, item)
        return os.path.getctime(item_path)

    items = os.listdir(directory)
    sorted_items = sorted(items, key=get_creation_time)
    return sorted_items
