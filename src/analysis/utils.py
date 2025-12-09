import numpy as np 
import networkx as nx
import pandas as pd
import os
from pathlib import Path


# def get_eta_and_gamma_from_filename(filename):
#     """
#     Extract eta and gamma values from the network filename.
    
#     Expected filename format:
#     net_eta-{eta}_gamma{gamma}_rule{generative_rule}.npy
#     """
#     import re
    
#     eta_match = re.search(r'eta([-+]?\d*\.\d+|\d+)', filename)
#     gamma_match = re.search(r'gamma([-+]?\d*\.\d+|\d+)', filename)
    
#     eta = float(eta_match.group(1)) if eta_match else None
#     gamma = float(gamma_match.group(1)) if gamma_match else None
    
#     return eta, gamma

# def get_eta_and_gamma_from_filename(filename):
#     """
#     Extract eta and gamma values from the network filename.
    
#     Expected filename format:
#     net_eta-{eta}_gamma{gamma}_rule{generative_rule}.npy
    
#     Examples:
#     - net_eta-6.748_gamma0.819_ruleMatchingIndex_id130.npy
#     - net_eta3.5_gamma1.0_ruleMatchingIndex_id017.npy
#     """
#     try:
#         # Split by 'eta' and take everything after it
#         eta_part = filename.split('eta')[1]
#         # Split by '_' to get just the eta value
#         eta_str = eta_part.split('_')[0]
#         eta = float(eta_str)
        
#         # Split by 'gamma' and take everything after it
#         gamma_part = filename.split('gamma')[1]
#         # Split by '_' to get just the gamma value
#         gamma_str = gamma_part.split('_')[0]
#         gamma = float(gamma_str)
        
#         return eta, gamma
#     except (IndexError, ValueError) as e:
#         print(f"Warning: Could not parse filename '{filename}': {e}")
#         return None, None


def sorted_listing_by_creation_time(directory):
    def get_creation_time(item):
        item_path = os.path.join(directory, item)
        return os.path.getctime(item_path)

    items = os.listdir(directory)
    sorted_items = sorted(items, key=get_creation_time)
    return sorted_items
