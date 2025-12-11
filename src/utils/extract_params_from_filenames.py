def get_eta_gamma_id_from_filename(filename):
    """
    Extract eta and gamma values from the network filename.
    
    Expected filename format:
    net_eta-{eta}_gamma{gamma}_rule{generative_rule}.npy
    
    Examples:
    - net_eta-6.748_gamma0.819_ruleMatchingIndex_id130.npy
    - net_eta3.5_gamma1.0_ruleMatchingIndex_id017.npy
    """
    try:
        # Split by 'eta' and take everything after it
        eta_part = filename.split('eta')[1]
        # Split by '_' to get just the eta value
        eta_str = eta_part.split('_')[0]
        eta = float(eta_str)
        
        # Split by 'gamma' and take everything after it
        gamma_part = filename.split('gamma')[1]
        # Split by '_' to get just the gamma value
        gamma_str = gamma_part.split('_')[0]
        gamma = float(gamma_str)
        
        # Try to extract id (defaults to 0 if not present)
        id_match = filename.split('_id')[1].split('.npy')[0]
        net_id = int(id_match) # id_match.group(1)) if id_match else 0
        
        return eta, gamma, net_id

    except (IndexError, ValueError) as e:
        print(f"Warning: Could not parse filename '{filename}': {e}")
        return None, None, None
 