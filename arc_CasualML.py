"""
Simulates an e-commerce clickstream session log with embedded treatment and confounding variables.
"""

import numpy as np
import pandas as pd

def generate_synthetic_clickstream(n_samples=50000, random_state=42):
    np.random.seed(random_state)
    
    scroll_velocity = np.random.exponential(scale=2.0, size=n_samples)
    dwell_time_sec = np.random.gamma(shape=2.0, scale=15.0, size=n_samples)
    cart_item_count = np.random.poisson(lam=1.5, size=n_samples) + 1
    historical_spend = np.random.exponential(scale=50.0, size=n_samples)
    session_depth = np.random.randint(1, 10, size=n_samples)
    
    # Propensity to receive treatment (e.g., automated nudge email)
    propensity = 1 / (1 + np.exp(-(0.1 * scroll_velocity - 0.05 * dwell_time_sec)))
    is_targeted = np.random.binomial(1, propensity)
    
    # Baseline conversion outcome influenced by features and treatment (CATE effect)
    base_conversion = 0.05 + 0.02 * cart_item_count + 0.0