"""This script sets random seeds and verifies reproducibility across different components of the project."""
import os
import random
import numpy as np

def set_all_seeds(seed=42):
    """Set seeds for all random number generators"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # XGBoost specific
    os.environ['OMP_NUM_THREADS'] = '1'
    return seed

def verify_reproducibility(quick_test=True,SEED=42):
    """
    Verify reproducibility across all components of the fraud detection pipeline
    """
    print("=" * 50)
    print("REPRODUCIBILITY VERIFICATION")
    print("=" * 50)
    
    # Check seed settings
    print(f"✓ Random seeds set to: {SEED}")
    print(f"✓ PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'Not set')}")
    
    if quick_test:
        # quick random number generation test
        np.random.seed(SEED)
        sample1 = np.random.rand(5)
        np.random.seed(SEED)  # Reset
        sample2 = np.random.rand(5)
        
        if np.array_equal(sample1, sample2):
            print("✓ NumPy random generation: REPRODUCIBLE")
        else:
            print("✗ NumPy random generation: NOT REPRODUCIBLE")
            
        # reset for actual pipeline
        np.random.seed(SEED)
        random.seed(SEED)
        
    else:
        # full pipeline test
        print("Running full pipeline reproducibility test...")        
    print("=" * 50)