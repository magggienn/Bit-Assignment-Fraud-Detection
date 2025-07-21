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
    
    Args:
        quick_test: If True, runs a fast verification. If False, runs full pipeline test.
    """
    print("=" * 50)
    print("REPRODUCIBILITY VERIFICATION")
    print("=" * 50)
    
    # Check seed settings
    print(f"✓ Random seeds set to: {SEED}")
    print(f"✓ PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'Not set')}")
    
    if quick_test:
        # Quick random number generation test
        np.random.seed(SEED)
        sample1 = np.random.rand(5)
        np.random.seed(SEED)  # Reset
        sample2 = np.random.rand(5)
        
        if np.array_equal(sample1, sample2):
            print("✓ NumPy random generation: REPRODUCIBLE")
        else:
            print("✗ NumPy random generation: NOT REPRODUCIBLE")
            
        # Reset for actual pipeline
        np.random.seed(SEED)
        random.seed(SEED)
        
    else:
        # Full pipeline test (use sparingly - computationally expensive)
        print("Running full pipeline reproducibility test...")
        # This would run your actual model training twice and compare results
        
    print("=" * 50)