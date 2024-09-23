import os
import torch


# Set to True to use CUDA if available
USE_CUDA = False

def get_device():
    """
    Function to return the appropriate device based on the global USE_CUDA flag.
    
    Returns:
    str: Device type ("cuda" or "cpu").
    """
    # Check if CUDA is available and return the device type
    return "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"

