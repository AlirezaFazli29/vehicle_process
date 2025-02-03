from torch.cuda import (
    is_available,
    get_device_name
)


def select_device():
    """
    Selects the appropriate device for computation based on GPU availability.

    This function checks if a CUDA-compatible GPU is available. 
    - If a GPU is detected, it prints a message indicating the GPU is selected.
    - If no GPU is found, it defaults to the CPU and prints a message accordingly.

    Returns:
        str: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    if is_available(): 
        print(
            f"{get_device_name()} have been located and selected"
        )
    else: 
        print(
            "No GPU cuda core found on this device. cpu is selected as network processor"
        )
    return 'cuda' if is_available() else 'cpu'