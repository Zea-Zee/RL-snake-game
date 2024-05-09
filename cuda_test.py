import torch

def print_cuda_devices():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s):")
        for i in range(device_count):
            device = torch.cuda.get_device_name(i)
            print(f"Device {i}: {device}")
    else:
        print("CUDA is not available on this system.")

print_cuda_devices()
