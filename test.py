import torch
from custom_op import custom_fill, custom_add

# check mps device
assert torch.backends.mps.is_available()

mps_device = torch.device("mps")
cpu_device = torch.device("cpu")

if __name__ == "__main__":

    print(custom_fill(torch.zeros(42, device=mps_device), 42))
    print(custom_add(torch.zeros(100, device=mps_device), torch.arange(100, device=mps_device, dtype=torch.float32)))

    print("OK")
