import torch
from custom_fill import custom_fill

# check mps device
assert torch.backends.mps.is_available()

mps_device = torch.device("mps")
cpu_device = torch.device("cpu")

if __name__ == "__main__":
    for i in range(1000):
        input_mps = torch.zeros(42, 42, 2, device=mps_device)
        input_cpu = torch.zeros(42, 42, 2, device=cpu_device)

        custom_result = custom_fill(input_mps, 42)
        input_cpu.fill_(42)

        assert torch.equal(custom_result.detach().cpu(), input_cpu)
    print("OK")
