import torch
from nbr.utils.device import get_device


def test_get_device():
    device = get_device()
    assert isinstance(device, torch.device)
    # Should be one of cuda, mps, or cpu
    assert device.type in ["cuda", "mps", "cpu"]
    print(f"Selected device: {device}")


if __name__ == "__main__":
    test_get_device()
