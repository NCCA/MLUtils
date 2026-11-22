#!/usr/bin/env python

"""
Module to provide utility functions for working with PyTorch.
"""

try:
    import torch
except ImportError:
    print("torch not found")
    print("refer to this URL for install instructions https://pytorch.org/get-started/locally/ ")


import torch


def print_cuda_info(device: torch.device) -> None:
    backend = "ROCm" if torch.version.hip else "CUDA"
    properties = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)

    print(f"Device: {device}")
    print(f"Backend: {backend}")
    print(f"GPU: {properties.name}")
    print(f"Total memory: {total / 2**30:.2f} GiB")
    print(f"Free memory: {free / 2**30:.2f} GiB")


def print_mps_info() -> None:
    print("Device: mps")
    print("Backend: Apple Metal / MPS")

    for method, label in (
        ("get_name", "GPU"),
        ("get_core_count", "GPU cores"),
    ):
        getter = getattr(torch.backends.mps, method, None)
        if getter is not None:
            print(f"{label}: {getter()}")

    torch.mps.synchronize()

    for label, method in (
        ("Tensor memory", "current_allocated_memory"),
        ("Metal driver memory", "driver_allocated_memory"),
        ("Recommended working set", "recommended_max_memory"),
    ):
        getter = getattr(torch.mps, method, None)
        if getter is not None:
            print(f"{label}: {getter() / 2**30:.2f} GiB")


def print_cpu_info() -> None:
    print("Device: cpu")
    print("No supported GPU available to this PyTorch installation")


def get_device(show_info: bool = False) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        if show_info:
            print_cuda_info(device)

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if show_info:
            print_mps_info()

    else:
        device = torch.device("cpu")
        if show_info:
            print_cpu_info()

    return device


if __name__ == "__main__":
    print(f"found device {get_device()}")
