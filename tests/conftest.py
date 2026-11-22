import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture(autouse=True)
def fixed_seed():
    # keeps the random data and weight init the same on every run
    torch.manual_seed(0)


@pytest.fixture
def cpu():
    return torch.device("cpu")


@pytest.fixture
def separable_loader():
    # two well separated blobs, so a linear model can learn them in a few epochs
    class0 = torch.randn(32, 2) - 3.0
    class1 = torch.randn(32, 2) + 3.0
    inputs = torch.cat([class0, class1])
    targets = torch.cat([torch.zeros(32, dtype=torch.long), torch.ones(32, dtype=torch.long)])
    return DataLoader(TensorDataset(inputs, targets), batch_size=10, shuffle=True)


@pytest.fixture
def model():
    return torch.nn.Linear(2, 2)
