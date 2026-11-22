# MLUtils

This is a standalone version of the MLUtils library I use in my teaching code. It's useful for teaching and experimentation, and it saves copying the same helper functions into every notebook.

The training helpers are the same `train_epoch` / `evaluate` functions I build up in the Free Spoken Digits and MNIST notebooks. Once the loop has been explained they move here, so the later notebooks can concentrate on the model and the data.

## Installation

It is expected you will use [uv](https://docs.astral.sh/uv/). The package isn't on PyPI, so add it straight from [GitHub](https://github.com/NCCA/MLUtils):

```bash
uv add git+https://github.com/NCCA/MLUtils
```

Everything is exported from the top level of the package, so you can import any of the functions below like this:

```python
from mlutils import get_device, train_epoch, evaluate
```

## Contents

- [General utilities](#general-utilities)
  - [in_lab](#in_lab)
  - [download](#download)
  - [unzip_file](#unzip_file)
  - [get_batch_accuracy](#get_batch_accuracy)
  - [accuracy](#accuracy)
- [PyTorch utilities](#pytorch-utilities)
  - [get_device](#get_device)
- [Training and evaluation](#training-and-evaluation)
  - [argmax_predict](#argmax_predict)
  - [binary_predict](#binary_predict)
  - [Metrics](#metrics)
  - [train_epoch](#train_epoch)
  - [evaluate](#evaluate)
  - [copy_weights](#copy_weights)
- [Developer notes](#developer-notes)

## General utilities

These live in `mlutils/functions.py`.

### in_lab

```python
in_lab() -> bool
```

Returns `True` if the code is running on one of the NCCA lab machines. It checks the hostname against the lab naming scheme (`pg` or `w` followed by a number) using a regular expression. I use this to change paths or settings depending on whether students are in the labs or on their own machines.

```python
from mlutils import in_lab

DATA_DIR = "/transfer/data" if in_lab() else "./data"
```

### download

```python
download(url: str, fname: str) -> None
```

Downloads the file at `url` and saves it as `fname`, showing a [tqdm](https://tqdm.github.io/) progress bar as it goes. The file is streamed in 1 KiB chunks, so large datasets don't have to fit in memory.

Note that SSL verification is turned off (`verify=False`), so only use this with sources you trust.

```python
from mlutils import download

download("https://example.com/dataset.zip", "dataset.zip")
```

### unzip_file

```python
unzip_file(zip_file: Path, dest_dir: Path) -> None
```

Extracts everything in `zip_file` into `dest_dir`. This is usually paired with `download`.

```python
from pathlib import Path
from mlutils import download, unzip_file

download("https://example.com/dataset.zip", "dataset.zip")
unzip_file(Path("dataset.zip"), Path("data"))
```

### get_batch_accuracy

```python
get_batch_accuracy(output: torch.Tensor, y: torch.Tensor, N: int) -> float
```

Works out the accuracy of a batch of predictions. `output` is the raw model output (one score per class), the predicted class is taken with `argmax` over dimension 1 and compared with the labels in `y`. The number of correct predictions is divided by `N`, so the result is in the range 0-1.

```python
import torch
from mlutils import get_batch_accuracy

output = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
labels = torch.tensor([1, 0, 0])
print(get_batch_accuracy(output, labels, len(labels)))  # 0.666...
```

### accuracy

```python
accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float
```

Compares the true labels with the predicted labels and returns the accuracy as a percentage (0-100). Unlike `get_batch_accuracy` this expects labels, not raw model output, so do the `argmax` (or threshold) first.

```python
import torch
from mlutils import accuracy

y_true = torch.tensor([0, 1, 1, 0])
y_pred = torch.tensor([0, 1, 0, 0])
print(accuracy(y_true, y_pred))  # 75.0
```

## PyTorch utilities

These live in `mlutils/torch_utils.py`.

### get_device

```python
get_device(show_info: bool = False) -> torch.device
```

Returns the best device available to your PyTorch install. It checks for CUDA (or ROCm) first, then Apple Metal (MPS), and falls back to the CPU. If `show_info` is `True` it also prints some details about the device, such as the GPU name and how much memory is free.

```python
from mlutils import get_device

device = get_device(show_info=True)
model = model.to(device)
```

On my Mac this prints the following

```
Device: mps
Backend: Apple Metal / MPS
GPU: Apple M4 Max
GPU cores: 32
Tensor memory: 0.00 GiB
Metal driver memory: 0.00 GiB
Recommended working set: 27.00 GiB
```

## Training and evaluation

These live in `mlutils/training.py`. The examples in this section use the following small setup, a linear classifier on some random data.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from mlutils import get_device

device = get_device()
inputs = torch.randn(100, 2)
targets = (inputs[:, 0] > 0).long()
loader = DataLoader(TensorDataset(inputs, targets), batch_size=10, shuffle=True)

model = nn.Linear(2, 2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
```

### argmax_predict

```python
argmax_predict(logits: torch.Tensor) -> torch.Tensor
```

Turns raw model output into class labels by picking the class with the highest score in dimension 1. This is the default way `Metrics`, `train_epoch` and `evaluate` decide what the model predicted, and is the one to use for a multi-class model (for example with `nn.CrossEntropyLoss`).

```python
import torch
from mlutils import argmax_predict

logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 0.2, 3.0]])
print(argmax_predict(logits))  # tensor([0, 2])
```

### binary_predict

```python
binary_predict(logits: torch.Tensor) -> torch.Tensor
```

For a model with a single output logit, anything above zero is class 1 (as `sigmoid(0)` is 0.5). The result is a float tensor so it compares directly with the float targets `nn.BCEWithLogitsLoss` needs. Pass this as the `predict` argument when training a binary classifier.

```python
import torch
from mlutils import binary_predict

logits = torch.tensor([-1.2, 0.3, 2.5])
print(binary_predict(logits))  # tensor([0., 1., 1.])
```

### Metrics

```python
Metrics(predict: Callable[[torch.Tensor], torch.Tensor] = argmax_predict)
```

Keeps running totals of loss and accuracy over one pass of the data. Call `update` after each batch, then `result` at the end to get the mean loss and accuracy (0-1).

Each batch is weighted by the number of predictions it holds (`targets.numel()`), so a short final batch doesn't count as much as a full one. For a classifier this is the batch size, for a sequence model it is batch size * sequence length.

`train_epoch` and `evaluate` use this internally, you only need it directly if you are writing your own loop.

```python
from mlutils import Metrics

metrics = Metrics()
for x, y in loader:
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = loss_fn(logits, y)
    metrics.update(loss, logits, y)

mean_loss, acc = metrics.result()
print(f"loss {mean_loss:.4f} accuracy {acc:.2%}")
```

### train_epoch

```python
train_epoch(
    model, loader, loss_fn, optimiser, device,
    transform=None, predict=argmax_predict,
) -> tuple[float, float]
```

Does one pass over the training data, updating the weights after each batch, and returns the mean loss and accuracy for the epoch. The model should already be on `device`, each batch is moved there for you.

`transform` is applied to the inputs before the forward pass, which is useful for data augmentation. `predict` is used for the accuracy, see [Metrics](#metrics). The `loader` can be anything that yields `(inputs, targets)` batches, so in a [marimo](https://marimo.io/) notebook you can wrap it in `mo.status.progress_bar` to see progress for each batch.

```python
from mlutils import train_epoch

for epoch in range(10):
    loss, acc = train_epoch(model, loader, loss_fn, optimiser, device)
    print(f"epoch {epoch} loss {loss:.4f} accuracy {acc:.2%}")
```

For a binary classifier with a single output use `binary_predict` and float targets

```python
from mlutils import binary_predict, train_epoch

model = nn.Linear(2, 1).to(device)
loss_fn = nn.BCEWithLogitsLoss()
# targets need to be float and the same shape as the output, e.g. targets.float().unsqueeze(1)
loss, acc = train_epoch(model, loader, loss_fn, optimiser, device, predict=binary_predict)
```

### evaluate

```python
evaluate(model, loader, loss_fn, device, predict=argmax_predict) -> tuple[float, float]
```

Does one pass over the data without gradients or weight updates (it runs under `torch.inference_mode()` and puts the model into `eval` mode), and returns the mean loss and accuracy. Use this for the validation and test sets.

```python
from mlutils import evaluate, train_epoch

for epoch in range(10):
    train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimiser, device)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
    print(f"epoch {epoch} train {train_acc:.2%} val {val_acc:.2%}")
```

### copy_weights

```python
copy_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]
```

Takes a CPU copy of the model weights, which I use to keep the weights from the best epoch. `state_dict()` returns references to the live tensors, so without the clone the "best" weights would keep changing as training carries on. Load them back with `model.load_state_dict`.

```python
from mlutils import copy_weights, evaluate, train_epoch

best_acc, best_weights = 0.0, None
for epoch in range(20):
    train_epoch(model, train_loader, loss_fn, optimiser, device)
    _, val_acc = evaluate(model, val_loader, loss_fn, device)
    if val_acc > best_acc:
        best_acc, best_weights = val_acc, copy_weights(model)

model.load_state_dict(best_weights)
```

## Developer notes

To run the tests

```bash
uv run pytest
```

This README.md is automatically generated from the source code using CLAUDE.
