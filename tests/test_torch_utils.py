import pytest
import torch

from mlutils import get_device, torch_utils


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def test_get_device_falls_back_to_cpu(no_gpu):
    assert get_device() == torch.device("cpu")


def test_get_device_prints_cpu_info_when_asked(no_gpu, capsys):
    get_device(show_info=True)

    out = capsys.readouterr().out
    assert "Device: cpu" in out
    assert "No supported GPU" in out


def test_get_device_is_silent_by_default(no_gpu, capsys):
    get_device()

    assert capsys.readouterr().out == ""


def test_get_device_prefers_mps_when_cuda_is_missing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert get_device() == torch.device("mps")


def test_get_device_prefers_cuda_over_everything(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert get_device() == torch.device("cuda", 1)


def test_get_device_shows_cuda_info_when_asked(monkeypatch):
    shown = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch_utils, "print_cuda_info", shown.append)

    device = get_device(show_info=True)

    assert shown == [device]


def test_get_device_shows_mps_info_when_asked(monkeypatch):
    shown = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch_utils, "print_mps_info", lambda: shown.append("mps"))

    get_device(show_info=True)

    assert shown == ["mps"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_print_mps_info_reports_the_backend(capsys):
    torch_utils.print_mps_info()

    assert "Apple Metal / MPS" in capsys.readouterr().out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA or ROCm GPU")
def test_print_cuda_info_reports_the_gpu(capsys):
    torch_utils.print_cuda_info(torch.device("cuda", 0))

    out = capsys.readouterr().out
    assert "GPU:" in out
    assert "Total memory:" in out
