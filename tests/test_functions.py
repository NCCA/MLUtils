import zipfile

import pytest
import torch

from mlutils import functions
from mlutils.functions import accuracy, download, get_batch_accuracy, in_lab, unzip_file


@pytest.mark.parametrize("hostname", ["pg01", "pg123.ncca.bournemouth.ac.uk", "w10234"])
def test_in_lab_is_true_for_lab_hostnames(monkeypatch, hostname):
    monkeypatch.setattr(functions.platform, "node", lambda: hostname)

    assert in_lab() is True


@pytest.mark.parametrize("hostname", ["my-macbook", "pg", "", "desktop"])
def test_in_lab_is_false_for_other_hostnames(monkeypatch, hostname):
    monkeypatch.setattr(functions.platform, "node", lambda: hostname)

    assert in_lab() is False


class FakeResponse:
    def __init__(self, payload: bytes, with_length: bool = True):
        self.payload = payload
        self.headers = {"content-length": str(len(payload))} if with_length else {}

    def iter_content(self, chunk_size):
        for start in range(0, len(self.payload), chunk_size):
            yield self.payload[start : start + chunk_size]


@pytest.mark.parametrize("with_length", [True, False])
def test_download_writes_the_response_body_to_disk(monkeypatch, tmp_path, with_length):
    payload = bytes(range(256)) * 10  # more than one 1024 byte chunk
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(payload, with_length)

    monkeypatch.setattr(functions.requests, "get", fake_get)
    target = tmp_path / "data.bin"

    download("https://example.com/data.bin", str(target))

    assert calls == ["https://example.com/data.bin"]
    assert target.read_bytes() == payload


def test_unzip_file_extracts_every_file(tmp_path):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.txt", "hello")
        zf.writestr("sub/b.txt", "world")
    dest = tmp_path / "out"

    unzip_file(archive, dest)

    assert (dest / "a.txt").read_text() == "hello"
    assert (dest / "sub" / "b.txt").read_text() == "world"


def test_unzip_file_raises_for_missing_archive(tmp_path):
    with pytest.raises(FileNotFoundError):
        unzip_file(tmp_path / "missing.zip", tmp_path)


def test_get_batch_accuracy_counts_argmax_matches():
    output = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    y = torch.tensor([0, 1, 1, 1])

    assert get_batch_accuracy(output, y, N=4) == pytest.approx(0.75)


def test_get_batch_accuracy_divides_by_n_not_batch_size():
    output = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 1])

    assert get_batch_accuracy(output, y, N=4) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        ([1, 2, 3, 4], [1, 2, 3, 4], 100.0),
        ([1, 2, 3, 4], [0, 0, 0, 0], 0.0),
        ([1, 2, 3, 4], [1, 2, 0, 0], 50.0),
    ],
)
def test_accuracy_returns_a_percentage(y_true, y_pred, expected):
    assert accuracy(torch.tensor(y_true), torch.tensor(y_pred)) == pytest.approx(expected)
