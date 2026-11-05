"""PyTorch models fall back to CPU when the torch build has no CUDA.

The ML image ships the CPU-only torch wheel while ML_SERVICE_ML_DEVICE=cuda
drives ONNX Runtime onto the GPU. Before this, the blur model was loaded with
map_location="cuda" and failed, which left the capture-quality check without
its model and made every gated checkpoint request return 503.
"""

import pytest

torch = pytest.importorskip("torch")
app_module = pytest.importorskip("zepiris.ml_inference.app")


def test_cuda_requested_without_torch_cuda_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert app_module._torch_device("cuda") == "cpu"


def test_cuda_requested_with_torch_cuda_keeps_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert app_module._torch_device("cuda") == "cuda"


def test_cpu_stays_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert app_module._torch_device("cpu") == "cpu"
