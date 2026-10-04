import asyncio
from io import BytesIO

import pytest
import torch
from fastapi import BackgroundTasks
from starlette.datastructures import UploadFile

from patchguard.evaluator import app as evaluator
from patchguard.model import BasicCNN
from patchguard.predict import patchguard_batch_predict
from range_device import select_device


def test_auto_falls_back_to_cpu_and_explicit_cuda_fails_without_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    assert select_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        select_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_patchguard_upload_validation_and_scoring_work_on_cuda(monkeypatch):
    select_device("cuda")
    torch.manual_seed(31)
    model = BasicCNN(16, 1).eval()
    images = torch.rand(5, 1, 28, 28)
    labels = patchguard_batch_predict(model, images)
    hidden = {"clean_images": images.cuda(), "clean_labels": labels.cuda(),
              "adversarial_images": images.cuda(), "adversarial_labels": labels.cuda()}
    for key, value in {"reference_model": BasicCNN(16, 1), "hidden": hidden,
                       "clean_threshold": 1.0, "robust_threshold": 1.0,
                       "flag": "flag{gpu-test}"}.items():
        monkeypatch.setattr(evaluator.app.state, key, value, raising=False)
    data = BytesIO()
    torch.save(model.state_dict(), data)
    upload = UploadFile(filename="model.pt", file=BytesIO(data.getvalue()))
    result = asyncio.run(evaluator.score(BackgroundTasks(), upload, None, None))
    assert result["success"] is True
    assert result["clean_accuracy"] == result["robust_accuracy"] == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_patchguard_training_and_adversarial_gradients_run_on_cuda():
    device = select_device("cuda")
    model = BasicCNN(28, 1).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    images = torch.rand(4, 1, 28, 28, device=device, requires_grad=True)
    labels = torch.arange(4, device=device)
    loss = torch.nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    assert images.grad.is_cuda and torch.isfinite(images.grad).all()
    optimizer.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_xray_local_attacks_predictions_and_smoothing_use_cuda(monkeypatch, tmp_path):
    import numpy as np
    from PIL import Image
    from participant.workspace import red_blue_arena as arena

    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones((), device="cuda"))

        def forward(self, images):
            assert images.is_cuda
            score = images.mean(dim=(1, 2, 3)) * self.scale
            return torch.stack((score, -score), dim=1)

    source = tmp_path / "source.png"
    Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8)).save(source)
    model = TinyClassifier().eval()
    monkeypatch.setattr(arena, "load_standard_model", lambda: model)
    monkeypatch.setattr(arena, "download_source", lambda source_id: source)
    monkeypatch.setattr(arena, "get_sources", lambda: {
        "sources": [{"source_id": "1", "class_index": 0}],
        "class_names": ["class-a", "class-b"],
    })
    assert arena.predict_image(source)["class_index"] == 0
    for method in ("FGSM", "SPARSE_FGSM", "PGD"):
        output = arena.make_adversarial_candidate("1", method=method, iterations=2,
                                                  destination=tmp_path / (method + ".png"))
        changed = np.asarray(Image.open(output), dtype=np.int16)
        assert np.max(np.abs(changed - 128)) <= 5
    result = arena.local_randomized_smoothing(source, num_samples=10, batch_size=3)
    assert sum(result["votes"]) == 10
