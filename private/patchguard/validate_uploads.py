from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import requests
import torch

from patchguard.model import BasicCNN


def submit(url: str, payload: bytes, *, user_id: str, username: str) -> dict:
    response = requests.post(
        f"{url.rstrip('/')}/score",
        files={"model": ("patchguard_model.pt", BytesIO(payload), "application/octet-stream")},
        data={"user_id": user_id, "username": username},
        timeout=180,
    )
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text[:200]}
    # Never print the actual flag, even when this script solves the challenge.
    safe_body = {key: value for key, value in body.items() if key != "flag"}
    safe_body["flag_returned"] = "flag" in body
    return {"http_status": response.status_code, **safe_body}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://patchguard-target:8000")
    parser.add_argument("--models-dir", default="/opt/artifacts/patchguard")
    parser.add_argument("--user-id", default="anonymous")
    parser.add_argument("--username", default="anonymous")
    args = parser.parse_args()
    root = Path(args.models_dir)
    output = {}
    for name in ("random_model.pt", "weak_model.pt", "reference_model.pt"):
        path = root / name
        result = submit(
            args.url,
            path.read_bytes(),
            user_id=args.user_id,
            username=args.username,
        )
        output[name] = result
        print(json.dumps({name: result}, sort_keys=True))

    reference_bytes = (root / "reference_model.pt").read_bytes()
    repeated = [
        submit(args.url, reference_bytes, user_id=args.user_id, username=args.username)
        for _ in range(2)
    ]
    output["reference_repeatability"] = {
        "clean_accuracy": [
            output["reference_model.pt"].get("clean_accuracy"),
            *(item.get("clean_accuracy") for item in repeated),
        ],
        "robust_accuracy": [
            output["reference_model.pt"].get("robust_accuracy"),
            *(item.get("robust_accuracy") for item in repeated),
        ],
        "success": [
            output["reference_model.pt"].get("success"),
            *(item.get("success") for item in repeated),
        ],
    }

    baseline = torch.load(root / "baseline_model.pt", map_location="cpu", weights_only=True)
    invalid: dict[str, bytes] = {"random_bytes": b"not a PyTorch state_dict"}
    invalid["empty"] = b""
    invalid["wrong_architecture"] = (root / "baseline_model.pt").read_bytes()
    invalid["missing_key"] = _save({key: value for key, value in list(baseline.items())[1:]})
    invalid["extra_key"] = _save({**baseline, "unexpected": torch.zeros(1)})
    first_key = next(iter(baseline))
    invalid["wrong_shape"] = _save({**baseline, first_key: torch.zeros(1)})
    invalid["nan_weight"] = _save({**baseline, first_key: torch.full_like(baseline[first_key], float("nan"))})
    invalid["inf_weight"] = _save({**baseline, first_key: torch.full_like(baseline[first_key], float("inf"))})
    invalid_results = {}
    for name, payload in invalid.items():
        if not payload:
            # Multipart uses an empty file body; the endpoint rejects it before decoding.
            pass
        invalid_results[name] = submit(
            args.url,
            payload,
            user_id=args.user_id,
            username=args.username,
        )
    output["invalid_uploads"] = invalid_results
    print(json.dumps(output["reference_repeatability"], sort_keys=True))
    print(json.dumps({"invalid_uploads": invalid_results}, sort_keys=True))


def _save(value: dict) -> bytes:
    stream = BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


if __name__ == "__main__":
    main()
