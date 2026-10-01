"""Read-only singular spectra of four frozen down-projection matrices."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    start = time.monotonic()
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "torch": torch.__version__,
        "device": "cpu",
        "calculation_dtype": "float64",
        "method": "torch.linalg.svdvals on actual checkpoint down weights",
        "rank_definition": "count(singular_value > rtol * maximum_singular_value), atol=0",
        "data_or_labels_used": False,
        "parameters_updated": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matrices": [],
    }
    models = (
        ("qwen3", "data/hebbian-learning-v1/source/qwen3-0.6b-base", "model"),
        (
            "qwen35",
            "data/architecture-bridge-v1/models/qwen3.5-0.8b-base",
            "model.language_model",
        ),
    )
    for model_name, model_path, key_prefix in models:
        directory = root / model_path
        index_path = directory / "model.safetensors.index.json"
        mapping = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else {}
        for layer in (6, 14):
            key = f"{key_prefix}.layers.{layer}.mlp.down_proj.weight"
            shard = (
                directory / mapping[key]
                if key in mapping
                else next(directory.glob("*.safetensors"))
            )
            with safe_open(shard, framework="pt", device="cpu") as source:
                original = source.get_tensor(key)
                matrix = original.to(dtype=torch.float64)
            spectrum = torch.linalg.svdvals(matrix)
            maximum, minimum = spectrum.max().item(), spectrum.min().item()
            result["matrices"].append(
                {
                    "model": model_name,
                    "layer_index": layer,
                    "checkpoint_path": str(shard),
                    "checkpoint_key": key,
                    "checkpoint_dtype": str(original.dtype),
                    "shape": list(matrix.shape),
                    "minimum_singular_value": minimum,
                    "maximum_singular_value": maximum,
                    "condition_number": maximum / minimum,
                    "numerical_rank_rtol_1e_6": int((spectrum > maximum * 1e-6).sum()),
                    "numerical_rank_rtol_1e_5": int((spectrum > maximum * 1e-5).sum()),
                    "full_row_rank_at_both_tolerances": bool(
                        (spectrum > maximum * 1e-5).sum() == matrix.shape[0]
                    ),
                    "singular_values_descending": spectrum.tolist(),
                }
            )
    result["wall_seconds"] = time.monotonic() - start
    output = root / "docs/development-artifacts/architecture-bridge-v1/parameter-structure.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            [
                {key: value for key, value in row.items() if key != "singular_values_descending"}
                for row in result["matrices"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
