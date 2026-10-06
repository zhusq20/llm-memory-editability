"""External two-call reference using only self-generated bridge identities."""

from llm_memory_editability.capacity_controls import (
    decode_bridge,
    first_queries,
    second_queries,
)
from llm_memory_editability.capacity_scaling import pack


def serial_recall(model, world, spec, device, evaluate_rows, native_arrays):
    metrics, arrays = {}, {}
    for pool in ("II", "IO", "OI", "OO"):
        rows = world[pool]
        if not len(rows):
            continue
        first, _ = evaluate_rows(model, first_queries(rows), device, spec["evaluation_batch_size"])
        bridges, valid = decode_bridge(first)
        # The ground truth is accessible only to the separately named oracle.
        oracle = world["first"][rows[:, 1], rows[:, 2]]
        labels = pack(rows)[2]
        both = native_arrays[pool + "_both_atomic"]
        arrays[pool + "_bridge_predictions"] = first
        arrays[pool + "_bridge_valid"] = valid
        for method, selected in (
            ("serial", bridges),
            ("wrong_bridge", (bridges + 1) % spec["values_n"]),
            ("oracle_bridge", oracle),
        ):
            prediction, _ = evaluate_rows(
                model, second_queries(rows, selected), device, spec["evaluation_batch_size"]
            )
            ok = (prediction == labels).all(1)
            if method != "oracle_bridge":
                ok &= valid
            key = method + "_" + pool
            metrics[key] = {
                "n": len(rows),
                "accuracy": float(ok.mean()),
                "both_atomic_n": int(both.sum()),
                "both_atomic_coverage": float(both.mean()),
                "both_atomic_composition_accuracy": float(ok[both].mean()) if both.any() else None,
                "valid_bridge_fraction": float(valid.mean()),
                "external_decomposition": True,
                "oracle_information": method == "oracle_bridge",
                "generation_forward_calls_per_query": 6 if method == "oracle_bridge" else 12,
            }
            arrays[key + "_predictions"] = prediction
            arrays[key + "_correct"] = ok
    return metrics, arrays
