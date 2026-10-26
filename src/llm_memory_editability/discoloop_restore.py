"""Identity and capability checks for rebuilding the successful DiscoLoop."""

from __future__ import annotations

import math

PANELS = ("atomic_id", "atomic_ood", "train_ii", "test_ii", "test_io", "test_oi", "test_oo")
INITIAL_MODEL_SHA256 = "ba0af28431d36db482f2676c1e111a725cf51041c3dca9d006dd10d7a7c18518"
WORLD_SHA256 = "e31ec9959ce0a7b25465191a1d9ef8e83ec42163ed967fa56a60605f6ac1c3b4"
PANELS_SHA256 = "8655858c0de4646cfb5c7ea1779b5374ef28b492e6ac59bb6eda9e89c7936bf9"
PARAMETERS = 69_431_040
SPEC = dict(
    unique_layers=4,
    repeats=2,
    phi=7.2,
    weight_decay=0.1,
    initialization=791101,
    stream_seed=792101,
    lr=1e-4,
    warmup=2000,
    batch_size=512,
    steps=300000,
)
ARM = dict(
    kind="none",
    coefficient=0.0,
    scope="supervised",
    first_round_kv=False,
    discoloop=True,
    discoloop_alpha=1.0,
    discoloop_temperature=1.0,
    discoloop_eps=1e-6,
)
MINIMUM = dict.fromkeys(PANELS[:4], 0.98) | dict.fromkeys(PANELS[4:], 0.90)
REFERENCE = dict(test_io=0.9753333333333334, test_oi=0.993, test_oo=0.9565868263473054)


def validate_recipe(config, metadata=None, hf_config=None):
    """Reject a changed scientific recipe before any scientific model updates."""
    if config["batch"] != "discoloop-success-restore-v1":
        raise ValueError("Use the separate restoration batch")
    for key, value in SPEC.items():
        if config["spec"].get(key) != value:
            raise ValueError(f"Original recipe differs: {key}")
    if config["arms"] != {"discoloop": ARM} or config["dropout_seed"] != 20261012:
        raise ValueError("Preserve the original DiscoLoop forward and objective")
    gate = config["capability_gate"]
    if (
        gate["minimum_accuracy"] != MINIMUM
        or gate["reference_accuracy"] != REFERENCE
        or gate["maximum_reference_drop"] != 0.02
    ):
        raise ValueError("Capability thresholds must be frozen before training")
    if config["restore"]["expected_initial_model_sha256"] != INITIAL_MODEL_SHA256:
        raise ValueError("Initial tensors must match the successful historical recipe")
    if config["restore"]["expected_parameters"] != PARAMETERS:
        raise ValueError("The successful model has 69,431,040 trainable parameters")
    if metadata is not None:
        expected = dict(
            world=790101,
            entities=2000,
            relations=200,
            degree=20,
            atomic_id=38000,
            atomic_ood=2000,
            vocab_size=52463,
            world_sha256=WORLD_SHA256,
            panels_sha256=PANELS_SHA256,
        )
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError("Restore the original Wang graph and vocabulary")
    if hf_config is not None:
        expected = dict(
            n_embd=768,
            n_head=12,
            n_positions=1024,
            activation_function="gelu_new",
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
        )
        if any(hf_config.get(k) != v for k, v in expected.items()):
            raise ValueError("Preserve original GPT2 width, heads, normalization and dropout")


def foundation_gate(config, complete, audit, checkpoint_sha256):
    """The fixed endpoint must be independently replayed before any writing."""
    validate_recipe(config)
    identity = {
        "complete": complete.get("state") == "complete",
        "fixed_endpoint": complete.get("step") == SPEC["steps"],
        "full_budget": complete.get("budget") == SPEC["steps"],
        "science_endpoint": complete.get("engineering_only") is False,
        "parameter_count": complete.get("parameters") == PARAMETERS,
        "initial_tensors": complete.get("initial_model_sha256") == INITIAL_MODEL_SHA256,
        "checkpoint": complete.get("final_checkpoint_sha256") == checkpoint_sha256,
        "independent_reload": audit.get("passed") is True,
        "reload_endpoint": audit.get("step") == SPEC["steps"],
    }
    findings, metrics = {}, complete.get("endpoint", {}).get("metrics", {})
    for name in PANELS:
        value = metrics.get(name, {}).get("accuracy", float("nan"))
        audited = audit.get("findings", {}).get("panels", {}).get(name, {})
        minimum = max(MINIMUM[name], REFERENCE.get(name, 0) - 0.02)
        findings[name] = {
            "accuracy": value,
            "minimum_accuracy": minimum,
            "passed": math.isfinite(value)
            and value >= minimum
            and audited.get("passed") is True
            and audited.get("metrics", {}).get("accuracy") == value,
        }
    return {
        "passed": all(identity.values()) and all(row["passed"] for row in findings.values()),
        "step": SPEC["steps"],
        "checkpoint_sha256": checkpoint_sha256,
        "model_sha256": complete.get("model_sha256"),
        "initial_model_sha256": INITIAL_MODEL_SHA256,
        "metrics": metrics,
        "identity": identity,
        "capability": findings,
        "scope": "Original seven panels at fixed 300k endpoint, before any reader/writer "
        "training; no edited composition labels or outcomes used for selection",
    }
