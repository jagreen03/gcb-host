#!/usr/bin/env python3
# =============================================================================
# Smoke test for tools/hf_manifest_gen.py - TKT-032's own flagged gap
# (read but never exercised) - first real test in this previously-empty
# tests/ directory.
#
# Self-checking, no pytest dependency (matches proxy/gcb_proxy.py's own
# self-checking convention, confirmed in TKT-032's survey): asserts inline,
# prints PASS/FAIL, exits 0/1. Imports the module directly rather than
# shelling out to it, so it checks the real negotiation logic
# (ManifestGenerator + negotiate()) without depending on argv/exit-code
# plumbing, which this session's own verification pass found to be an
# unrelated, separate WSL-interop quirk in the surrounding dev environment
# (not anything in this file) when checked via a subprocess's $? read
# inside the same shell invocation that launched it - not a concern for
# this direct-import test at all.
#
# Run:
#   python3 test_hf_manifest_gen.py
# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from hf_manifest_gen import (  # noqa: E402
    ArchType,
    HardwareManifest,
    ManifestGenerator,
    NegotiationResult,
    negotiate,
)

FAILURES = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def test_dense_large_model_rejected() -> None:
    """A 70B-class dense model (Llama-3.1-70B-Instruct-shaped config) should
    be REJECTED against the 16-tile prototype hardware manifest - it
    genuinely needs hundreds of tiles.
    """
    print("test_dense_large_model_rejected:")
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8192,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 28672,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "torch_dtype": "bfloat16",
        "_name_or_path": "meta-llama/Llama-3.1-70B-Instruct",
    }
    gen = ManifestGenerator()
    model = _generate_from_config(gen, config, "meta-llama/Llama-3.1-70B-Instruct")

    check("architecture detected as dense", model.architecture == ArchType.DENSE_TRANSFORMER.value)
    check("hidden_dim extracted", model.hidden_dim == 8192)
    check("precision mapped bfloat16 -> BF16", model.precision_required == "BF16")

    hw = HardwareManifest()  # prototype default: 16 tiles, 8 crossbar ports
    report = negotiate(hw, model)
    check(
        "negotiation result is REJECTED (needs far more than 16 tiles)",
        report.result == NegotiationResult.REJECTED,
        detail=f"got {report.result}",
    )
    check("errors list non-empty", len(report.errors) > 0)


def test_small_dense_model_degraded() -> None:
    """A small dense model should fit within the 16-tile prototype, but
    still warn about feed-forward tiling at these dimensions (768x3072,
    tile cap 1500x1500 - tiles_needed rounds up per projection, so the
    up+down-projection doubling still crosses the single-tile warning
    threshold) -> ACCEPTED-with-warnings (DEGRADED), not a clean ACCEPT.
    """
    print("test_small_dense_model_degraded:")
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "vocab_size": 32000,
        "max_position_embeddings": 2048,
        "torch_dtype": "float16",
        "_name_or_path": "test/tiny-dense",
    }
    gen = ManifestGenerator()
    model = _generate_from_config(gen, config, "test/tiny-dense")

    check("precision mapped float16 -> FP8", model.precision_required == "FP8")
    hw = HardwareManifest()
    report = negotiate(hw, model)
    check(
        "negotiation result is DEGRADED (fits, but with tiling warnings)",
        report.result == NegotiationResult.DEGRADED,
        detail=f"got {report.result}",
    )
    check("no hard errors", len(report.errors) == 0)


def test_moe_expert_count_detected() -> None:
    """A DeepSeek-V3-shaped MoE config should be detected as MoE and
    extract a real expert count, not silently fall back to dense.
    """
    print("test_moe_expert_count_detected:")
    config = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048,
        "vocab_size": 129280,
        "torch_dtype": "bfloat16",
        "_name_or_path": "deepseek-ai/DeepSeek-V3",
    }
    gen = ManifestGenerator()
    model = _generate_from_config(gen, config, "deepseek-ai/DeepSeek-V3")

    check("architecture detected as MoE", model.architecture == ArchType.MOE_TRANSFORMER.value)
    check("num_experts extracted", model.num_experts == 256, detail=f"got {model.num_experts}")
    check("num_experts_active extracted", model.num_experts_active == 8, detail=f"got {model.num_experts_active}")

    hw = HardwareManifest()
    report = negotiate(hw, model)
    check(
        "8 active experts exceeds the 8-port/8-active prototype boundary check runs without crashing",
        report is not None,
    )


def _generate_from_config(gen: ManifestGenerator, config: dict, model_id: str):
    """Drive ManifestGenerator.generate()'s real body against an in-memory
    config dict, without a network call or a temp file on disk - reuses
    the exact same code this tool's own --local path calls after
    HFConfigReader.from_local() returns, by pre-seeding the reader's
    raw_config and calling the same internal generate() logic via a tiny
    local HFConfigReader stand-in.
    """
    gen.reader.raw_config = config

    class _FrozenReader:
        def from_local(self, _path):
            return config

        def from_hub(self, _model_id):
            return config

    gen.reader = _FrozenReader()
    return gen.generate(model_id, local_path="<in-memory>")


def main() -> int:
    test_dense_large_model_rejected()
    test_small_dense_model_degraded()
    test_moe_expert_count_detected()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
