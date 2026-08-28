#!/usr/bin/env python3
# =============================================================================
# GreenCoyote-Bay — HuggingFace Model Manifest Generator
# Project: ODIN_Bridge / Cascaded Optical Reduction Fabric
# Author:  John A. Green
# Date:    June 2026
# =============================================================================
# Reads a HuggingFace model config.json and produces a GreenCoyote
# model manifest describing what hardware resources the model requires.
#
# Usage:
#   python3 hf_manifest_gen.py deepseek-ai/DeepSeek-R1
#   python3 hf_manifest_gen.py meta-llama/Llama-3.1-70B-Instruct
#   python3 hf_manifest_gen.py --local /path/to/model/config.json
#
# Output:
#   model_manifest.json — GreenCoyote hardware requirement description
#   negotiation result  — compatible / incompatible with current hardware
# =============================================================================

import json
import sys
import math
import argparse
import requests
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

# =============================================================================
# Constants
# =============================================================================

HF_CONFIG_URL = "https://huggingface.co/{model_id}/resolve/main/config.json"
HF_API_URL    = "https://huggingface.co/api/models/{model_id}"

GREENCOYOTE_VERSION = "1.0"

# Tile dimensions at current process node
TILE_MAX_ROWS = 1500
TILE_MAX_COLS = 1500
TILE_ELEMENTS = TILE_MAX_ROWS * TILE_MAX_COLS   # 2.25M elements per tile

# =============================================================================
# Model architecture types
# =============================================================================

class ArchType(str, Enum):
    DENSE_TRANSFORMER     = "transformer-dense"
    MOE_TRANSFORMER       = "transformer-moe"
    UNKNOWN               = "unknown"

# =============================================================================
# GreenCoyote model manifest dataclass
# =============================================================================

@dataclass
class ModelManifest:
    # Identity
    model_id:               str  = ""
    model_name:             str  = ""
    architecture:           str  = ArchType.UNKNOWN
    greencoyote_version:    str  = GREENCOYOTE_VERSION

    # Transformer topology
    num_layers:             int  = 0
    hidden_dim:             int  = 0
    num_attention_heads:    int  = 0
    num_kv_heads:           int  = 0
    intermediate_dim:       int  = 0   # feed-forward hidden dim
    vocab_size:             int  = 0
    max_sequence_length:    int  = 0

    # MoE topology (zero if dense)
    num_experts:            int  = 0
    num_experts_active:     int  = 0
    num_shared_experts:     int  = 0
    moe_layer_freq:         int  = 0   # every N layers is MoE (0 = all dense)
    num_dense_layers:       int  = 0
    num_moe_layers:         int  = 0

    # Precision
    precision_required:     str  = "FP8"
    dtype_native:           str  = ""

    # GreenCoyote hardware requirements
    min_tile_arrays:        int  = 0
    tiles_attention:        int  = 0
    tiles_feedforward:      int  = 0
    tiles_expert:           int  = 0
    min_crossbar_ports:     int  = 0
    min_cascade_stages:     int  = 0
    min_fiber_channels:     int  = 0
    requires_moe_routing:   bool = False

    # Tiling details
    attention_tiled:        bool = False
    feedforward_tiled:      bool = False
    tiles_per_attention:    int  = 0
    tiles_per_feedforward:  int  = 0
    tiles_per_expert:       int  = 0

    # Warnings generated during manifest creation
    warnings:               list = field(default_factory=list)
    errors:                 list = field(default_factory=list)

# =============================================================================
# HuggingFace config.json reader
# =============================================================================

class HFConfigReader:

    def __init__(self):
        self.raw_config = {}

    def from_hub(self, model_id: str) -> dict:
        """Fetch config.json from HuggingFace Hub."""
        url = HF_CONFIG_URL.format(model_id=model_id)
        print(f"[HF] Fetching config from {url}")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            self.raw_config = resp.json()
            print(f"[HF] Config loaded — architecture: "
                  f"{self.raw_config.get('architectures', ['unknown'])}")
            return self.raw_config
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Model '{model_id}' not found on HuggingFace Hub")
            raise RuntimeError(f"HuggingFace fetch failed: {e}")
        except requests.ConnectionError:
            raise RuntimeError("Cannot reach HuggingFace Hub — check network connection")

    def from_local(self, path: str) -> dict:
        """Load config.json from local file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        print(f"[HF] Loading local config from {path}")
        with open(p) as f:
            self.raw_config = json.load(f)
        return self.raw_config

    def get(self, *keys, default=None):
        """Try multiple key names — HuggingFace is inconsistent across models."""
        for key in keys:
            if key in self.raw_config:
                return self.raw_config[key]
        return default

# =============================================================================
# Architecture detector
# =============================================================================

class ArchitectureDetector:

    # Known MoE architecture class names
    MOE_ARCH_NAMES = {
        "DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM",
        "MistralMoEForCausalLM", "MixtralForCausalLM",
        "Qwen2MoeForCausalLM", "OlmoeForCausalLM",
        "PhiMoEForCausalLM", "GrokForCausalLM",
        "JambaForCausalLM", "ArcticForCausalLM"
    }

    # Known dense architecture class names
    DENSE_ARCH_NAMES = {
        "LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM",
        "GemmaForCausalLM", "Gemma2ForCausalLM", "FalconForCausalLM",
        "GPTNeoXForCausalLM", "GPT2LMHeadModel", "BloomForCausalLM",
        "OPTForCausalLM", "GPTJForCausalLM", "PhiForCausalLM",
        "Phi3ForCausalLM", "StableLmForCausalLM"
    }

    @classmethod
    def detect(cls, config: dict) -> ArchType:
        arch_classes = config.get("architectures", [])
        for arch in arch_classes:
            if arch in cls.MOE_ARCH_NAMES:
                return ArchType.MOE_TRANSFORMER
            if arch in cls.DENSE_ARCH_NAMES:
                return ArchType.DENSE_TRANSFORMER
        # Fallback: check for MoE-specific fields
        if any(k in config for k in [
            "num_experts", "num_local_experts", "n_routed_experts",
            "num_experts_per_tok", "moe_num_experts"
        ]):
            return ArchType.MOE_TRANSFORMER
        return ArchType.DENSE_TRANSFORMER

# =============================================================================
# Tile calculator
# =============================================================================

class TileCalculator:

    @staticmethod
    def tiles_needed(rows: int, cols: int) -> int:
        """Calculate number of tiles needed to cover a matrix of given dimensions."""
        row_tiles = math.ceil(rows / TILE_MAX_ROWS)
        col_tiles = math.ceil(cols / TILE_MAX_COLS)
        return row_tiles * col_tiles

    @staticmethod
    def is_tiled(rows: int, cols: int) -> bool:
        return rows > TILE_MAX_ROWS or cols > TILE_MAX_COLS

# =============================================================================
# Manifest generator — core translation logic
# =============================================================================

class ManifestGenerator:

    def __init__(self):
        self.reader   = HFConfigReader()
        self.detector = ArchitectureDetector()
        self.calc     = TileCalculator()

    def generate(self, model_id: str, local_path: Optional[str] = None) -> ModelManifest:
        """Generate a GreenCoyote model manifest from a HuggingFace config."""

        # Load config
        if local_path:
            config = self.reader.from_local(local_path)
        else:
            config = self.reader.from_hub(model_id)

        manifest = ModelManifest()
        manifest.model_id = model_id

        # Extract model name
        manifest.model_name = config.get("_name_or_path", model_id)

        # Detect architecture type
        arch = self.detector.detect(config)
        manifest.architecture = arch.value

        # =================================================================
        # Extract transformer topology
        # HuggingFace uses inconsistent field names across model families
        # =================================================================

        manifest.num_layers = self._get(config,
            "num_hidden_layers", "n_layer", "num_layers", "n_layers",
            default=0)

        manifest.hidden_dim = self._get(config,
            "hidden_size", "n_embd", "d_model", "dim",
            default=0)

        manifest.num_attention_heads = self._get(config,
            "num_attention_heads", "n_head", "num_heads",
            default=0)

        manifest.num_kv_heads = self._get(config,
            "num_key_value_heads", "num_kv_heads",
            default=manifest.num_attention_heads)

        manifest.intermediate_dim = self._get(config,
            "intermediate_size", "ffn_dim", "n_inner",
            "moe_intermediate_size",  # DeepSeek uses this for expert FFN
            default=manifest.hidden_dim * 4)

        manifest.vocab_size = self._get(config,
            "vocab_size", default=0)

        manifest.max_sequence_length = self._get(config,
            "max_position_embeddings", "max_sequence_length",
            "n_positions", "model_max_length",
            default=4096)

        manifest.dtype_native = self._get(config,
            "torch_dtype", default="float16")

        # =================================================================
        # MoE topology extraction
        # =================================================================

        if arch == ArchType.MOE_TRANSFORMER:
            manifest.requires_moe_routing = True

            manifest.num_experts = self._get(config,
                "n_routed_experts", "num_local_experts",
                "num_experts", "moe_num_experts",
                default=0)

            manifest.num_experts_active = self._get(config,
                "num_experts_per_tok", "top_k", "num_experts_per_token",
                "moe_top_k",
                default=0)

            manifest.num_shared_experts = self._get(config,
                "n_shared_experts", "num_shared_experts",
                default=0)

            # MoE layer frequency — how often MoE layers appear
            # DeepSeek: first_k_dense_replace dense layers then alternating
            first_dense = self._get(config,
                "first_k_dense_replace", "num_dense_layers",
                default=0)

            moe_layer_freq = self._get(config,
                "moe_layer_freq", default=1)

            manifest.moe_layer_freq   = moe_layer_freq
            manifest.num_dense_layers = first_dense

            if first_dense > 0:
                moe_count = manifest.num_layers - first_dense
                manifest.num_moe_layers = moe_count
            else:
                manifest.num_moe_layers = manifest.num_layers
                manifest.num_dense_layers = 0

        else:
            manifest.num_dense_layers = manifest.num_layers
            manifest.num_moe_layers   = 0

        # =================================================================
        # Precision mapping
        # HuggingFace dtype → GreenCoyote precision mode
        # =================================================================

        dtype_map = {
            "float32":  "FP8",    # Run at FP8 — headroom
            "float16":  "FP8",    # FP16 model → FP8 inference
            "bfloat16": "BF16",   # BF16 native
            "float8":   "FP8",    # Native FP8
            "int8":     "FP8",    # Quantized → FP8 equivalent
            "int4":     "INT4",   # Heavily quantized
        }
        manifest.precision_required = dtype_map.get(
            manifest.dtype_native, "FP8")

        # =================================================================
        # Tile calculation
        # =================================================================

        H  = manifest.hidden_dim
        FF = manifest.intermediate_dim
        N  = manifest.num_experts if manifest.num_experts > 0 else 1

        # Attention projection tiles (Q, K, V, O) per layer
        # Q: hidden × hidden, K/V: hidden × kv_heads×head_dim, O: hidden × hidden
        attn_tiles = self.calc.tiles_needed(H, H)
        manifest.tiles_per_attention = attn_tiles * 4  # 4 projections per layer
        manifest.attention_tiled     = self.calc.is_tiled(H, H)

        # Feed-forward tiles per dense layer
        ff_tiles = self.calc.tiles_needed(H, FF)
        manifest.tiles_per_feedforward = ff_tiles * 2  # up + down projection
        manifest.feedforward_tiled      = self.calc.is_tiled(H, FF)

        # Expert tiles per MoE layer
        if arch == ArchType.MOE_TRANSFORMER:
            expert_tiles = self.calc.tiles_needed(H, FF)
            manifest.tiles_per_expert = expert_tiles * 2
        else:
            manifest.tiles_per_expert = 0

        # Total tile arrays needed
        # Dense layers: attention + feed-forward
        # MoE layers: attention + (num_experts × expert_ff)
        tiles_dense = manifest.num_dense_layers * (
            manifest.tiles_per_attention + manifest.tiles_per_feedforward)

        if arch == ArchType.MOE_TRANSFORMER:
            tiles_moe = manifest.num_moe_layers * (
                manifest.tiles_per_attention +
                manifest.num_experts * manifest.tiles_per_expert)
        else:
            tiles_moe = 0

        # Minimum tiles needed — streaming layer by layer
        # Only need tiles for ONE layer at a time in pipeline mode
        # Attention: tiles_per_attention (shared across dense and MoE)
        # Dense layer: attention + feedforward
        # MoE layer: attention + (active_experts × expert_ff)
        # Minimum = worst case single layer

        tiles_dense_layer = (manifest.tiles_per_attention +
                             manifest.tiles_per_feedforward)

        if arch == ArchType.MOE_TRANSFORMER:
            # Only active experts need tiles simultaneously
            tiles_moe_layer = (manifest.tiles_per_attention +
                               manifest.num_experts_active *
                               manifest.tiles_per_expert)
            manifest.min_tile_arrays = max(tiles_dense_layer, tiles_moe_layer)
        else:
            manifest.min_tile_arrays = tiles_dense_layer

        # =================================================================
        # Crossbar and interconnect requirements
        # =================================================================

        manifest.min_crossbar_ports = max(
            8,                              # Minimum 8 ports
            manifest.num_experts_active     # Need one port per active expert
        )

        manifest.min_fiber_channels = manifest.min_crossbar_ports

        manifest.min_cascade_stages = math.ceil(
            math.log(manifest.min_crossbar_ports, 8))

        # =================================================================
        # Warnings for borderline cases
        # =================================================================

        if manifest.attention_tiled:
            manifest.warnings.append(
                f"Attention matrices ({H}×{H}) exceed single tile — "
                f"{manifest.tiles_per_attention} tiles per layer required")

        if manifest.feedforward_tiled:
            manifest.warnings.append(
                f"Feed-forward matrices ({H}×{FF}) exceed single tile — "
                f"{manifest.tiles_per_feedforward} tiles per layer required")

        if manifest.num_experts > 64:
            manifest.warnings.append(
                f"Model has {manifest.num_experts} experts — "
                f"crossbar expansion required beyond 8-port base configuration")

        if manifest.max_sequence_length > 8192:
            manifest.warnings.append(
                f"Context length {manifest.max_sequence_length} — "
                f"KV cache may require additional CXL memory capacity")

        # =================================================================
        # Errors for hard incompatibilities
        # =================================================================

        if manifest.num_layers == 0:
            manifest.errors.append(
                "Could not determine layer count — config.json may be incomplete")

        if manifest.hidden_dim == 0:
            manifest.errors.append(
                "Could not determine hidden dimension — unsupported architecture")

        if manifest.num_experts_active > 64:
            manifest.errors.append(
                f"Model activates {manifest.num_experts_active} experts per token — "
                f"exceeds maximum crossbar fan-out of 64")

        return manifest

    @staticmethod
    def _get(config: dict, *keys, default=None):
        """Try multiple key names, return first match or default."""
        for key in keys:
            if key in config:
                return config[key]
        return default

# =============================================================================
# Hardware capability reader
# =============================================================================

@dataclass
class HardwareManifest:
    tile_arrays:         int  = 16
    tile_max_rows:       int  = TILE_MAX_ROWS
    tile_max_cols:       int  = TILE_MAX_COLS
    crossbar_ports:      int  = 8
    reduction_channels:  int  = 8
    cascade_stages:      int  = 4
    precision_modes:     list = field(default_factory=lambda: [
                                    "INT4", "FP4", "FP8", "BF16"])
    max_experts:         int  = 64
    max_active_experts:  int  = 8
    fiber_channels:      int  = 8
    adc_bits:            int  = 12
    cxl_capacity_gb:     int  = 256

    @classmethod
    def from_fpga(cls):
        """Read capability from live FPGA registers via greencoyote driver."""
        # In production: call greencoyote_pilot.hpp get_hardware_manifest()
        # For now: return default prototype configuration
        print("[HW] Reading hardware capability manifest from FPGA registers")
        print("[HW] Using prototype configuration: 16 tiles, 8 crossbar ports")
        return cls()

    @classmethod
    def from_json(cls, path: str):
        """Load hardware manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

# =============================================================================
# Negotiation engine
# =============================================================================

class NegotiationResult(str, Enum):
    ACCEPTED  = "ACCEPTED"
    DEGRADED  = "DEGRADED"   # Runs with reduced capability
    REJECTED  = "REJECTED"

@dataclass
class NegotiationReport:
    result:       NegotiationResult
    model_id:     str
    errors:       list = field(default_factory=list)
    warnings:     list = field(default_factory=list)
    suggestions:  list = field(default_factory=list)

def negotiate(hw: HardwareManifest,
              model: ModelManifest) -> NegotiationReport:
    """Compare hardware capability against model requirements."""

    errors      = list(model.errors)   # Carry forward manifest errors
    warnings    = list(model.warnings)
    suggestions = []

    # --- Hard requirements ---

    if model.min_tile_arrays > hw.tile_arrays:
        errors.append(
            f"Tile arrays: need {model.min_tile_arrays}, "
            f"hardware has {hw.tile_arrays}")
        suggestions.append(
            f"Add {model.min_tile_arrays - hw.tile_arrays} tile expansion modules")

    if model.min_crossbar_ports > hw.crossbar_ports:
        errors.append(
            f"Crossbar ports: need {model.min_crossbar_ports}, "
            f"hardware has {hw.crossbar_ports}")

    if model.precision_required not in hw.precision_modes:
        errors.append(
            f"Precision mode {model.precision_required} not supported")

    if model.num_experts_active > hw.max_active_experts:
        errors.append(
            f"Active experts per token: need {model.num_experts_active}, "
            f"crossbar supports {hw.max_active_experts}")

    if model.min_cascade_stages > hw.cascade_stages:
        errors.append(
            f"Cascade stages: need {model.min_cascade_stages}, "
            f"hardware has {hw.cascade_stages}")

    # --- Soft warnings ---

    if model.attention_tiled:
        warnings.append(
            f"Attention tiling active: "
            f"{model.tiles_per_attention} tiles per layer — "
            f"adds {(model.tiles_per_attention - 1)} cascade boundaries per attention block")

    if model.feedforward_tiled:
        warnings.append(
            f"Feed-forward tiling active: "
            f"{model.tiles_per_feedforward} tiles per layer")

    if model.max_sequence_length > 4096:
        warnings.append(
            f"Long context ({model.max_sequence_length} tokens) — "
            f"verify CXL capacity for KV cache")

    # --- Suggestions for rejected models ---

    if errors:
        suggestions.append(
            "Run 'greencoyote models list' to see compatible models")
        suggestions.append(
            "Consider a distilled version of this model")

    if errors:
        result = NegotiationResult.REJECTED
    elif warnings:
        result = NegotiationResult.DEGRADED
    else:
        result = NegotiationResult.ACCEPTED

    return NegotiationReport(
        result      = result,
        model_id    = model.model_id,
        errors      = errors,
        warnings    = warnings,
        suggestions = suggestions
    )

# =============================================================================
# CLI output formatter
# =============================================================================

def print_report(model: ModelManifest, hw: HardwareManifest,
                 report: NegotiationReport):
    """Print human-readable negotiation report."""

    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    RESET  = "\033[0m"
    BOLD   = "\033[1m"

    print(f"\n{BOLD}GreenCoyote Pilot v{GREENCOYOTE_VERSION}{RESET}")
    print(f"Loading: {model.model_id}")
    print(f"Architecture: {model.architecture}")
    if model.num_experts > 0:
        print(f"MoE: {model.num_experts} experts, "
              f"{model.num_experts_active} active per token")
    print()
    print("Hardware capability check:")
    print(f"  {'✓' if model.min_tile_arrays <= hw.tile_arrays else '✗'} "
          f"Tile arrays: {hw.tile_arrays} available, "
          f"{model.min_tile_arrays} required")
    print(f"  {'✓' if model.min_crossbar_ports <= hw.crossbar_ports else '✗'} "
          f"Crossbar ports: {hw.crossbar_ports} available, "
          f"{model.min_crossbar_ports} required")
    print(f"  {'✓' if model.precision_required in hw.precision_modes else '✗'} "
          f"Precision: {model.precision_required}")
    if model.num_experts_active > 0:
        print(f"  {'✓' if model.num_experts_active <= hw.max_active_experts else '✗'} "
              f"Active experts: {model.num_experts_active} per token, "
              f"{hw.max_active_experts} supported")
    print()

    if report.warnings:
        print(f"{YELLOW}Warnings:{RESET}")
        for w in report.warnings:
            print(f"  ⚠ {w}")
        print()

    if report.errors:
        print(f"{RED}Errors:{RESET}")
        for e in report.errors:
            print(f"  ✗ {e}")
        print()

    if report.result == NegotiationResult.ACCEPTED:
        print(f"{GREEN}{BOLD}ACCEPTED{RESET} — model is compatible")
        print("Compiling pilot program...")

    elif report.result == NegotiationResult.DEGRADED:
        print(f"{YELLOW}{BOLD}ACCEPTED with warnings{RESET} — "
              f"model runs with tiling")

    elif report.result == NegotiationResult.REJECTED:
        print(f"{RED}{BOLD}REJECTED{RESET} — model cannot run on current hardware")
        if report.suggestions:
            print()
            print("Suggestions:")
            for s in report.suggestions:
                print(f"  → {s}")

    print()

# =============================================================================
# Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GreenCoyote-Bay HuggingFace Model Manifest Generator",
        epilog="Examples:\n"
               "  python3 hf_manifest_gen.py deepseek-ai/DeepSeek-R1\n"
               "  python3 hf_manifest_gen.py --local config.json\n"
               "  python3 hf_manifest_gen.py --hw-manifest hw.json "
               "meta-llama/Llama-3.1-70B-Instruct",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model_id", nargs="?",
                        help="HuggingFace model ID (e.g. deepseek-ai/DeepSeek-R1)")
    parser.add_argument("--local", metavar="PATH",
                        help="Load config.json from local path instead of Hub")
    parser.add_argument("--hw-manifest", metavar="PATH",
                        help="Hardware capability manifest JSON (default: read from FPGA)")
    parser.add_argument("--output", metavar="PATH",
                        default="model_manifest.json",
                        help="Output manifest file (default: model_manifest.json)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed output, print only result")

    args = parser.parse_args()

    if not args.model_id and not args.local:
        parser.print_help()
        sys.exit(1)

    model_id = args.model_id or args.local

    # Load hardware manifest
    if args.hw_manifest:
        hw = HardwareManifest.from_json(args.hw_manifest)
    else:
        hw = HardwareManifest.from_fpga()

    # Generate model manifest
    gen = ManifestGenerator()
    try:
        model = gen.generate(model_id, local_path=args.local)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Run negotiation
    report = negotiate(hw, model)

    # Print report
    if not args.quiet:
        print_report(model, hw, report)

    # Save manifest
    manifest_dict = asdict(model)
    with open(args.output, "w") as f:
        json.dump(manifest_dict, f, indent=2)
    print(f"Manifest saved: {args.output}")

    # Exit code reflects negotiation result
    exit_codes = {
        NegotiationResult.ACCEPTED:  0,
        NegotiationResult.DEGRADED:  0,
        NegotiationResult.REJECTED:  1
    }
    sys.exit(exit_codes[report.result])

if __name__ == "__main__":
    main()
