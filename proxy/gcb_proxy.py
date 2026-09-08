# =============================================================================
# gcb_proxy.py
# GreenCoyote-Bay (GCB) Host-Side MMIO Proxy
# =============================================================================
# Architecture:  Hybrid Photonic-Electronic AI Inference Fabric (ODIN_Bridge)
# Register map:  generated/regmap.py  (map_version 0.6.0)
#
# Role
#   Host-side driver + simulator for the GCB pilot MMIO surface. Composes and
#   decodes register words through the *generated* dataclasses (Field / Register
#   / REGISTERS) so the proxy can never drift from registers.yaml.
#
# v0.6.0 architecture (this rewrite)
#   * Streaming Tensor DMA block 0xC4-0xE4: an autonomous descriptor-ring DMA
#     feeds activation + weight operand tiles to the optical modulators with no
#     per-operand host instruction. Programmed by:
#       - DMA_DESC_WINDOW  : auto-increment 32-bit write port; 8 writes commit
#                            one 32-byte descriptor into the on-die ring.
#       - DMA_DESC_TAIL    : producer index; a write here is the DOORBELL.
#       - DMA_WEIGHT_TILE_* : base + shape of the weight-stationary tile; a write
#                            to _HI latches the tile and starts the load.
#       - DMA_CTRL         : EN, WINDOW_W, RING_SRC, ARM (w1p), ABORT (w1p).
#       - DMA_STATUS (ro)  : RUNNING / ARMED / WT_LOADED / UNDERRUN / DESC_ERR /
#                            ROWS_BUFFERED / LAST_DESC_ID.
#   * REDUCTION_CTRL 0x10: FIRE (w1p) triggers one optical reduction;
#     ROUND_MODE selects the block-floating-point rounding; the rest is ro
#     sequencer status.
#
# The legacy queue-manager / GCB-MEM-1 / SOFTMAX model of v0.1-v0.2 is removed
# in its entirety.
# =============================================================================

from __future__ import annotations

from generated.regmap import (
    MAP_VERSION,
    UNMAPPED_READ,
    Field,
    Register,
    REGISTERS,
    BY_OFFSET,
    decode,
)

# One 32-byte tensor descriptor = 8 x 32-bit words
# (docs/streaming_dma_architecture.md sec 2.2).
DESC_WORDS = 8
DESC_FLAG_LAST    = 1 << 0
DESC_FLAG_IRQ     = 1 << 1
DESC_FLAG_WEIGHT  = 1 << 2
DESC_FLAG_PRELOAD = 1 << 3
DESC_PRECISION_SHIFT = 4     # flags[7:4] mirrors PRECISION_MODE.MODE_SEL


# -----------------------------------------------------------------------------
# MockMMIO — 32-bit word-addressed FPGA register file simulation
# -----------------------------------------------------------------------------
class MockMMIO:
    """Simulates the pilot AXI-Lite register file for the v0.6.0 map.

    Backed by a flat {offset: 32-bit word} store seeded from each register's
    composed reset value. Models the access-type semantics the hardware
    enforces: `w1p` fields self-clear after a write, `wo` registers read back 0,
    and an unmapped offset returns UNMAPPED_READ (0xDEADBEEF).
    """

    MASK32 = 0xFFFFFFFF

    def __init__(self) -> None:
        self.store: dict[int, int] = {}
        # write-only registers (no readback path at all)
        self._wo_offsets = {
            r.offset for r in REGISTERS.values()
            if r.fields and all(f.access in ("wo", "rsvd") for f in r.fields)
        }
        self.reset()

    # ---- lifecycle ----------------------------------------------------------
    @staticmethod
    def _reset_word(reg: Register) -> int:
        word = 0
        for f in reg.fields:
            if f.reset is not None:
                word = f.set(word, f.reset)
        return word & MockMMIO.MASK32

    def reset(self) -> None:
        self.store = {r.offset: self._reset_word(r) for r in REGISTERS.values()}

    # ---- raw word access --------------------------------------------------
    def read(self, offset: int) -> int:
        if offset in self._wo_offsets:
            return 0
        if offset not in BY_OFFSET:
            return UNMAPPED_READ
        return self.store.get(offset, 0) & self.MASK32

    def write(self, offset: int, value: int) -> None:
        value &= self.MASK32
        if offset not in BY_OFFSET:
            raise KeyError(f"write to unmapped MMIO offset 0x{offset:02X}")
        reg = BY_OFFSET[offset]
        self.store[offset] = value
        # w1p fields are one-cycle strobes: they always read back 0.
        for f in reg.fields:
            if f.access == "w1p":
                self.store[offset] = f.set(self.store[offset], 0)

    # ---- field-level helpers -------------------------------------------
    def read_fields(self, name: str) -> dict:
        reg = REGISTERS[name]
        return reg.decode(self.read(reg.offset))

    def write_fields(self, name: str, **field_values: int) -> None:
        reg = REGISTERS[name]
        word = self.read(reg.offset) if reg.offset not in self._wo_offsets else 0
        fmap = {f.name: f for f in reg.fields}
        for fname, fval in field_values.items():
            if fname not in fmap:
                raise KeyError(f"{name} has no field {fname!r}")
            if isinstance(fval, str):
                fld = fmap[fname]
                if not fld.enum:
                    raise ValueError(f"{name}.{fname} has no enum for {fval!r}")
                inv = {v: k for k, v in fld.enum.items()}
                fval = inv[fval]
            word = fmap[fname].set(word, fval)
        self.write(reg.offset, word)


# -----------------------------------------------------------------------------
# GCBProxy — host driver for the v0.6.0 fabric
# -----------------------------------------------------------------------------
class GCBProxy:
    """Host-side orchestrator for the GreenCoyote-Bay pilot.

    Primary flow:  set precision -> load weight tile -> push descriptor(s) ->
                   ring the doorbell -> arm the DMA window -> trigger a
                   reduction -> read status.
    """

    def __init__(self, mmio: MockMMIO | None = None) -> None:
        self.mmio = mmio or MockMMIO()
        self._ring_words = 0        # words written into the current descriptor
        self._ring_tail = 0        # descriptors committed to the on-die ring

    # ---- precision -------------------------------------------------------
    def set_precision(self, mode: str = "FP8") -> None:
        self.mmio.write_fields("PRECISION_MODE", MODE_SEL=mode)

    def precision(self) -> int:
        return REGISTERS["PRECISION_MODE"].fields[0].decode(
            self.mmio.read(REGISTERS["PRECISION_MODE"].offset))

    # ---- Streaming Tensor DMA -----------------------------------------
    def load_weight_tile(self, base_addr: int, shape: int = 0) -> None:
        """Program the weight-stationary tile. Writing _HI starts the load."""
        self.mmio.write_fields("DMA_WEIGHT_TILE_LO", WT_BASE_LO=base_addr & 0xFFFFFFFF)
        self.mmio.write_fields(
            "DMA_WEIGHT_TILE_HI",
            WT_BASE_HI=(base_addr >> 32) & 0xFFFF,
            WT_SHAPE=shape & 0xF,
        )

    def push_descriptor(self, words: list[int]) -> int:
        """Append one 32-byte descriptor to the on-die ring via the
        auto-increment DMA_DESC_WINDOW port, then advance / doorbell TAIL."""
        if len(words) != DESC_WORDS:
            raise ValueError(f"descriptor must be {DESC_WORDS} words, got {len(words)}")
        win = REGISTERS["DMA_DESC_WINDOW"].offset
        for w in words:
            self.mmio.write(win, w & 0xFFFFFFFF)
            self._ring_words += 1
        self._ring_tail += 1
        self.mmio.write_fields("DMA_DESC_TAIL", TAIL=self._ring_tail)   # DOORBELL
        return self._ring_tail

    def arm_dma_window(self, window_rows: int, ring_src: str = "ONDIE") -> dict:
        """Enable the DMA and arm the feed for the next accumulation window.

        Writes REGISTERS["DMA_CTRL"].offset:  EN=1, WINDOW_W=window_rows,
        RING_SRC, then a second write pulsing the self-clearing ARM strobe.
        """
        ctrl_off = REGISTERS["DMA_CTRL"].offset
        self.mmio.write_fields(
            "DMA_CTRL", EN=1, WINDOW_W=window_rows & 0xFFF, RING_SRC=ring_src,
        )
        cur = self.mmio.read(ctrl_off)
        arm = {f.name: f for f in REGISTERS["DMA_CTRL"].fields}["ARM"]
        self.mmio.write(ctrl_off, arm.set(cur, 1))     # ARM (w1p) -> self-clears
        return self.dma_status()

    def abort_dma(self) -> dict:
        ctrl_off = REGISTERS["DMA_CTRL"].offset
        cur = self.mmio.read(ctrl_off)
        ab = {f.name: f for f in REGISTERS["DMA_CTRL"].fields}["ABORT"]
        self.mmio.write(ctrl_off, ab.set(cur, 1))
        return self.dma_status()

    def dma_status(self) -> dict:
        return self.mmio.read_fields("DMA_STATUS")

    # ---- optical reduction -----------------------------------------------
    def trigger_reduction(self, round_mode: str = "RNE", auto_fire: bool = False) -> dict:
        """Fire one optical reduction.

        Writes REGISTERS["REDUCTION_CTRL"].offset:  ROUND_MODE + AUTO_FIRE
        config, then a FIRE (w1p) strobe.
        """
        rc_off = REGISTERS["REDUCTION_CTRL"].offset
        self.mmio.write_fields(
            "REDUCTION_CTRL", ROUND_MODE=round_mode, AUTO_FIRE=int(auto_fire),
        )
        cur = self.mmio.read(rc_off)
        fire = {f.name: f for f in REGISTERS["REDUCTION_CTRL"].fields}["FIRE"]
        self.mmio.write(rc_off, fire.set(cur, 1))       # FIRE (w1p) -> self-clears
        return self.mmio.read_fields("REDUCTION_CTRL")

    # ---- introspection -------------------------------------------------
    def dump(self, names: list[str] | None = None) -> None:
        names = names or list(REGISTERS)
        print(f"--- GCB register state (map_version {MAP_VERSION}) ---")
        for name in names:
            reg = REGISTERS[name]
            raw = self.mmio.read(reg.offset)
            fields = decode(reg.offset, raw)
            body = ", ".join(f"{k}={v}" for k, v in fields.items()) or "(ro/wo)"
            print(f"  0x{reg.offset:02X} {name:<22} = 0x{raw:08X}  [{reg.domain}]  {body}")


# -----------------------------------------------------------------------------
# Descriptor helper (host-side packing of the 8-word tensor descriptor)
# -----------------------------------------------------------------------------
def make_descriptor(
    base_addr: int,
    inner_extent: int,
    outer_extent: int,
    *,
    elem_log2_bytes: int = 1,
    inner_stride: int = 1,
    outer_stride: int = 0,
    is_weight: bool = False,
    last: bool = True,
    precision: int = 2,          # FP8 -> mirrors PRECISION_MODE.MODE_SEL
) -> list[int]:
    """Pack a 256-bit descriptor into 8 little-endian 32-bit words
    (docs/streaming_dma_architecture.md sec 2.2)."""
    d = 0
    d |= (base_addr & ((1 << 48) - 1)) << 0
    d |= (elem_log2_bytes & 0xF) << 48
    d |= (inner_extent & 0xFFFF) << 52
    d |= (inner_stride & 0xFFFFFFFF) << 68
    d |= (outer_extent & 0xFFFF) << 100
    d |= (outer_stride & 0xFFFFFFFF) << 116
    flags = 0
    flags |= DESC_FLAG_LAST if last else 0
    flags |= DESC_FLAG_WEIGHT if is_weight else 0
    flags |= (precision & 0xF) << DESC_PRECISION_SHIFT
    d |= (flags & 0xFFFF) << 208
    return [(d >> (32 * i)) & 0xFFFFFFFF for i in range(DESC_WORDS)]


# -----------------------------------------------------------------------------
# Simulation entry point — demonstrate the v0.6.0 MMIO map end to end
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"GCB proxy — regmap map_version {MAP_VERSION}, "
          f"{len(REGISTERS)} registers, unmapped read 0x{UNMAPPED_READ:08X}")

    proxy = GCBProxy()

    # 1. precision
    proxy.set_precision("FP8")

    # 2. weight-stationary tile: base 0x0001_0000, shape 0
    proxy.load_weight_tile(0x0001_0000, shape=0)

    # 3. mock DMA descriptors -> on-die ring, then doorbell
    wt_desc = make_descriptor(
        0x0001_0000, inner_extent=8192, outer_extent=64,
        outer_stride=8192, is_weight=True, last=False,
    )
    act_desc = make_descriptor(
        0x0002_0000, inner_extent=16384, outer_extent=1, last=True,
    )
    tail = proxy.push_descriptor(wt_desc)
    tail = proxy.push_descriptor(act_desc)
    print(f"pushed 2 descriptors; ring tail (doorbell) = {tail}")

    # 4. arm the DMA window (EN + WINDOW_W + ARM strobe)
    st = proxy.arm_dma_window(window_rows=8)
    print(f"after arm_dma_window: DMA_STATUS = {st}")

    # 5. trigger one optical reduction (ROUND_MODE + FIRE strobe)
    rc = proxy.trigger_reduction(round_mode="RNE")
    print(f"after trigger_reduction: REDUCTION_CTRL = {rc}")

    # 6. show the live register state — proves the v0.6.0 map is in use
    proxy.dump([
        "PRECISION_MODE",
        "REDUCTION_CTRL",
        "DMA_CTRL",
        "DMA_DESC_TAIL",
        "DMA_WEIGHT_TILE_LO",
        "DMA_WEIGHT_TILE_HI",
        "DMA_STATUS",
    ])

    # 7. sanity: w1p strobes self-cleared, unmapped read sentinel
    assert proxy.mmio.read_fields("DMA_CTRL")["ARM"] == 0, "ARM should self-clear"
    assert proxy.mmio.read_fields("REDUCTION_CTRL")["FIRE"] == 0, "FIRE should self-clear"
    assert proxy.mmio.read(0xF0) == UNMAPPED_READ, "unmapped read sentinel"
    print("OK — v0.6.0 MMIO map exercised (DMA descriptor + doorbell + arm + reduction).")
