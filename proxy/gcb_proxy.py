# =============================================================================
# gcb_proxy.py
# GreenCoyote-Bay (GCB) Host-Side MMIO Proxy — Sovereign Safety Supervisor
# =============================================================================
# Architecture:  Hybrid Photonic-Electronic AI Inference Fabric
# Project:       GreenCoyote-Bay — ODIN_Bridge Hierarchy
# Status:        DESIGN SPEC v0.2 — GCB-MEM-1 Integration
# Author:        GCB Architecture Team
#
# Revision History:
#   v0.1 — Queue-manager model: OVERFLOW checks, QUEUE_FULL back-pressure,
#           TieredMMIO spill logic, SOFTMAX_CTRL register bit manipulation.
#   v0.2 — Memory-fabric model: GCB-MEM-1 flat CXL address space replaces
#           all queue logic. TID routing via address arithmetic. Proxy role
#           shifts from queue-manager to memory-fabric orchestrator.
#           Stretch capacity: gracefully absorbed above smooth design limit,
#           STRETCH status bit set, no fault, no stall, no intervention needed.
#
# Responsibilities:
#   1. SOFTMAX_READY IRQ handler — write raw vector to GCB-MEM-1, normalize,
#      write result back. No queue. No depth check. No overflow path.
#   2. MZM bias autonomous calibration — DAC_FREEZE / RATE_LIMIT RMW
#   3. Cascade ADC reset at stage boundaries
#   4. Forensic persistence → gcb_snapshots.db (all events logged)
#   5. GCB-MEM-1 health monitoring — FAULT and STRETCH status tracking
#
# Domain-Crossing Philosophy:
#   The photonic fabric is the math engine (QKV dot-product).
#   This proxy is the traffic controller (temporal orchestration,
#   normalization, fabric health). 3 crossings per attention head per layer.
#   GCB-MEM-1 is internal to the electronic tier — invisible to the fabric.
#
# =============================================================================

import struct
import sqlite3
import time
import threading
import numpy as np
from typing import List, Optional
from gcb_core.gcb_regmap import (
    Reg,
    Mem1CtrlBits, Mem1StatusBits,
    VECTOR_DIM, VECTOR_BYTES, MEM1_SLOT_BYTES,
    MEM1_SMOOTH_TIDS, MEM1_STRETCH_TIDS,
    tid_raw_addr, tid_result_addr, tid_in_stretch,
    # Legacy shim imports — do not use in new logic
    SoftmaxCtrlBits, QUEUE_MAX_DEPTH,
)

SNAPSHOT_DB = "gcb_snapshots.db"


# =============================================================================
# GCB-MEM-1: Flat Byte-Addressable Memory Abstraction
# =============================================================================

class GCBMem1:
    """
    Abstraction for the GCB-MEM-1 CXL Type-3 persistent memory fabric.

    In hardware: this wraps an mmap() against the CXL BAR exposed by the
    Microchip PM40028 or Xilinx Versal CXL IP controller.

    In simulation: backed by a bytearray — unlimited, instantaneous,
    behaviorally identical to the hardware target.

    Design invariant:
      - No queue. No depth. No overflow path.
      - TID routing is pure address arithmetic.
      - Smooth capacity: MEM1_SMOOTH_TIDS concurrent TIDs — nominal operation.
      - Stretch capacity: MEM1_SMOOTH_TIDS to MEM1_STRETCH_TIDS — gracefully
        absorbed. STRETCH status bit set for telemetry. No fault. No stall.
      - Above stretch: physical hardware limit — not a software concern.
    """

    def __init__(self, sim_capacity_tids: int = MEM1_STRETCH_TIDS * 4):
        # Simulation backing store — sized generously beyond stretch ceiling
        self._mem = bytearray(sim_capacity_tids * MEM1_SLOT_BYTES)
        self._base = 0   # In hardware: CXL BAR physical base address
        self._lock = threading.Lock()
        self._active_tids: set = set()
        print(
            f"[GCB-MEM-1] Initialized. "
            f"Smooth={MEM1_SMOOTH_TIDS} TIDs | "
            f"Stretch={MEM1_STRETCH_TIDS} TIDs | "
            f"Sim backing={sim_capacity_tids * MEM1_SLOT_BYTES // 1024}KB"
        )

    def write_raw(self, tid: int, vec_bytes: bytes):
        """Write raw photonic output vector to this TID's raw slot."""
        addr = tid_raw_addr(tid, self._base)
        with self._lock:
            self._mem[addr : addr + VECTOR_BYTES] = vec_bytes
            self._active_tids.add(tid)

    def write_result(self, tid: int, vec_bytes: bytes):
        """Write normalized Softmax result to this TID's result slot."""
        addr = tid_result_addr(tid, self._base)
        with self._lock:
            self._mem[addr : addr + VECTOR_BYTES] = vec_bytes

    def read_raw(self, tid: int) -> bytes:
        """Read raw vector from this TID's raw slot."""
        addr = tid_raw_addr(tid, self._base)
        with self._lock:
            return bytes(self._mem[addr : addr + VECTOR_BYTES])

    def read_result(self, tid: int) -> bytes:
        """Read normalized result from this TID's result slot."""
        addr = tid_result_addr(tid, self._base)
        with self._lock:
            return bytes(self._mem[addr : addr + VECTOR_BYTES])

    def release(self, tid: int):
        """Mark TID slot as free (zero the slot, remove from active set)."""
        raw_addr    = tid_raw_addr(tid, self._base)
        result_addr = tid_result_addr(tid, self._base)
        with self._lock:
            self._mem[raw_addr    : raw_addr    + VECTOR_BYTES] = bytes(VECTOR_BYTES)
            self._mem[result_addr : result_addr + VECTOR_BYTES] = bytes(VECTOR_BYTES)
            self._active_tids.discard(tid)

    @property
    def active_tid_count(self) -> int:
        with self._lock:
            return len(self._active_tids)

    @property
    def in_stretch(self) -> bool:
        """True when active TID count is above smooth design capacity."""
        return self.active_tid_count > MEM1_SMOOTH_TIDS


# =============================================================================
# MockMMIO — FPGA Register File Simulation
# =============================================================================

class MockMMIO:
    """
    Simulated MMIO register file for simulation harness.
    Replace with platform MMIO driver (/dev/mem mmap, PCIe BAR) on hardware.
    """
    def __init__(self):
        self._regs: dict = {}
        self._bram: dict = {}
        self._lock = threading.Lock()

    def read(self, addr: int) -> int:
        with self._lock:
            return self._regs.get(addr, 0)

    def write(self, addr: int, value: int):
        with self._lock:
            self._regs[addr] = value

    def read_block(self, addr: int, length: int) -> bytes:
        with self._lock:
            return self._bram.get(addr, bytes(length))

    def write_block(self, addr: int, data: bytes):
        with self._lock:
            self._bram[addr] = data

    def read_modify_write(self, addr: int, mask: int, value: int):
        """Atomic RMW — used by supervisor for bias calibration."""
        with self._lock:
            current = self._regs.get(addr, 0)
            self._regs[addr] = (current & ~mask) | (value & mask)


# =============================================================================
# Forensic Snapshot Logger
# =============================================================================

class SnapshotLogger:
    """
    Persistent forensic log — all proxy actions recorded.
    gcb_snapshots.db is the primary evidence of production-grade stability.
    Stretch-capacity events are logged as telemetry, not faults.
    """

    def __init__(self, db_path: str = SNAPSHOT_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS softmax_events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            REAL,
                tid           INTEGER,
                mem1_raw_addr INTEGER,
                mem1_res_addr INTEGER,
                latency_us    REAL,
                stretch       INTEGER,
                fault         INTEGER,
                fault_code    TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS supervisor_actions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL,
                action      TEXT,
                register    TEXT,
                old_value   INTEGER,
                new_value   INTEGER,
                reason      TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS adc_resets (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           REAL,
                stage        INTEGER,
                noise_floor  REAL,
                triggered_by TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mem1_telemetry (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              REAL,
                active_tids     INTEGER,
                in_stretch      INTEGER,
                smooth_ceiling  INTEGER,
                stretch_ceiling INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def log_softmax_event(
        self,
        tid: int,
        mem1_raw_addr: int,
        mem1_res_addr: int,
        latency_us: float,
        stretch: bool,
        fault: bool,
        fault_code: str = ""
    ):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO softmax_events
               (ts, tid, mem1_raw_addr, mem1_res_addr,
                latency_us, stretch, fault, fault_code)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                time.time(), tid,
                mem1_raw_addr, mem1_res_addr,
                latency_us, int(stretch),
                int(fault), fault_code,
            )
        )
        conn.commit()
        conn.close()

    def log_supervisor_action(
        self,
        action: str,
        register: str,
        old_value: int,
        new_value: int,
        reason: str
    ):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO supervisor_actions
               (ts, action, register, old_value, new_value, reason)
               VALUES (?,?,?,?,?,?)""",
            (time.time(), action, register, old_value, new_value, reason)
        )
        conn.commit()
        conn.close()

    def log_adc_reset(self, stage: int, noise_floor: float, triggered_by: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO adc_resets (ts, stage, noise_floor, triggered_by)
               VALUES (?,?,?,?)""",
            (time.time(), stage, noise_floor, triggered_by)
        )
        conn.commit()
        conn.close()

    def log_mem1_telemetry(self, active_tids: int, in_stretch: bool):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO mem1_telemetry
               (ts, active_tids, in_stretch, smooth_ceiling, stretch_ceiling)
               VALUES (?,?,?,?,?)""",
            (
                time.time(), active_tids, int(in_stretch),
                MEM1_SMOOTH_TIDS, MEM1_STRETCH_TIDS,
            )
        )
        conn.commit()
        conn.close()


# =============================================================================
# Softmax — Electronic Domain Implementation
# =============================================================================

def softmax_electronic(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable Softmax — runs in electronic (DSP) domain only.
    Photonic domain cannot perform exponentiation (non-linear, global op).
    x_shifted prevents e^x overflow for large attention scores.
    """
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum()


# =============================================================================
# GCB Proxy — Sovereign Safety Supervisor v0.2
# =============================================================================

class GCBProxy:
    """
    v0.2: Memory-Fabric Orchestrator.

    The proxy no longer manages queue depth, overflow, or spill tiers.
    It writes to GCB-MEM-1 addresses and monitors fabric health.
    Stretch capacity is a telemetry signal, not an intervention trigger.
    """

    def __init__(
        self,
        mmio: MockMMIO,
        mem1: GCBMem1,
        db_path: str = SNAPSHOT_DB
    ):
        self.mmio   = mmio
        self.mem1   = mem1
        self.logger = SnapshotLogger(db_path)
        print("[GCB PROXY] v0.2 Initialized. Memory-fabric orchestrator online.")

    # ── SOFTMAX_READY IRQ Handler (v0.2 — address-based) ────────────────────

    def handle_softmax_ready(self, tid: int, raw_bytes: bytes) -> bool:
        """
        IRQ handler: SOFTMAX_READY  (v0.2)

        Flow:
          1. Check stretch status — log telemetry if above smooth capacity.
             This is NOT a fault. Operations continue without intervention.
          2. Write raw vector to GCB-MEM-1 TID raw slot (address arithmetic)
          3. Apply electronic Softmax
          4. Write result to GCB-MEM-1 TID result slot
          5. Signal photonic fabric to resume (BRAM_STREAM_PORT write)
          6. Forensic log — latency, stretch flag, addresses

        Note: No OVERFLOW check. No QUEUE_FULL check. No TID_VALID register
        read. No SOFTMAX_CTRL manipulation. TID is passed directly from the
        IRQ dispatcher — it is already validated at the fabric interface layer.

        Returns True on success, False on hardware fault.
        """
        t_start = time.perf_counter()

        # ── Stretch capacity check — telemetry only, not intervention ────────
        in_stretch = self.mem1.in_stretch
        if in_stretch:
            print(
                f"[GCB PROXY] STRETCH: {self.mem1.active_tid_count} active TIDs "
                f"(smooth ceiling={MEM1_SMOOTH_TIDS}). "
                f"Gracefully absorbed by GCB-MEM-1."
            )
            self.logger.log_mem1_telemetry(
                active_tids=self.mem1.active_tid_count,
                in_stretch=True
            )
            # Update MEM1_STATUS STRETCH bit on FPGA for fabric awareness
            status = self.mmio.read(Reg.MEM1_STATUS)
            self.mmio.write(Reg.MEM1_STATUS, status | Mem1StatusBits.STRETCH)

        # ── GCB-MEM-1 fault check ────────────────────────────────────────────
        mem1_status = self.mmio.read(Reg.MEM1_STATUS)
        if mem1_status & Mem1StatusBits.FAULT:
            print(f"[GCB PROXY] FAULT: GCB-MEM-1 fabric fault. Asserting DAC_FREEZE.")
            self.assert_dac_freeze(reason="MEM1_FAULT")
            self.logger.log_softmax_event(
                tid=tid,
                mem1_raw_addr=tid_raw_addr(tid),
                mem1_res_addr=tid_result_addr(tid),
                latency_us=0.0,
                stretch=in_stretch,
                fault=True,
                fault_code="MEM1_FAULT"
            )
            return False

        # ── Write raw vector → GCB-MEM-1 raw slot ───────────────────────────
        raw_addr = tid_raw_addr(tid)
        self.mem1.write_raw(tid, raw_bytes)

        # ── Electronic Softmax ───────────────────────────────────────────────
        raw_vec    = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        result_arr = softmax_electronic(raw_vec)
        result_bytes = result_arr.astype(np.float32).tobytes()

        # ── Write result → GCB-MEM-1 result slot ────────────────────────────
        res_addr = tid_result_addr(tid)
        self.mem1.write_result(tid, result_bytes)

        # ── Signal photonic fabric to resume ─────────────────────────────────
        # Write normalized result to BRAM_STREAM_PORT for EOM array pickup.
        # Domain crossing #3: Electronic → Photonic.
        self.mmio.write_block(Reg.BRAM_STREAM_PORT, result_bytes)

        # ── Release TID slot in GCB-MEM-1 ───────────────────────────────────
        self.mem1.release(tid)

        t_end = time.perf_counter()
        latency_us = (t_end - t_start) * 1e6

        # ── Forensic log ─────────────────────────────────────────────────────
        self.logger.log_softmax_event(
            tid=tid,
            mem1_raw_addr=raw_addr,
            mem1_res_addr=res_addr,
            latency_us=latency_us,
            stretch=in_stretch,
            fault=False
        )

        peak = float(result_arr.max())
        stretch_tag = " [STRETCH]" if in_stretch else ""
        print(
            f"[GCB PROXY] TID={tid:#06x}{stretch_tag} | "
            f"Softmax OK | "
            f"Latency={latency_us:.2f}µs | "
            f"Peak={peak:.4f} | "
            f"MEM1_raw={raw_addr:#010x} MEM1_res={res_addr:#010x}"
        )
        return True

    # ── Cascade ADC Reset ────────────────────────────────────────────────────

    def cascade_adc_reset(self, stage: int, noise_floor: float):
        """
        Assert ADC reset at stage boundary.
        Prevents exponential noise accumulation across deep photonic layers.
        """
        print(f"[GCB PROXY] ADC RESET: Stage {stage} | Noise floor={noise_floor:.6f}")
        self.mmio.write(Reg.ADC_RESET, 0x01)
        time.sleep(0.0001)
        self.mmio.write(Reg.ADC_RESET, 0x00)
        self.logger.log_adc_reset(stage, noise_floor, triggered_by="proxy_supervisor")

    # ── MZM Bias Calibration ─────────────────────────────────────────────────

    def calibrate_mzm_bias(self, target_snr: float, measured_snr: float):
        """
        Autonomous RMW on MZM_BIAS_BASE.
        Maintains SNR during multi-week runs. All adjustments logged.
        """
        if measured_snr >= target_snr:
            return

        old_bias   = self.mmio.read(Reg.MZM_BIAS_BASE)
        correction = int((target_snr - measured_snr) * 10)
        new_bias   = old_bias + correction
        self.mmio.write(Reg.MZM_BIAS_BASE, new_bias)

        self.logger.log_supervisor_action(
            action="MZM_BIAS_ADJUST",
            register="MZM_BIAS_BASE",
            old_value=old_bias,
            new_value=new_bias,
            reason=f"SNR drift: measured={measured_snr:.3f} target={target_snr:.3f}"
        )
        print(
            f"[GCB PROXY] MZM BIAS: {old_bias} → {new_bias} | "
            f"SNR {measured_snr:.3f} → target {target_snr:.3f}"
        )

    # ── DAC Freeze ───────────────────────────────────────────────────────────

    def assert_dac_freeze(self, reason: str = "supervisor_override"):
        old = self.mmio.read(Reg.DAC_FREEZE)
        self.mmio.write(Reg.DAC_FREEZE, 0x01)
        self.logger.log_supervisor_action(
            action="DAC_FREEZE", register="DAC_FREEZE",
            old_value=old, new_value=0x01, reason=reason
        )
        print(f"[GCB PROXY] DAC FREEZE asserted. Reason: {reason}")

    def release_dac_freeze(self):
        old = self.mmio.read(Reg.DAC_FREEZE)
        self.mmio.write(Reg.DAC_FREEZE, 0x00)
        self.logger.log_supervisor_action(
            action="DAC_UNFREEZE", register="DAC_FREEZE",
            old_value=old, new_value=0x00, reason="supervisor_release"
        )
        print("[GCB PROXY] DAC FREEZE released.")


# =============================================================================
# Entry Point — Simulation / Dev Mode
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GreenCoyote-Bay — GCB Proxy Supervisor v0.2")
    print("Mode: Simulation (MockMMIO + GCBMem1 sim backing)")
    print("=" * 70)

    mmio  = MockMMIO()
    mem1  = GCBMem1()
    proxy = GCBProxy(mmio, mem1, db_path=SNAPSHOT_DB)

    # Activate GCB-MEM-1 fabric
    mmio.write(Reg.MEM1_CTRL, Mem1CtrlBits.ENABLE)
    mmio.write(Reg.MEM1_STATUS, Mem1StatusBits.READY)

    print("\n── Smooth Capacity Test (TIDs within smooth ceiling) ──")
    for tid in [0x001, 0x002, 0x003]:
        raw = np.random.randn(VECTOR_DIM).astype(np.float32).tobytes()
        proxy.handle_softmax_ready(tid, raw)

    print("\n── Stretch Capacity Test (TIDs above smooth ceiling) ──")
    # Simulate active TID count above smooth ceiling
    # by pre-loading mem1 active set
    for fake_tid in range(MEM1_SMOOTH_TIDS + 1):
        mem1._active_tids.add(fake_tid)

    stretch_tid = 0x200
    raw = np.random.randn(VECTOR_DIM).astype(np.float32).tobytes()
    proxy.handle_softmax_ready(stretch_tid, raw)

    # Clear artificial load
    mem1._active_tids.clear()

    print("\n── ADC Reset at Stage Boundary ──")
    proxy.cascade_adc_reset(stage=3, noise_floor=0.000312)

    print("\n── MZM Bias Drift Correction ──")
    proxy.calibrate_mzm_bias(target_snr=18.5, measured_snr=16.2)

    print("\n── Forensic DB Summary ──")
    import sqlite3 as _sql
    conn = _sql.connect(SNAPSHOT_DB)
    for tbl in ["softmax_events", "supervisor_actions", "adc_resets", "mem1_telemetry"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl}: {count} row(s)")
    conn.close()

    print("\n[GCB PROXY] Simulation complete. Forensic log: gcb_snapshots.db")
