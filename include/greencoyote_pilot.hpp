// =============================================================================
// GreenCoyote-Bay — Host-Side C++ Driver
// Project: ODIN_Bridge / Cascaded Optical Reduction Fabric
// Author:  John A. Green
// Date:    June 2026
// =============================================================================
// Provides a clean C++ API for the GreenCoyote-Bay Pilot FPGA.
// Targets Alveo U55C via Xilinx Runtime (XRT) MMIO interface.
// Falls back to /dev/mem mmap for bare-metal or debug access.
//
// Usage:
//   GreenCoyotePilot pilot("path/to/xclbin", 0);
//   pilot.initialize(GreenCoyotePilot::MODE_FP8);
//   pilot.set_gain_trim_all(0xFF);
//   pilot.enable_auto_fire(true);
//   pilot.enable_irq(true);
//   // ... GPU kernels write to buf_data ...
//   pilot.wait_fire_done(5000);  // 5 second timeout
//   pilot.clear_irq();
// =============================================================================

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <functional>

// XRT headers — available when targeting Alveo
#ifdef USE_XRT
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "experimental/xrt_ip.h"
#endif

// Bare-metal fallback headers
#ifndef USE_XRT
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace greencoyote {

// =============================================================================
// Register map — byte addresses
// =============================================================================

namespace reg {
    constexpr uint32_t PRECISION_MODE   = 0x00;
    constexpr uint32_t GLOBAL_CAL       = 0x04;
    constexpr uint32_t GAIN_TRIM_0_3    = 0x08;
    constexpr uint32_t GAIN_TRIM_4_7    = 0x0C;
    constexpr uint32_t REDUCTION_CTRL   = 0x10;
    constexpr uint32_t IRQ_CTRL         = 0x14;
}

// =============================================================================
// Register 0x00 — Precision Mode Configuration
// =============================================================================

namespace precision_mode {
    constexpr uint32_t MODE_SEL_MASK        = 0x00000007;
    constexpr uint32_t MODE_SEL_SHIFT       = 0;
    constexpr uint32_t NOISE_HEADROOM_MASK  = 0x00000018;
    constexpr uint32_t NOISE_HEADROOM_SHIFT = 3;
    constexpr uint32_t SAT_GUARD_MASK       = 0x00000060;
    constexpr uint32_t SAT_GUARD_SHIFT      = 5;

    constexpr uint32_t INT4 = 0x0;
    constexpr uint32_t FP4  = 0x1;
    constexpr uint32_t FP8  = 0x2;
    constexpr uint32_t BF16 = 0x3;
}

// =============================================================================
// Register 0x04 — Global Calibration
// =============================================================================

namespace global_cal {
    constexpr uint32_t DARK_OFFSET_MASK     = 0x00000FFF;
    constexpr uint32_t DARK_OFFSET_SHIFT    = 0;
    constexpr uint32_t GAIN_TRIM_MASK       = 0x00FFF000;
    constexpr uint32_t GAIN_TRIM_SHIFT      = 12;
    constexpr uint32_t CAL_VALID_BIT        = (1u << 24);
    constexpr uint32_t CAL_LOCK_BIT         = (1u << 25);
}

// =============================================================================
// Register 0x10 — Reduction Control and Status
// =============================================================================

namespace reduction_ctrl {
    constexpr uint32_t FIRE_BIT             = (1u << 0);
    constexpr uint32_t MODE_ACTIVE_BIT      = (1u << 1);
    constexpr uint32_t PASS_INDEX_MASK      = 0x0000000C;
    constexpr uint32_t PASS_INDEX_SHIFT     = 2;
    constexpr uint32_t QUEUE_DEPTH_MASK     = 0x000000F0;
    constexpr uint32_t QUEUE_DEPTH_SHIFT    = 4;
    constexpr uint32_t CAL_ERR_BIT          = (1u << 8);
    constexpr uint32_t FIRE_DONE_BIT        = (1u << 9);
    constexpr uint32_t AUTO_FIRE_BIT        = (1u << 10);

    constexpr uint32_t PASS_1A = 0x0;
    constexpr uint32_t PASS_1B = 0x1;
    constexpr uint32_t PASS_2  = 0x2;
}

// =============================================================================
// Register 0x14 — Interrupt Control
// =============================================================================

namespace irq_ctrl {
    constexpr uint32_t IRQ_EN_BIT           = (1u << 0);
    constexpr uint32_t IRQ_FIRE_DONE_BIT    = (1u << 1);
    constexpr uint32_t IRQ_CAL_ERR_BIT      = (1u << 2);
    constexpr uint32_t IRQ_QUEUE_FULL_BIT   = (1u << 3);
    constexpr uint32_t IRQ_PASS_DONE_BIT    = (1u << 4);
    constexpr uint32_t IRQ_STATUS_BIT       = (1u << 8);
    constexpr uint32_t IRQ_CLEAR_BIT        = (1u << 9);
}

// =============================================================================
// Error types
// =============================================================================

class pilot_error : public std::runtime_error {
public:
    explicit pilot_error(const std::string& msg)
        : std::runtime_error("GreenCoyotePilot: " + msg) {}
};

class cal_error : public pilot_error {
public:
    explicit cal_error(const std::string& msg)
        : pilot_error("Calibration error — " + msg) {}
};

class timeout_error : public pilot_error {
public:
    explicit timeout_error(const std::string& msg)
        : pilot_error("Timeout — " + msg) {}
};

// =============================================================================
// Reduction result
// =============================================================================

struct ReductionResult {
    uint32_t adc_value;         // Raw 12-bit ADC result
    uint32_t pass_index;        // Which pass completed
    bool     cal_err;           // Clip detected during reduction
    uint32_t queue_depth;       // Channels that were active
    std::chrono::microseconds latency; // Wall time for reduction
};

// =============================================================================
// Main driver class
// =============================================================================

class GreenCoyotePilot {
public:

    // Precision mode constants — expose for user code
    static constexpr uint32_t MODE_INT4 = precision_mode::INT4;
    static constexpr uint32_t MODE_FP4  = precision_mode::FP4;
    static constexpr uint32_t MODE_FP8  = precision_mode::FP8;
    static constexpr uint32_t MODE_BF16 = precision_mode::BF16;

    // ==========================================================================
    // Constructor — XRT path
    // ==========================================================================

#ifdef USE_XRT
    GreenCoyotePilot(const std::string& xclbin_path, uint32_t device_index = 0)
        : use_xrt_(true)
    {
        try {
            device_  = xrt::device(device_index);
            uuid_    = device_.load_xclbin(xclbin_path);
            ip_      = xrt::ip(device_, uuid_, "greencoyote_top");
        } catch (const std::exception& e) {
            throw pilot_error(std::string("XRT init failed: ") + e.what());
        }
        std::cout << "[GreenCoyote] XRT device " << device_index
                  << " opened, xclbin loaded\n";
    }
#endif

    // ==========================================================================
    // Constructor — bare-metal /dev/mem fallback
    // ==========================================================================

    GreenCoyotePilot(uintptr_t phys_base_addr, size_t map_size = 0x1000)
        : use_xrt_(false), phys_base_(phys_base_addr)
    {
#ifndef USE_XRT
        fd_ = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd_ < 0)
            throw pilot_error("Failed to open /dev/mem — run as root");

        base_ptr_ = static_cast<volatile uint32_t*>(
            mmap(nullptr, map_size, PROT_READ | PROT_WRITE,
                 MAP_SHARED, fd_, phys_base_addr));

        if (base_ptr_ == MAP_FAILED) {
            close(fd_);
            throw pilot_error("mmap failed for phys addr 0x" +
                              std::to_string(phys_base_addr));
        }
        map_size_ = map_size;
        std::cout << "[GreenCoyote] /dev/mem mapped at 0x"
                  << std::hex << phys_base_addr << std::dec << "\n";
#else
        throw pilot_error("Bare-metal constructor called but USE_XRT defined");
#endif
    }

    // ==========================================================================
    // Destructor
    // ==========================================================================

    ~GreenCoyotePilot() {
#ifndef USE_XRT
        if (!use_xrt_ && base_ptr_ != nullptr && base_ptr_ != MAP_FAILED) {
            munmap(const_cast<uint32_t*>(base_ptr_), map_size_);
            close(fd_);
        }
#endif
    }

    // Disable copy — owns hardware resource
    GreenCoyotePilot(const GreenCoyotePilot&)            = delete;
    GreenCoyotePilot& operator=(const GreenCoyotePilot&) = delete;

    // ==========================================================================
    // Initialization — call once after construction
    // ==========================================================================

    void initialize(uint32_t mode      = MODE_FP8,
                    uint32_t sat_guard = 0x1,
                    uint32_t noise_hdr = 0x1)
    {
        // Set precision mode
        uint32_t prec = ((mode      & 0x7) << precision_mode::MODE_SEL_SHIFT)      |
                        ((noise_hdr & 0x3) << precision_mode::NOISE_HEADROOM_SHIFT) |
                        ((sat_guard & 0x3) << precision_mode::SAT_GUARD_SHIFT);
        write_reg(reg::PRECISION_MODE, prec);

        // Default calibration — zero dark current, unity gain, valid
        write_reg(reg::GLOBAL_CAL, global_cal::CAL_VALID_BIT);

        // Unity gain on all 8 channels
        write_reg(reg::GAIN_TRIM_0_3, 0xFFFFFFFF);
        write_reg(reg::GAIN_TRIM_4_7, 0xFFFFFFFF);

        // Clear any stale IRQ
        write_reg(reg::IRQ_CTRL, irq_ctrl::IRQ_CLEAR_BIT);

        std::cout << "[GreenCoyote] Initialized — mode=0x" << std::hex << mode
                  << " sat_guard=0x" << sat_guard
                  << " noise_hdr=0x"  << noise_hdr << std::dec << "\n";
    }

    // ==========================================================================
    // Calibration
    // ==========================================================================

    void set_dark_offset(uint32_t offset) {
        if (is_cal_locked())
            throw cal_error("Cannot write dark_offset — CAL_LOCK is set");
        uint32_t reg = read_reg(reg::GLOBAL_CAL);
        reg = (reg & ~global_cal::DARK_OFFSET_MASK) |
              ((offset & 0xFFF) << global_cal::DARK_OFFSET_SHIFT);
        write_reg(reg::GLOBAL_CAL, reg);
    }

    void set_cal_lock(bool lock) {
        uint32_t reg = read_reg(reg::GLOBAL_CAL);
        if (lock) reg |=  global_cal::CAL_LOCK_BIT;
        else      reg &= ~global_cal::CAL_LOCK_BIT;
        write_reg(reg::GLOBAL_CAL, reg);
    }

    bool is_cal_locked() const {
        return (read_reg(reg::GLOBAL_CAL) & global_cal::CAL_LOCK_BIT) != 0;
    }

    void set_cal_valid(bool valid) {
        uint32_t reg = read_reg(reg::GLOBAL_CAL);
        if (valid) reg |=  global_cal::CAL_VALID_BIT;
        else       reg &= ~global_cal::CAL_VALID_BIT;
        write_reg(reg::GLOBAL_CAL, reg);
    }

    // ==========================================================================
    // Gain trim — per channel and bulk
    // ==========================================================================

    void set_gain_trim(uint32_t ch, uint8_t trim) {
        if (ch > 7)
            throw pilot_error("Channel " + std::to_string(ch) + " out of range 0-7");

        uint32_t reg_addr = (ch < 4) ? reg::GAIN_TRIM_0_3 : reg::GAIN_TRIM_4_7;
        uint32_t bit_pos  = (ch % 4) * 8;
        uint32_t val      = read_reg(reg_addr);
        val = (val & ~(0xFFu << bit_pos)) | (static_cast<uint32_t>(trim) << bit_pos);
        write_reg(reg_addr, val);
    }

    void set_gain_trim_all(uint8_t trim) {
        uint32_t packed = (static_cast<uint32_t>(trim) << 24) |
                          (static_cast<uint32_t>(trim) << 16) |
                          (static_cast<uint32_t>(trim) <<  8) |
                          (static_cast<uint32_t>(trim) <<  0);
        write_reg(reg::GAIN_TRIM_0_3, packed);
        write_reg(reg::GAIN_TRIM_4_7, packed);
    }

    std::array<uint8_t, 8> get_gain_trim_all() const {
        std::array<uint8_t, 8> trims;
        uint32_t lo = read_reg(reg::GAIN_TRIM_0_3);
        uint32_t hi = read_reg(reg::GAIN_TRIM_4_7);
        for (int i = 0; i < 4; i++) {
            trims[i]   = (lo >> (i * 8)) & 0xFF;
            trims[i+4] = (hi >> (i * 8)) & 0xFF;
        }
        return trims;
    }

    // ==========================================================================
    // Precision mode
    // ==========================================================================

    void set_precision(uint32_t mode) {
        uint32_t reg = read_reg(reg::PRECISION_MODE);
        reg = (reg & ~precision_mode::MODE_SEL_MASK) |
              ((mode & 0x7) << precision_mode::MODE_SEL_SHIFT);
        write_reg(reg::PRECISION_MODE, reg);
    }

    uint32_t get_precision() const {
        return (read_reg(reg::PRECISION_MODE) & precision_mode::MODE_SEL_MASK)
               >> precision_mode::MODE_SEL_SHIFT;
    }

    // ==========================================================================
    // Interrupt control
    // ==========================================================================

    void enable_irq(bool fire_done   = true,
                    bool cal_err     = false,
                    bool queue_full  = false,
                    bool pass_done   = false)
    {
        uint32_t val = irq_ctrl::IRQ_EN_BIT;
        if (fire_done)  val |= irq_ctrl::IRQ_FIRE_DONE_BIT;
        if (cal_err)    val |= irq_ctrl::IRQ_CAL_ERR_BIT;
        if (queue_full) val |= irq_ctrl::IRQ_QUEUE_FULL_BIT;
        if (pass_done)  val |= irq_ctrl::IRQ_PASS_DONE_BIT;
        write_reg(reg::IRQ_CTRL, val);
    }

    void clear_irq() {
        uint32_t val = read_reg(reg::IRQ_CTRL);
        val |= irq_ctrl::IRQ_CLEAR_BIT;
        write_reg(reg::IRQ_CTRL, val);
    }

    bool irq_pending() const {
        return (read_reg(reg::IRQ_CTRL) & irq_ctrl::IRQ_STATUS_BIT) != 0;
    }

    // ==========================================================================
    // Crossbar routing control
    // ==========================================================================

    // Crossbar mode constants
    static constexpr uint32_t XBAR_TENSOR   = 0x0;  // All ports → reduction → broadcast
    static constexpr uint32_t XBAR_PIPELINE = 0x1;  // Sequential chain routing
    static constexpr uint32_t XBAR_MOE      = 0x2;  // Selective expert routing
    static constexpr uint32_t XBAR_CUSTOM   = 0x3;  // Per-port manual routing

    // Destination encoding
    static constexpr uint32_t XBAR_DEST_BROADCAST = 0x7;  // Send to all GPUs
    static constexpr uint32_t XBAR_DEST_REDUCE    = 0x6;  // Send to reduction stage
    static constexpr uint32_t XBAR_DEST_PIPELINE  = 0x5;  // Send to next in chain

    void set_xbar_mode(uint32_t mode) {
        write_reg(0x20, mode & 0x3);
        wait_xbar_ready();
    }

    // Set routing for all 8 ports simultaneously
    // dest[i] specifies where input port i routes to
    void set_xbar_routing(const std::array<uint32_t, 8>& dest) {
        uint32_t lo = ((dest[3] & 0x7) << 9) | ((dest[2] & 0x7) << 6) |
                      ((dest[1] & 0x7) << 3) | (dest[0] & 0x7);
        uint32_t hi = ((dest[7] & 0x7) << 9) | ((dest[6] & 0x7) << 6) |
                      ((dest[5] & 0x7) << 3) | (dest[4] & 0x7);
        write_reg(0x18, lo);
        write_reg(0x1C, hi);
        wait_xbar_ready();
    }

    // Configure tensor parallel — all ports broadcast to reduction then all GPUs
    void configure_tensor_parallel() {
        std::array<uint32_t, 8> dest;
        dest.fill(XBAR_DEST_REDUCE);
        set_xbar_routing(dest);
        set_xbar_mode(XBAR_TENSOR);
    }

    // Configure pipeline parallel — sequential chain GPU 0→1→2→...→N
    void configure_pipeline(uint32_t num_stages) {
        std::array<uint32_t, 8> dest;
        for (uint32_t i = 0; i < 8; i++)
            dest[i] = (i < num_stages - 1) ? (i + 1) : XBAR_DEST_BROADCAST;
        set_xbar_routing(dest);
        set_xbar_mode(XBAR_PIPELINE);
    }

    // Configure MoE routing — route token to specific expert GPUs only
    // active_experts: bitmask of which GPU ports receive the token
    // e.g. 0b00101001 = GPUs 0, 3, 5 are active experts
    void configure_moe(uint8_t active_experts) {
        std::array<uint32_t, 8> dest;
        for (uint32_t i = 0; i < 8; i++) {
            if (active_experts & (1u << i))
                dest[i] = i;                   // Route to this expert GPU
            else
                dest[i] = XBAR_DEST_REDUCE;    // Inactive: route to reduction (no signal sent)
        }
        set_xbar_routing(dest);
        set_xbar_mode(XBAR_MOE);
    }

    bool xbar_ready() const {
        return (read_reg(0x20) & (1u << 2)) != 0;
    }

    void wait_xbar_ready(uint32_t timeout_ms = 100) {
        auto start = std::chrono::steady_clock::now();
        while (!xbar_ready()) {
            if (std::chrono::steady_clock::now() - start >
                std::chrono::milliseconds(timeout_ms))
                throw timeout_error("crossbar routing not applied within " +
                                    std::to_string(timeout_ms) + "ms");
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }

    // ==========================================================================

    void enable_auto_fire(bool enable) {
        uint32_t val = read_reg(reg::REDUCTION_CTRL);
        if (enable) val |=  reduction_ctrl::AUTO_FIRE_BIT;
        else        val &= ~reduction_ctrl::AUTO_FIRE_BIT;
        write_reg(reg::REDUCTION_CTRL, val);
    }

    void fire_manual() {
        uint32_t val = read_reg(reg::REDUCTION_CTRL);
        val |= reduction_ctrl::FIRE_BIT;    // Self-clearing in hardware
        write_reg(reg::REDUCTION_CTRL, val);
    }

    bool is_active() const {
        return (read_reg(reg::REDUCTION_CTRL) & reduction_ctrl::MODE_ACTIVE_BIT) != 0;
    }

    bool is_fire_done() const {
        return (read_reg(reg::REDUCTION_CTRL) & reduction_ctrl::FIRE_DONE_BIT) != 0;
    }

    bool is_cal_err() const {
        return (read_reg(reg::REDUCTION_CTRL) & reduction_ctrl::CAL_ERR_BIT) != 0;
    }

    uint32_t get_pass_index() const {
        return (read_reg(reg::REDUCTION_CTRL) & reduction_ctrl::PASS_INDEX_MASK)
               >> reduction_ctrl::PASS_INDEX_SHIFT;
    }

    uint32_t get_queue_depth() const {
        return (read_reg(reg::REDUCTION_CTRL) & reduction_ctrl::QUEUE_DEPTH_MASK)
               >> reduction_ctrl::QUEUE_DEPTH_SHIFT;
    }

    // ==========================================================================
    // Blocking wait for fire_done with timeout
    // timeout_ms: milliseconds to wait before throwing timeout_error
    // poll_us:    microseconds between polls (default 10us)
    // ==========================================================================

    ReductionResult wait_fire_done(uint32_t timeout_ms = 1000,
                                   uint32_t poll_us    = 10)
    {
        auto start = std::chrono::steady_clock::now();
        auto limit = std::chrono::milliseconds(timeout_ms);

        while (true) {
            if (is_fire_done()) {
                auto end     = std::chrono::steady_clock::now();
                auto latency = std::chrono::duration_cast<std::chrono::microseconds>
                               (end - start);

                ReductionResult result;
                result.adc_value   = 0;                // Read from result buffer
                result.pass_index  = get_pass_index();
                result.cal_err     = is_cal_err();
                result.queue_depth = get_queue_depth();
                result.latency     = latency;

                if (result.cal_err)
                    std::cerr << "[GreenCoyote] WARNING: cal_err detected "
                              << "— ADC clip during reduction\n";

                return result;
            }

            auto elapsed = std::chrono::steady_clock::now() - start;
            if (elapsed > limit)
                throw timeout_error("wait_fire_done exceeded " +
                                    std::to_string(timeout_ms) + "ms");

            std::this_thread::sleep_for(std::chrono::microseconds(poll_us));
        }
    }

    // ==========================================================================
    // Status dump — print all registers to stdout
    // ==========================================================================

    void dump_registers() const {
        uint32_t prec  = read_reg(reg::PRECISION_MODE);
        uint32_t cal   = read_reg(reg::GLOBAL_CAL);
        uint32_t trim0 = read_reg(reg::GAIN_TRIM_0_3);
        uint32_t trim1 = read_reg(reg::GAIN_TRIM_4_7);
        uint32_t ctrl  = read_reg(reg::REDUCTION_CTRL);
        uint32_t irq   = read_reg(reg::IRQ_CTRL);

        std::cout << "=== GreenCoyote-Bay Register Dump ===\n";
        std::cout << std::hex;
        std::cout << "  0x00 PRECISION_MODE : 0x" << prec  << "\n";
        std::cout << "    MODE_SEL          : 0x" << (prec & 0x7) << "\n";
        std::cout << "    NOISE_HEADROOM    : 0x" << ((prec >> 3) & 0x3) << "\n";
        std::cout << "    SAT_GUARD         : 0x" << ((prec >> 5) & 0x3) << "\n";
        std::cout << "  0x04 GLOBAL_CAL     : 0x" << cal   << "\n";
        std::cout << "    DARK_OFFSET       : 0x" << (cal & 0xFFF) << "\n";
        std::cout << "    CAL_VALID         : "   << ((cal >> 24) & 1) << "\n";
        std::cout << "    CAL_LOCK          : "   << ((cal >> 25) & 1) << "\n";
        std::cout << "  0x08 GAIN_TRIM_0_3  : 0x" << trim0 << "\n";
        std::cout << "  0x0C GAIN_TRIM_4_7  : 0x" << trim1 << "\n";
        std::cout << "  0x10 REDUCTION_CTRL : 0x" << ctrl  << "\n";
        std::cout << "    MODE_ACTIVE       : "   << ((ctrl >> 1) & 1) << "\n";
        std::cout << "    PASS_INDEX        : 0x" << ((ctrl >> 2) & 0x3) << "\n";
        std::cout << "    QUEUE_DEPTH       : "   << ((ctrl >> 4) & 0xF) << "\n";
        std::cout << "    CAL_ERR           : "   << ((ctrl >> 8) & 1) << "\n";
        std::cout << "    FIRE_DONE         : "   << ((ctrl >> 9) & 1) << "\n";
        std::cout << "    AUTO_FIRE         : "   << ((ctrl >> 10) & 1) << "\n";
        std::cout << "  0x14 IRQ_CTRL       : 0x" << irq   << "\n";
        std::cout << "    IRQ_STATUS        : "   << ((irq >> 8) & 1) << "\n";
        std::cout << std::dec;
        std::cout << "=====================================\n";
    }

private:

    // ==========================================================================
    // Model manifest loader
    // Reads model_manifest.json produced by hf_manifest_gen.py
    // Automatically configures crossbar, precision, and calibration
    // ==========================================================================

    struct ModelManifest {
        std::string model_id;
        std::string architecture;
        int         num_layers          = 0;
        int         hidden_dim          = 0;
        int         num_experts         = 0;
        int         num_experts_active  = 0;
        int         num_dense_layers    = 0;
        int         num_moe_layers      = 0;
        int         min_tile_arrays     = 0;
        int         min_crossbar_ports  = 0;
        std::string precision_required;
        bool        requires_moe_routing = false;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
    };

    // Load manifest from JSON file and configure hardware automatically
    // Returns the parsed manifest for use by the inference runtime
    ModelManifest load_model(const std::string& manifest_path) {
        ModelManifest m = parse_manifest(manifest_path);

        // Validate before touching hardware
        if (!m.errors.empty()) {
            for (const auto& e : m.errors)
                std::cerr << "[GreenCoyote] Manifest error: " << e << "\n";
            throw pilot_error("Model manifest has errors — cannot load");
        }

        if (!m.warnings.empty()) {
            for (const auto& w : m.warnings)
                std::cout << "[GreenCoyote] Warning: " << w << "\n";
        }

        std::cout << "[GreenCoyote] Loading model: " << m.model_id << "\n";
        std::cout << "[GreenCoyote] Architecture: " << m.architecture << "\n";

        // --- Step 1: Set precision mode from manifest ---
        uint32_t mode = MODE_FP8;  // default
        if      (m.precision_required == "INT4") mode = MODE_INT4;
        else if (m.precision_required == "FP4")  mode = MODE_FP4;
        else if (m.precision_required == "FP8")  mode = MODE_FP8;
        else if (m.precision_required == "BF16") mode = MODE_FP8; // BF16 runs at FP8 precision
        set_precision(mode);
        std::cout << "[GreenCoyote] Precision: " << m.precision_required
                  << " → mode 0x" << std::hex << mode << std::dec << "\n";

        // --- Step 2: Configure crossbar based on architecture ---
        if (m.requires_moe_routing) {
            // MoE model — start in tensor parallel, switch to MoE per token
            configure_tensor_parallel();
            std::cout << "[GreenCoyote] Crossbar: MoE model detected — "
                      << m.num_experts << " experts, "
                      << m.num_experts_active << " active per token\n";
        } else {
            // Dense model — tensor parallel throughout
            configure_tensor_parallel();
            std::cout << "[GreenCoyote] Crossbar: dense model — tensor parallel\n";
        }

        // --- Step 3: Verify crossbar port count ---
        if (m.min_crossbar_ports > 8) {
            throw pilot_error(
                "Model requires " + std::to_string(m.min_crossbar_ports) +
                " crossbar ports but hardware has 8");
        }

        // --- Step 4: Apply default calibration ---
        // Full calibration performed by calibrate() — this sets safe defaults
        set_dark_offset(0x010);
        set_gain_trim_all(0xFF);
        set_cal_valid(true);

        std::cout << "[GreenCoyote] Model loaded — ready for inference\n";
        std::cout << "[GreenCoyote] Layers: " << m.num_layers
                  << " (" << m.num_dense_layers << " dense, "
                  << m.num_moe_layers << " MoE)\n";

        return m;
    }

    // Route one token through the correct crossbar configuration
    // Call this per-token in the inference loop for MoE models
    // active_experts: bitmask of which expert GPU ports are active for this token
    void route_token(const ModelManifest& m,
                     int layer_index,
                     uint8_t active_experts = 0xFF) {

        bool is_moe_layer = (m.requires_moe_routing &&
                             layer_index >= m.num_dense_layers);

        if (is_moe_layer && active_experts != 0xFF) {
            // Route only to active experts for this token
            configure_moe(active_experts);
        } else {
            // Dense layer or full broadcast
            configure_tensor_parallel();
        }
    }

private:

    // Minimal JSON field extractor — no external dependency required
    // Handles the flat key-value structure of model_manifest.json
    ModelManifest parse_manifest(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw pilot_error("Cannot open manifest: " + path);

        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());

        ModelManifest m;
        m.model_id           = json_str(content,  "model_id");
        m.architecture       = json_str(content,  "architecture");
        m.precision_required = json_str(content,  "precision_required");
        m.num_layers         = json_int(content,  "num_layers");
        m.hidden_dim         = json_int(content,  "hidden_dim");
        m.num_experts        = json_int(content,  "num_experts");
        m.num_experts_active = json_int(content,  "num_experts_active");
        m.num_dense_layers   = json_int(content,  "num_dense_layers");
        m.num_moe_layers     = json_int(content,  "num_moe_layers");
        m.min_tile_arrays    = json_int(content,  "min_tile_arrays");
        m.min_crossbar_ports = json_int(content,  "min_crossbar_ports");
        m.requires_moe_routing = json_bool(content, "requires_moe_routing");

        return m;
    }

    // Simple JSON string field extractor
    static std::string json_str(const std::string& json,
                                const std::string& key) {
        std::string search = "\"" + key + "\": \"";
        auto pos = json.find(search);
        if (pos == std::string::npos) return "";
        pos += search.size();
        auto end = json.find("\"", pos);
        return json.substr(pos, end - pos);
    }

    // Simple JSON integer field extractor
    static int json_int(const std::string& json,
                        const std::string& key) {
        std::string search = "\"" + key + "\": ";
        auto pos = json.find(search);
        if (pos == std::string::npos) return 0;
        pos += search.size();
        try { return std::stoi(json.substr(pos, 20)); }
        catch (...) { return 0; }
    }

    // Simple JSON boolean field extractor
    static bool json_bool(const std::string& json,
                          const std::string& key) {
        std::string search = "\"" + key + "\": ";
        auto pos = json.find(search);
        if (pos == std::string::npos) return false;
        pos += search.size();
        return json.substr(pos, 4) == "true";
    }

    void write_reg(uint32_t offset, uint32_t value) const {
#ifdef USE_XRT
        ip_.write_register(offset, value);
#else
        if (base_ptr_ == nullptr)
            throw pilot_error("Driver not initialized");
        base_ptr_[offset / 4] = value;
#endif
    }

    uint32_t read_reg(uint32_t offset) const {
#ifdef USE_XRT
        return ip_.read_register(offset);
#else
        if (base_ptr_ == nullptr)
            throw pilot_error("Driver not initialized");
        return base_ptr_[offset / 4];
#endif
    }

    // ==========================================================================
    // Member variables
    // ==========================================================================

    bool                        use_xrt_;
    uintptr_t                   phys_base_  = 0;

#ifdef USE_XRT
    xrt::device                 device_;
    xrt::uuid                   uuid_;
    xrt::ip                     ip_;
#endif

#ifndef USE_XRT
    volatile uint32_t*          base_ptr_   = nullptr;
    int                         fd_         = -1;
    size_t                      map_size_   = 0;
#endif
};

} // namespace greencoyote


// =============================================================================
// Example usage
// =============================================================================

#ifdef GREENCOYOTE_EXAMPLE

#include <iostream>

int main() {
    using namespace greencoyote;

    try {
        // XRT path — production
        // GreenCoyotePilot pilot("greencoyote.xclbin", 0);

        // Bare-metal path — debug
        GreenCoyotePilot pilot(0xA0000000);  // Replace with actual BAR address

        // One-time initialization
        pilot.initialize(GreenCoyotePilot::MODE_FP8);

        // Load model from manifest — auto-configures crossbar and precision
        auto model = pilot.load_model("model_manifest.json");

        // Lock calibration during inference
        pilot.set_cal_lock(true);

        // Enable interrupts
        pilot.enable_irq(true, true);

        // Enable autonomous operation
        pilot.enable_auto_fire(true);

        // Dump state before inference
        pilot.dump_registers();

        // Main inference loop — manifest-driven routing
        for (int layer = 0; layer < model.num_layers; layer++) {

            // For MoE layers: get active expert bitmask from router
            // In production: router output comes from previous layer result
            uint8_t active_experts = 0xFF;  // placeholder — all experts
            if (model.requires_moe_routing && layer >= model.num_dense_layers) {
                // Router selects which experts — replace with real router output
                active_experts = 0b00101001;  // Example: experts 0, 3, 5 active
            }

            // Configure crossbar for this layer
            pilot.route_token(model, layer, active_experts);

            // Wait for reduction complete
            auto result = pilot.wait_fire_done(5000);

            std::cout << "Layer " << layer
                      << (layer >= model.num_dense_layers ? " [MoE]" : " [dense]")
                      << " reduction complete"
                      << " pass=" << result.pass_index
                      << " latency=" << result.latency.count() << "us"
                      << " cal_err=" << result.cal_err << "\n";

            pilot.clear_irq();
        }

        std::cout << "Inference complete\n";

    } catch (const timeout_error& e) {
        std::cerr << "TIMEOUT: " << e.what() << "\n";
        return 1;
    } catch (const cal_error& e) {
        std::cerr << "CALIBRATION: " << e.what() << "\n";
        return 2;
    } catch (const greencoyote::pilot_error& e) {
        std::cerr << "PILOT: " << e.what() << "\n";
        return 3;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 4;
    }

    return 0;
}

#endif // GREENCOYOTE_EXAMPLE

// =============================================================================
// Crossbar register map addition
// =============================================================================

namespace reg {
    constexpr uint32_t XBAR_ROUTING_LO  = 0x18;
    constexpr uint32_t XBAR_ROUTING_HI  = 0x1C;
    constexpr uint32_t XBAR_MODE_STATUS = 0x20;
}

namespace xbar {
    // Destination encoding per port
    constexpr uint32_t DEST_GPU0        = 0x0;
    constexpr uint32_t DEST_GPU1        = 0x1;
    constexpr uint32_t DEST_GPU2        = 0x2;
    constexpr uint32_t DEST_GPU3        = 0x3;
    constexpr uint32_t DEST_GPU4        = 0x4;
    constexpr uint32_t DEST_GPU5        = 0x5;
    constexpr uint32_t DEST_PIPELINE    = 0x5;  // Route to next in pipeline chain
    constexpr uint32_t DEST_REDUCE      = 0x6;  // Route to optical reduction stage
    constexpr uint32_t DEST_BROADCAST   = 0x7;  // Broadcast to all GPUs

    // Mode encoding
    constexpr uint32_t MODE_TENSOR      = 0x0;  // All ports broadcast — tensor parallel
    constexpr uint32_t MODE_PIPELINE    = 0x1;  // Sequential chain — pipeline parallel
    constexpr uint32_t MODE_MOE         = 0x2;  // Token routing — mixture of experts
    constexpr uint32_t MODE_CUSTOM      = 0x3;  // Arbitrary routing from register values

    constexpr uint32_t READY_BIT        = (1u << 2);
}

// Crossbar API extension — add these methods to GreenCoyotePilot class
// Usage: call set_crossbar_mode() then set_crossbar_route() per port as needed

inline void set_crossbar_mode_tensor(uint32_t base) {
    // Tensor parallel: all ports broadcast to all GPUs
    // Write broadcast to all 8 ports then set mode
    uint32_t all_broadcast = (xbar::DEST_BROADCAST << 9) |
                             (xbar::DEST_BROADCAST << 6) |
                             (xbar::DEST_BROADCAST << 3) |
                             (xbar::DEST_BROADCAST << 0);
    volatile uint32_t* regs = reinterpret_cast<volatile uint32_t*>(base);
    regs[reg::XBAR_ROUTING_LO / 4]  = all_broadcast;
    regs[reg::XBAR_ROUTING_HI / 4]  = all_broadcast;
    regs[reg::XBAR_MODE_STATUS / 4] = xbar::MODE_TENSOR;
}

inline void set_crossbar_mode_pipeline(uint32_t base) {
    // Pipeline: GPU 0→1→2→3→4→5→6→7 sequential chain
    // Port N routes to GPU N+1, last port routes to host
    volatile uint32_t* regs = reinterpret_cast<volatile uint32_t*>(base);
    uint32_t lo = (xbar::DEST_GPU3 << 9) |  // port3 → GPU4 (via reduce encoding)
                  (xbar::DEST_GPU2 << 6) |  // port2 → GPU3
                  (xbar::DEST_GPU1 << 3) |  // port1 → GPU2
                  (xbar::DEST_GPU0 << 0);   // port0 → GPU1
    uint32_t hi = (xbar::DEST_REDUCE << 9) | // port7 → reduce/host
                  (xbar::DEST_GPU6   << 6) | // port6 → GPU7
                  (xbar::DEST_GPU5   << 3) | // port5 → GPU6
                  (xbar::DEST_GPU4   << 0);  // port4 → GPU5
    regs[reg::XBAR_ROUTING_LO / 4]  = lo;
    regs[reg::XBAR_ROUTING_HI / 4]  = hi;
    regs[reg::XBAR_MODE_STATUS / 4] = xbar::MODE_PIPELINE;
}

inline void set_crossbar_moe(uint32_t base,
                              const std::array<uint32_t, 8>& expert_map) {
    // MoE: expert_map[port] = destination GPU for that token/expert assignment
    // Inactive experts get DEST_REDUCE (dropped/ignored)
    volatile uint32_t* regs = reinterpret_cast<volatile uint32_t*>(base);
    uint32_t lo = ((expert_map[3] & 0x7) << 9) |
                  ((expert_map[2] & 0x7) << 6) |
                  ((expert_map[1] & 0x7) << 3) |
                  ((expert_map[0] & 0x7) << 0);
    uint32_t hi = ((expert_map[7] & 0x7) << 9) |
                  ((expert_map[6] & 0x7) << 6) |
                  ((expert_map[5] & 0x7) << 3) |
                  ((expert_map[4] & 0x7) << 0);
    regs[reg::XBAR_ROUTING_LO / 4]  = lo;
    regs[reg::XBAR_ROUTING_HI / 4]  = hi;
    regs[reg::XBAR_MODE_STATUS / 4] = xbar::MODE_MOE;
}

inline bool crossbar_ready(uint32_t base) {
    volatile uint32_t* regs = reinterpret_cast<volatile uint32_t*>(base);
    return (regs[reg::XBAR_MODE_STATUS / 4] & xbar::READY_BIT) != 0;
}

// =============================================================================
// MoE routing example — DeepSeek R1 style
// DeepSeek R1 activates 6 of 64 experts per token
// Map active expert GPU indices to crossbar ports
// =============================================================================
//
// Example usage:
//   std::array<uint32_t, 8> expert_map = {
//       xbar::DEST_GPU2,     // port 0 → expert on GPU 2
//       xbar::DEST_GPU5,     // port 1 → expert on GPU 5
//       xbar::DEST_REDUCE,   // port 2 → inactive, drain to reduce
//       xbar::DEST_REDUCE,   // port 3 → inactive
//       xbar::DEST_GPU1,     // port 4 → expert on GPU 1
//       xbar::DEST_GPU7,     // port 5 → expert on GPU 7
//       xbar::DEST_REDUCE,   // port 6 → inactive
//       xbar::DEST_REDUCE,   // port 7 → inactive
//   };
//   set_crossbar_moe(base_addr, expert_map);
//   while (!crossbar_ready(base_addr)) { /* spin */ }
//   // Now fire inference — only GPUs 1, 2, 5, 7 activated

