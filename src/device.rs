//! The NPU as a singleton resource.
//!
//! There is one Neural-ART accelerator, one activation arena and one set of
//! CACHEAXI counters on this part. [`NpuDevice`] owns all three. Everything
//! that is per-*model* — relocated blobs, IO bindings — lives in a
//! [`Program`](crate::program::Program) instead.
//!
//! That split is the whole point of this crate. It means N models cost N sets
//! of relocated blobs (tens of KiB each) rather than N arenas (hundreds of KiB
//! each), and it makes the sharing rule enforceable rather than conventional:
//! [`NpuDevice::run`] takes `&mut self`, so the borrow checker guarantees that
//! only one program is mid-dispatch at a time. Since programs share the arena,
//! and the arena is scribbled over by whichever schedule is running, that
//! exclusivity is not a nicety — two concurrent dispatches would silently
//! corrupt each other's activations.
//!
//! # Scheduling
//!
//! Inference is not preemptible. An epoch, once dispatched, runs to completion;
//! there is no way to suspend it and no way to save its state. A model that
//! needs to meet a deadline must therefore fit in the gaps left by the others,
//! which is a budgeting problem the caller owns:
//!
//! ```text
//! sum_i (hw_us_i * rate_i) < 0.85 * 1e6   microseconds per second
//! ```
//!
//! Leave headroom. The bound above ignores the CPU-side operators, which on a
//! debug build can exceed the hardware time.

use defmt::info;
use embassy_stm32::npu::{self, Npu};
use embassy_stm32::peripherals::NPU;
use embassy_time::{with_timeout, Duration, Instant};

use crate::error::InferError;
use crate::program::{Prepared, Program};
use crate::stats::Stats;
use crate::sw;

/// Device-wide settings.
#[derive(Clone, Copy)]
pub struct DeviceConfig {
    /// Per-epoch dispatch timeout.
    ///
    /// This is a deadlock escape, not a scheduling parameter — set it well
    /// above the slowest epoch. DeepLab's ASPP branches alone run tens of
    /// milliseconds, so the default is deliberately generous.
    pub epoch_timeout: Duration,
    /// Read the CACHEAXI hit/miss counters around each dispatch.
    ///
    /// Cheap (two register reads per inference) and the single most useful
    /// signal when a model's timing regresses: a jump in misses means weight or
    /// activation traffic started going out to NOR or PSRAM.
    pub cache_counters: bool,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            epoch_timeout: Duration::from_millis(4000),
            cache_counters: true,
        }
    }
}

/// Exclusive handle to the Neural-ART accelerator.
pub struct NpuDevice {
    npu: Npu<'static, NPU>,
    cfg: DeviceConfig,
}

impl NpuDevice {
    /// Bring up the NPU.
    ///
    /// `make_driver` is a closure rather than a peripheral or a pre-built
    /// driver because the bring-up order is load-bearing and split across two
    /// parties:
    ///
    /// 1. RIF must grant the NPU master and its RISC slave the CPU's
    ///    security/privilege attributes **before** the driver exists.
    /// 2. The driver ungates the NPU kernel clock.
    /// 3. CACHEAXI can only be touched **after** that clock is running —
    ///    register accesses fault while it is gated.
    ///
    /// Step 2 has to happen in the application, because `Npu::new` needs an
    /// interrupt binding and `bind_interrupts!` defines the handler symbol,
    /// which only the binary crate can do. Handing in a closure lets this
    /// function keep steps 1 and 3 on either side of it:
    ///
    /// ```ignore
    /// let device = NpuDevice::new(&DeviceConfig::default(), || {
    ///     Npu::new(unsafe { Peripherals::steal() }.NPU, NpuIrqs)
    /// });
    /// ```
    ///
    /// Idempotent in practice: the RIF and CACHEAXI writes are all
    /// set-to-the-same-value, so re-running after a device drop is harmless.
    pub fn new<F>(cfg: &DeviceConfig, make_driver: F) -> Self
    where
        F: FnOnce() -> Npu<'static, NPU>,
    {
        // ── RIF: NPU master + NPU RISC slave, matching the CPU's sec/priv ──
        // Mirrors ST's `Security_Config()`. Without this the NPU's bus
        // transactions are filtered and the first weight fetch faults.
        {
            use embassy_stm32::pac::RIFSC;
            RIFSC.risc_seccfgr(3).modify(|w| w.set_cfg(10, true));
            RIFSC.risc_privcfgr(3).modify(|w| w.set_cfg(10, true));
            RIFSC.rimc_attr(1).modify(|w| {
                w.set_mcid(1);
                w.set_msec(true);
                w.set_mpriv(true);
            });
        }

        let npu = make_driver();

        // ── CACHEAXI, strictly after the NPU kernel clock ──────────────────
        embassy_stm32::pac::RCC
            .memenr()
            .modify(|w| w.set_npucacheramen(true));
        npu::cache::npu_cache_enable();
        npu::cache::npu_cache_monitor_reset();
        let (cr1, sr) = npu::cache::npu_cache_debug_state();
        info!("NPU CACHEAXI: CR1={=u32:#010x} SR={=u32:#010x}", cr1, sr);

        Self { npu: npu, cfg: *cfg }
    }

    pub fn config(&self) -> &DeviceConfig {
        &self.cfg
    }

    /// Run one inference.
    ///
    /// The caller has already written `input.len()` bytes to the address bound
    /// in `program`'s [`IoBinding`](crate::ir::IoBinding); this function handles
    /// the cache maintenance on either side and leaves the output tensor
    /// readable at the bound output address.
    ///
    /// `output` is checked for length but is not necessarily written: when it
    /// already *is* the bound output buffer — the common case, since the caller
    /// usually binds the buffer it intends to read — copying would be a
    /// same-range `memcpy`, which is undefined behaviour under stacked borrows.
    /// [`Program::output_is`] lets the caller tell the two cases apart.
    pub async fn run(
        &mut self,
        program: &mut Program<'_>,
        input: &[u8],
        output: &mut [i8],
    ) -> Result<Stats, InferError> {
        let model = program.model();

        if input.len() != model.input_bytes {
            return Err(InferError::InputLen {
                got: input.len(),
                want: model.input_bytes,
            });
        }
        // The relocations baked the input address into the blobs at prepare
        // time. A different buffer here would be read from the old address —
        // silently, and with whatever happened to be left there.
        if input.as_ptr() as u32 != program.io().input {
            return Err(InferError::InputMoved {
                got: input.as_ptr() as u32,
                bound: program.io().input,
            });
        }
        let want_out = model.scores.count();
        if output.len() != want_out {
            return Err(InferError::OutputLen {
                got: output.len(),
                want: want_out,
            });
        }

        // Publish the input tile to the NPU's view, and drop any cached copy
        // of the output so the post-run read does not hit stale lines.
        npu::cache::mcu_clean_range(input.as_ptr() as u32, input.len() as u32);
        npu::cache::mcu_invalidate_range(program.io().output, output.len() as u32);

        let mut stats = Stats::default();
        let (hits0, misses0) = if self.cfg.cache_counters {
            npu::cache::npu_cache_read_counters()
        } else {
            (0, 0)
        };

        for (i, op) in program.ops().iter().enumerate() {
            match op {
                Prepared::Hw(blob) => {
                    let t0 = Instant::now();
                    let result =
                        with_timeout(self.cfg.epoch_timeout, self.npu.run_epoch_blob(blob)).await;
                    stats.hw_us = stats.hw_us.wrapping_add(t0.elapsed().as_micros() as u32);
                    match result {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => {
                            defmt::warn!("NPU: op {=usize} epoch error: {}", i, e);
                            return Err(InferError::EpochFailed { op: i });
                        }
                        Err(_) => return Err(InferError::EpochTimeout { op: i }),
                    }
                    stats.hw_ops += 1;
                }
                Prepared::Sw(op) => {
                    let t0 = Instant::now();
                    let r = sw::run(op);
                    stats.sw_us = stats.sw_us.wrapping_add(t0.elapsed().as_micros() as u32);
                    r.map_err(|cause| InferError::Sw { op: i, cause })?;
                    stats.sw_ops += 1;
                }
                Prepared::CacheClean { addr, size } => npu::cache::mcu_clean_range(*addr, *size),
                Prepared::CacheInvalidate { addr, size } => {
                    npu::cache::mcu_invalidate_range(*addr, *size)
                }
                // Unreachable: `Program` only ever hands out its populated
                // prefix. Treated as a no-op rather than a panic.
                Prepared::Empty => {}
            }
        }

        // Invalidate again after the last epoch — the final tensor was DMA'd
        // in behind the CPU's back.
        npu::cache::mcu_invalidate_range(program.io().output, output.len() as u32);

        if self.cfg.cache_counters {
            let (hits1, misses1) = npu::cache::npu_cache_read_counters();
            stats.cache_read_hits = hits1.wrapping_sub(hits0);
            stats.cache_read_misses = misses1.wrapping_sub(misses0);
        }

        Ok(stats)
    }
}
