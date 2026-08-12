//! Per-inference measurements.

/// Timings and cache counters for one [`crate::device::NpuDevice::run`].
///
/// Returned rather than logged, so the caller decides the sampling policy —
/// logging every inference at 15 Hz swamps RTT and perturbs the thing being
/// measured.
#[derive(Clone, Copy, Default, defmt::Format)]
pub struct Stats {
    /// Wall time inside `run_epoch_blob`, summed across hardware epochs. This
    /// is silicon time plus interrupt latency; it does not move with the build
    /// profile, which makes it the number worth tracking for regressions.
    pub hw_us: u32,
    /// Wall time inside CPU-side operators. Unlike `hw_us` this is very
    /// sensitive to `opt-level` — expect it to fall by an order of magnitude
    /// between dev and release.
    pub sw_us: u32,
    pub hw_ops: u32,
    pub sw_ops: u32,
    /// CACHEAXI read hits/misses across the whole schedule. A miss here is a
    /// weight or activation fetch that went out to NOR or PSRAM.
    pub cache_read_hits:   u32,
    pub cache_read_misses: u32,
}

impl Stats {
    /// Total time attributable to the schedule itself.
    pub fn total_us(&self) -> u32 {
        self.hw_us.wrapping_add(self.sw_us)
    }
}
