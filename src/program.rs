//! A model prepared for dispatch.
//!
//! Preparation is the expensive, once-per-model half of running a network:
//! parse each EC container, copy its blob into RAM, patch the relocations that
//! name the caller's IO buffers, and publish the result to the NPU's view of
//! memory. Dispatch is then a walk over the result.
//!
//! # Why the store is caller-supplied
//!
//! [`Program::prepare`] writes into a `&'a mut [Prepared]` the caller provides
//! rather than a `heapless::Vec<_, N>` it owns. That keeps the schedule
//! capacity out of the type, which matters as soon as there is more than one
//! model: a `Program<64>` and a `Program<32>` are different types, so every
//! function touching them — including the dispatcher — would be monomorphised
//! per model. With a slice there is one `Program` and one dispatcher no matter
//! how many networks the firmware carries.
//!
//! # Blob RAM
//!
//! Relocated blobs must live somewhere writable and coherent with the NPU. Each
//! program gets its own region, sized by the caller, because the blobs are
//! *patched* with that program's IO addresses — they cannot be shared between
//! programs the way the activation arena can. This is the per-model cost of
//! multi-model support, and it is tens of KiB, not hundreds.

use defmt::info;
use embassy_stm32::npu;

use crate::error::InitError;
use crate::ir::{Io, IoBinding, Model, Op, Region, SwOp};
use crate::model::InferenceModel;
use crate::weights::{self, ManifestLocation, WeightInfo};

/// One entry of a prepared schedule.
///
/// `Empty` exists so the caller can allocate the backing store with a `const`
/// initialiser; [`Program`] never dispatches it.
#[derive(Clone, Copy)]
pub enum Prepared {
    Empty,
    /// A relocated blob in blob RAM, or a bare blob still in flash.
    Hw(&'static [u64]),
    Sw(SwOp),
    CacheClean { addr: u32, size: u32 },
    CacheInvalidate { addr: u32, size: u32 },
}

impl Prepared {
    /// `const` initialiser for the backing store: `[Prepared::EMPTY; N]`.
    pub const EMPTY: Prepared = Prepared::Empty;
}

/// Everything [`Program::prepare`] needs beyond the model itself.
pub struct PrepareConfig {
    /// Addresses of the caller-owned input and output tensors. Baked into the
    /// blobs by relocation, so they are fixed for the life of the program.
    pub io: IoBinding,
    /// Writable, NPU-visible region for the relocated blobs. Must not overlap
    /// another program's.
    pub blob_ram: Region,
    /// Where the weight manifest lives.
    pub manifest: ManifestLocation,
    /// The weight-slot address this build was compiled against. Cross-checked
    /// against the manifest so a stale flash is caught rather than executed.
    pub weights_base: u32,
    /// CRC the entire weight image instead of only scanning for erasure.
    ///
    /// Off is the right default: the image is large and read over single-SPI
    /// XIP, so a full pass costs real boot time on every reset, and it defends
    /// against bit-rot that has never been observed here. The failure that
    /// *has* been observed — an erased slot — is caught either way. Turn it on
    /// when commissioning a board or chasing a suspect part.
    pub verify_weights_crc: bool,
}

/// A model, bound to buffers and ready to dispatch.
pub struct Program<'a> {
    model: &'static Model,
    ops: &'a mut [Prepared],
    io: IoBinding,
    weights: WeightInfo,
    blob_ram_used: u32,
}

impl<'a> Program<'a> {
    /// Prepare a model that has stated its own placement via
    /// [`InferenceModel`].
    ///
    /// Equivalent to building a [`PrepareConfig`] by hand and calling
    /// [`prepare`](Self::prepare); it exists so that adding a model does not
    /// mean restating the same five addresses at the call site.
    ///
    /// The generic parameter is only read for associated constants, so this
    /// monomorphises to a handful of immediate loads per model. The dispatcher
    /// it calls into stays non-generic — that is the whole reason `store` is a
    /// slice rather than a const-generic array.
    pub fn prepare_model<M: InferenceModel>(
        io: IoBinding,
        manifest: ManifestLocation,
        verify_weights_crc: bool,
        store: &'a mut [Prepared],
    ) -> Result<Self, InitError> {
        let cfg = PrepareConfig {
            io,
            blob_ram: M::BLOB_RAM,
            manifest,
            weights_base: M::WEIGHTS_BASE,
            verify_weights_crc,
        };
        Self::prepare(M::descriptor(), &cfg, store)
    }

    /// Verify the model's weights, relocate its blobs, and bind its IO.
    ///
    /// `store` must be at least `model.ops.len()` long. On success the program
    /// takes ownership of exactly that prefix.
    pub fn prepare(
        model: &'static Model,
        cfg: &PrepareConfig,
        store: &'a mut [Prepared],
    ) -> Result<Self, InitError> {
        if model.ops.is_empty() {
            return Err(InitError::EmptySchedule);
        }
        if store.len() < model.ops.len() {
            return Err(InitError::StoreTooSmall {
                needed: model.ops.len(),
                have: store.len(),
            });
        }

        // ── Weights ────────────────────────────────────────────────────────
        // The CPU never touches the weight image — the NPU fetches it over
        // XSPI2 on its own — so an erased or stale slot is invisible at every
        // other layer. It produces a clean run with a blank output. Check
        // before dispatching, and refuse rather than emit a plausible answer.
        let weights = weights::verify_and_log(
            &cfg.manifest,
            model.id,
            cfg.weights_base,
            cfg.verify_weights_crc,
        )
        .map_err(InitError::Weights)?;

        let store = &mut store[..model.ops.len()];
        let mut next_blob = cfg.blob_ram.base;
        let mut hw_index = 0u32;

        for (i, op) in model.ops.iter().enumerate() {
            store[i] = match op {
                Op::Sw(sw) => Prepared::Sw(*sw),
                Op::CacheClean { addr, size } => Prepared::CacheClean {
                    addr: *addr,
                    size: *size,
                },
                Op::CacheInvalidate { addr, size } => Prepared::CacheInvalidate {
                    addr: *addr,
                    size: *size,
                },
                Op::Hw(ep) => {
                    let first = match ep.data.first() {
                        Some(&w) => w as u32,
                        None => return Err(InitError::EmptyBlob { op: i }),
                    };

                    if first == npu::ecloader::BLOB_MAGIC {
                        // Bare blob: dispatched straight from flash. There is
                        // no relocation table, so a schedule that asks for one
                        // here is internally inconsistent.
                        if !ep.relocs.is_empty() {
                            return Err(InitError::BareBlobHasRelocs { op: i });
                        }
                        hw_index += 1;
                        Prepared::Hw(ep.data)
                    } else {
                        let bin = npu::ecloader::EcBinary::new(ep.data)
                            .map_err(|_| InitError::BadEcBinary { op: i })?;
                        let words = bin
                            .blob_len()
                            .map_err(|_| InitError::BadBlobSection { op: i })?;

                        // Cache-line align each copy so a clean of one blob
                        // cannot write back a stale line belonging to another.
                        next_blob = (next_blob + 63) & !63;
                        let bytes = (words * 8) as u32;
                        if next_blob + bytes > cfg.blob_ram.end() {
                            return Err(InitError::BlobRamExhausted {
                                needed: next_blob + bytes - cfg.blob_ram.base,
                                have: cfg.blob_ram.len,
                            });
                        }

                        // SAFETY: the region is caller-declared as writable,
                        // NPU-visible and exclusively owned by this program;
                        // the bounds check above keeps the write inside it.
                        let dst: &'static mut [u64] = unsafe {
                            core::slice::from_raw_parts_mut(next_blob as *mut u64, words)
                        };
                        bin.load_blob(dst)
                            .map_err(|_| InitError::BlobCopyFailed { op: i })?;

                        for &(id, io) in ep.relocs {
                            let base = match io {
                                Io::Input => cfg.io.input,
                                Io::Output => cfg.io.output,
                            };
                            let mut prev = 0u32;
                            if bin.reloc_by_id(dst, id, base, &mut prev).is_err() {
                                defmt::warn!(
                                    "NPU: op {=usize} relocation '{=str}' not found in container",
                                    i,
                                    id
                                );
                                return Err(InitError::RelocFailed { op: i });
                            }
                        }

                        // Publish the patched blob to the NPU's view.
                        npu::cache::mcu_clean_range(next_blob, bytes);

                        // SAFETY: same region, now immutable for the life of
                        // the program.
                        let ro: &'static [u64] = unsafe {
                            core::slice::from_raw_parts(next_blob as *const u64, words)
                        };
                        next_blob += bytes;
                        hw_index += 1;
                        Prepared::Hw(ro)
                    }
                }
            };
        }

        let blob_ram_used = next_blob - cfg.blob_ram.base;
        info!(
            "NPU: prepared '{=str}' — {=usize} ops ({=u32} HW), {=u32} B of {=u32} B blob RAM",
            model.id,
            store.len(),
            hw_index,
            blob_ram_used,
            cfg.blob_ram.len,
        );

        Ok(Self {
            model,
            ops: store,
            io: cfg.io,
            weights,
            blob_ram_used,
        })
    }

    pub fn model(&self) -> &'static Model {
        self.model
    }

    pub fn io(&self) -> IoBinding {
        self.io
    }

    /// The verified weight image backing this program.
    pub fn weights(&self) -> WeightInfo {
        self.weights
    }

    /// Bytes of blob RAM actually consumed. Compare against the region size
    /// when tuning the budget.
    pub fn blob_ram_used(&self) -> u32 {
        self.blob_ram_used
    }

    /// Whether `buf` is the output buffer bound at prepare time.
    ///
    /// Callers that intend to copy the result somewhere else must check this:
    /// when it is `true` the copy would be a same-range `memcpy`, which is
    /// undefined behaviour, and pointless besides.
    pub fn output_is(&self, buf: *const i8) -> bool {
        buf as u32 == self.io.output
    }

    pub(crate) fn ops(&self) -> &[Prepared] {
        self.ops
    }
}
