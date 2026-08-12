//! The [`InferenceModel`] trait — what the application must say about a model
//! before the runtime can prepare it.
//!
//! # Why this is a trait and not just a struct
//!
//! [`ir::Model`](crate::ir::Model) is generated data: the schedule, the score
//! layout, the tensor sizes. It is produced by the schedule generator and no
//! human should edit it.
//!
//! The things in this trait are the opposite — they are *placement* decisions
//! that a human makes, and that the generator has no way to know: which region
//! of PSRAM this model's relocated blobs live in, which NOR slot holds its
//! weights, and how many prepared-schedule entries to reserve. Keeping them in
//! a trait next to a zero-sized marker type means each model states them once,
//! in one place, instead of restating them at every call site.
//!
//! It exists to delete boilerplate, not to add abstraction. Before it, adding a
//! second model meant copying ~170 lines of adapter and changing five
//! constants. With it, the adapter is the constants.
//!
//! # What it deliberately does NOT cover
//!
//! **Pre- and post-processing.** A segmentation model wants "crop and resize a
//! camera frame"; an audio model wants "window and FFT a PCM ring buffer".
//! Those have nothing in common but a byte count, so forcing them behind one
//! trait would produce an associated type that means nothing and a set of
//! implementations that ignore each other. Modality-specific work lives next to
//! the model, in the application, where it can use whatever shape actually
//! fits.
//!
//! This trait is about *the schedule and its bindings* — the part that really
//! is the same for every model on this NPU, whatever the tensors mean. That is
//! also why it is named `InferenceModel` rather than anything vision-flavoured.

use crate::ir::{Model, Region};

/// A compiled network, plus where its pieces live.
///
/// Implemented on a zero-sized marker type per model:
///
/// ```ignore
/// pub struct Deeplab;
///
/// impl InferenceModel for Deeplab {
///     const BLOB_RAM:     Region = Region::new(0x9060_0000, 0x0002_0000);
///     const WEIGHTS_BASE: u32    = 0x7038_0000;
///     const MAX_OPS:      usize  = 64;
///
///     fn descriptor() -> &'static Model { &generated::MODEL }
/// }
/// ```
pub trait InferenceModel {
    /// PSRAM region for this model's relocated EC-blob containers.
    ///
    /// Per-model and NOT shareable, unlike the tensor arena: the blobs are
    /// patched with this program's IO addresses during `prepare`, so two models
    /// pointed at the same region would overwrite each other's bindings and
    /// the second one to run would scribble into the first one's buffers.
    ///
    /// Tens of KiB, not hundreds — only the hardware epochs are copied here.
    const BLOB_RAM: Region;

    /// Base address of this model's weight slot in NOR.
    ///
    /// Must match the slot the image was flashed to, because stedgeai bakes
    /// this address into the compiled blob. `prepare` cross-checks it against
    /// the flash manifest and refuses to run on a mismatch rather than letting
    /// the NPU fetch another model's weights.
    const WEIGHTS_BASE: u32;

    /// Upper bound on prepared-schedule entries, for sizing the caller's store.
    ///
    /// Compare against `descriptor().ops.len()` — the prepared schedule has
    /// exactly one entry per IR op. Round up; the slack is a few bytes each.
    const MAX_OPS: usize;

    /// The compiled schedule.
    ///
    /// `descriptor().id` is the lookup key into the flash weight manifest, so
    /// it must match the `[[model]]` name in memory.toml and the `--id` passed
    /// to the schedule generator. That string is the only thing tying the
    /// schedule compiled into this binary to the image sitting in NOR.
    fn descriptor() -> &'static Model;
}
