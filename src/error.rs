//! Error types.
//!
//! Split by phase, because the two have completely different audiences:
//! [`InitError`] is almost always a build or provisioning mistake and is read
//! by whoever is bringing the board up, while [`InferError`] is a runtime fault
//! and is read from a log after the fact.

use crate::weights::WeightError;

/// Something went wrong preparing a [`crate::program::Program`].
///
/// Every variant carries enough context to act on without attaching a
/// debugger — the numbers are the ones you would otherwise have to go and
/// measure.
#[derive(Clone, Copy, defmt::Format)]
pub enum InitError {
    /// The generated schedule is empty. The code generator has not been run, or
    /// ran against an empty `st_ai_output/`.
    EmptySchedule,
    /// The schedule has more ops than the caller's `Prepared` store can hold.
    StoreTooSmall { needed: usize, have: usize },
    /// A hardware epoch's blob array is zero-length.
    EmptyBlob { op: usize },
    /// A bare blob (no EC container) carries relocations, which cannot be
    /// applied because there is no relocation table to apply them to. The
    /// generator and the runtime disagree about the blob format.
    BareBlobHasRelocs { op: usize },
    /// The blob is neither a bare blob nor a parseable EC container.
    BadEcBinary { op: usize },
    /// The EC container's blob section is malformed.
    BadBlobSection { op: usize },
    /// Copying the blob out of the container failed.
    BlobCopyFailed { op: usize },
    /// A relocation symbol named by the schedule is absent from the container.
    RelocFailed { op: usize },
    /// The relocated blobs do not fit in the blob RAM region the caller
    /// provided. `needed` is the running total at the point of overflow, so it
    /// is a lower bound, not the final figure.
    BlobRamExhausted { needed: u32, have: u32 },
    /// The weight image named by `Model::id` failed verification.
    Weights(WeightError),
}

/// Something went wrong during a dispatch.
#[derive(Clone, Copy, defmt::Format)]
pub enum InferError {
    /// The caller passed an input slice of the wrong length.
    InputLen { got: usize, want: usize },
    /// The caller's input slice is not the buffer bound at prepare time. The
    /// relocations point at the original address, so running would read
    /// whatever is still there.
    InputMoved { got: u32, bound: u32 },
    /// The caller passed an output slice of the wrong length.
    OutputLen { got: usize, want: usize },
    /// The NPU reported an error completing an epoch.
    EpochFailed { op: usize },
    /// An epoch did not complete within the configured timeout.
    EpochTimeout { op: usize },
    /// A CPU-side operator rejected its arguments.
    Sw { op: usize, cause: SwError },
}

/// Why a software operator refused to run.
///
/// All of these are schedule/generator bugs rather than runtime conditions —
/// the shapes come from `stedgeai` and do not vary between inferences.
#[derive(Clone, Copy, defmt::Format)]
pub enum SwError {
    /// `SpaceToDepth` input dimensions are not divisible by the block size.
    S2dNotDivisible,
    /// `DepthToSpace` channel count is not divisible by `bs_h * bs_w`.
    D2sNotDivisible,
    /// Element width other than 1 or 2 bytes.
    UnsupportedElemBytes { got: u32 },
    /// A resize output extent exceeds the operator's fixed coordinate tables.
    /// Raising `sw::MAX_RESIZE_EXTENT` is the fix.
    ResizeTooLarge { extent: u32 },
}
