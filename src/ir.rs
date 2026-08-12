//! Intermediate representation of a compiled Neural-ART schedule.
//!
//! Everything here is what `stedgeai` produced, transcribed by
//! `scripts/deeplab_schedule_to_rust.py` (and its siblings) into `static`
//! data. Nothing in this module allocates, executes, or knows about the
//! peripheral — it is the contract between the code generator and
//! [`crate::device`].
//!
//! A [`Model`] is `'static` and immutable, which is the whole point: the
//! expensive per-model state (relocated blobs) lives in a caller-owned
//! [`crate::program::Program`], so several models can share one device and
//! one activation arena without the schedule data being duplicated per
//! instance.

/// Which caller-supplied IO buffer a relocation binds to.
///
/// `stedgeai` emits relocation symbols (`_user_io_input_0`,
/// `_user_io_output_0`) whose addresses are only known at run time. The
/// generator records which of the two roles each symbol plays; the loader
/// substitutes the actual address at prepare time.
#[derive(Clone, Copy, PartialEq, Eq, defmt::Format)]
pub enum Io {
    Input,
    Output,
}

/// One EC-blob container together with its symbol-name relocations.
///
/// `data` is either a bare blob (starts with `BLOB_MAGIC`, executable
/// straight from flash, no relocs possible) or an EC container that must be
/// parsed, copied into RAM and patched before dispatch.
pub struct Epoch {
    pub data:   &'static [u64],
    pub relocs: &'static [(&'static str, Io)],
}

/// A CPU-side operator interleaved between NPU hardware epochs.
///
/// These exist because the Neural-ART hardware cannot express certain
/// operators natively — dilated convolution becomes a
/// `SpaceToDepth → Conv → DepthToSpace` triple, and bilinear resize has no
/// hardware form at all. `stedgeai` emits them as library calls; we
/// re-implement them bit-exactly on the M55.
#[derive(Clone, Copy, defmt::Format)]
pub enum SwOp {
    SpaceToDepth {
        in_addr:  u32,
        in_h:     u32,
        in_w:     u32,
        in_c:     u32,
        out_addr: u32,
        bs_h:     u32,
        bs_w:     u32,
        elem_bytes: u32,
    },
    DepthToSpace {
        in_addr:  u32,
        in_h:     u32,
        in_w:     u32,
        in_c:     u32,
        out_addr: u32,
        bs_h:     u32,
        bs_w:     u32,
        elem_bytes: u32,
    },
    ResizeLinearHp {
        in_addr:  u32,
        in_h:     u32,
        in_w:     u32,
        in_c:     u32,
        out_addr: u32,
        out_h:    u32,
        out_w:    u32,
        is_signed: bool,
    },
}

/// One entry of a compiled schedule, in dispatch order.
///
/// The cache fences are transcribed 1:1 from the `LL_ATON_End_EpochBlock_NN`
/// bodies in ST's generated `network.c`. They are frequently no-ops given the
/// arena's Non-Cacheable MPU mapping, but they are preserved verbatim so that
/// changing the MPU policy does not silently introduce coherency bugs.
pub enum Op {
    Hw(Epoch),
    Sw(SwOp),
    CacheInvalidate { addr: u32, size: u32 },
    CacheClean { addr: u32, size: u32 },
}

/// Where the final output tensor lands.
#[derive(Clone, Copy, defmt::Format)]
pub enum Scores {
    /// At a fixed activation-arena address baked into the schedule. The
    /// caller's output binding is ignored.
    RawInt8 { addr: u32, count: usize },
    /// In the caller-supplied buffer, via a `_user_io_output_0` relocation on
    /// the last hardware epoch.
    UserInt8 { count: usize },
}

impl Scores {
    /// Byte count of the output tensor, whichever form it takes.
    pub const fn count(&self) -> usize {
        match *self {
            Scores::RawInt8 { count, .. } => count,
            Scores::UserInt8 { count } => count,
        }
    }
}

/// A complete compiled network.
///
/// Produced entirely by the code generator and placed in `.rodata`. `id` must
/// match the entry in the flash weight manifest — that string is the only link
/// between the schedule baked into the binary and the weight image sitting in
/// NOR, so a typo here surfaces as a refusal to run rather than a wrong answer.
pub struct Model {
    pub id:           &'static str,
    pub ops:          &'static [Op],
    pub scores:       Scores,
    pub input_bytes:  usize,
    pub output_bytes: usize,
}

impl Model {
    /// Number of hardware epochs in the schedule. Used to size blob RAM and to
    /// sanity-check dispatch.
    pub fn hw_epochs(&self) -> usize {
        self.ops.iter().filter(|o| matches!(o, Op::Hw(_))).count()
    }
}

/// A half-open byte range `[base, base + len)`.
#[derive(Clone, Copy, defmt::Format)]
pub struct Region {
    pub base: u32,
    pub len:  u32,
}

impl Region {
    pub const fn new(base: u32, len: u32) -> Self {
        Self { base, len }
    }

    pub const fn end(&self) -> u32 {
        self.base + self.len
    }
}

/// Physical addresses of the caller-owned input and output tensors.
///
/// These are raw addresses rather than slices deliberately: both buffers are
/// concurrently visible to the NPU, so holding a Rust reference across a
/// dispatch would be claiming exclusivity we do not have.
#[derive(Clone, Copy, defmt::Format)]
pub struct IoBinding {
    pub input:  u32,
    pub output: u32,
}

/// An `include_bytes!`-loaded epoch blob, forced to `u64` alignment.
///
/// # Why this exists
///
/// Epoch blobs are `u64` word streams. Emitting them as `static [u64; N]`
/// literals works, but costs a few thousand lines of hex per model for rustc to
/// tokenise and for a human to scroll past. `include_bytes!` avoids both — it
/// hands the file to the linker without parsing it.
///
/// The catch is that `include_bytes!` produces `[u8; N]`, whose alignment is 1.
/// Reinterpreting that as `[u64]` is undefined behaviour and, on this target,
/// practically wrong too: the NPU reads these words over AXI and an unaligned
/// base would fault or silently fetch the wrong bytes. `#[repr(C, align(8))]`
/// restores the alignment the literal array had for free, at compile time.
#[repr(C, align(8))]
pub struct Blob<const N: usize>(pub [u8; N]);

/// View an aligned blob as the `u64` word stream the dispatcher expects.
///
/// `const`, so the result can be baked straight into a `static [Op]` and the
/// blob never needs a runtime fix-up pass.
///
/// # Panics
///
/// At compile time, if `N` is not a multiple of 8. Every blob stedgeai emits is
/// a whole number of `u64` words, so a failure here means the `.bin` is
/// truncated or the wrong file was included.
pub const fn blob_words<const N: usize>(b: &'static Blob<N>) -> &'static [u64] {
    assert!(N % 8 == 0, "blob length is not a whole number of u64 words");
    // SAFETY: `Blob` is `align(8)`, so the pointer is correctly aligned for
    // `u64`; `N % 8 == 0` was just checked, so `N / 8` words lie entirely
    // within the array; and the returned lifetime is the `'static` one of the
    // borrow, so the data outlives the slice.
    unsafe { core::slice::from_raw_parts(b.0.as_ptr().cast::<u64>(), N / 8) }
}
