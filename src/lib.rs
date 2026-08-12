//! Multi-model runtime for the STM32N6 Neural-ART NPU.
//!
//! One device, many compiled schedules, one shared activation arena.
//!
//! # The shape of the problem
//!
//! `stedgeai` compiles a network into a sequence of *epochs*: opaque binary
//! blobs the NPU executes directly, interleaved with CPU-side operators for
//! everything the hardware cannot express, plus cache fences between them. It
//! also allocates an activation arena and bakes those addresses into the blobs.
//!
//! That last part is what makes multi-model interesting. Every model compiled
//! against the same memory pools uses the *same* arena addresses, so they
//! cannot run concurrently — but they also cost nothing to coexist, because the
//! arena is scratch space that is dead between inferences. The per-model cost
//! is only the relocated blobs.
//!
//! So the split is:
//!
//! * [`NpuDevice`] — the peripheral, the arena, the cache counters. Exactly
//!   one. `run` takes `&mut self`, which is what makes arena sharing sound.
//! * [`Program`] — one model bound to its buffers, with its blobs relocated
//!   into its own small region. As many as you like.
//! * [`ir::Model`] — the compiled schedule itself, `'static` and shared.
//!
//! # Usage
//!
//! ```ignore
//! let mut device = NpuDevice::new(&DeviceConfig::default(), || {
//!     Npu::new(unsafe { Peripherals::steal() }.NPU, NpuIrqs)
//! });
//!
//! let mut store = [Prepared::EMPTY; 64];
//! let mut program = Program::prepare(&MODEL, &cfg, &mut store)?;
//!
//! let stats = device.run(&mut program, input, output).await?;
//! ```
//!
//! # What this crate does not do
//!
//! It does not own the MPU configuration for the arena, does not map XSPI2, and
//! does not preprocess or postprocess tensors. Those are board and application
//! concerns and they differ per demo; putting them here would make the crate
//! specific to one product.
//!
//! It also does not schedule. Inference is not preemptible — an epoch runs to
//! completion once dispatched — so meeting deadlines across several models is a
//! budgeting exercise the application has to do with its eyes open. See
//! [`device`] for the arithmetic.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod device;
pub mod error;
pub mod ir;
pub mod model;
pub mod program;
pub mod stats;
pub mod weights;

mod sw;

pub use device::{DeviceConfig, NpuDevice};
pub use error::{InferError, InitError, SwError};
pub use ir::{Blob, Epoch, Io, IoBinding, Model, Op, Region, Scores, SwOp, blob_words};
pub use model::InferenceModel;
pub use program::{PrepareConfig, Prepared, Program};
pub use stats::Stats;
pub use weights::{ManifestLocation, WeightError, WeightInfo};
