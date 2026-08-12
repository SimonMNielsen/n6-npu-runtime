//! CPU-side operators interleaved between NPU hardware epochs.
//!
//! `stedgeai` emits these as calls into ST's `LL_ATON_LIB`; we re-implement
//! them on the M55 so the whole schedule stays in Rust. They are bit-exact
//! ports, not reimplementations from the operator spec — the reference is
//! whatever `network.c` actually calls.
//!
//! # Memory model
//!
//! Every source and destination lives in the NPU activation arena, mapped
//! Normal Non-Cacheable Outer-Shareable so the M55 D-cache does not shadow NPU
//! writes. The schedule's cache fences are still executed around these ops (see
//! [`crate::ir::Op`]) even though they are largely no-ops under that mapping —
//! dropping them would make a future change to the MPU policy silently
//! incorrect.

mod depth;
mod resize;

use crate::error::SwError;
use crate::ir::SwOp;

/// Largest output extent the resize operator can handle in either axis.
///
/// The per-column coordinate tables are stack arrays; this bounds them. The
/// current DeepLab schedule tops out at 256.
pub(crate) const MAX_RESIZE_EXTENT: usize = 512;

/// Execute one software operator in place on the activation arena.
pub(crate) fn run(op: &SwOp) -> Result<(), SwError> {
    match *op {
        SwOp::SpaceToDepth {
            in_addr,
            in_h,
            in_w,
            in_c,
            out_addr,
            bs_h,
            bs_w,
            elem_bytes,
        } => {
            if in_h % bs_h != 0 || in_w % bs_w != 0 {
                return Err(SwError::S2dNotDivisible);
            }
            // SAFETY: the shapes come from the compiled schedule and describe
            // regions stedgeai allocated in the arena; the divisibility
            // precondition is checked above.
            match elem_bytes {
                1 => unsafe {
                    depth::space_to_depth::<1>(in_addr, in_h, in_w, in_c, out_addr, bs_h, bs_w)
                },
                2 => unsafe {
                    depth::space_to_depth::<2>(in_addr, in_h, in_w, in_c, out_addr, bs_h, bs_w)
                },
                got => return Err(SwError::UnsupportedElemBytes { got }),
            }
            Ok(())
        }

        SwOp::DepthToSpace {
            in_addr,
            in_h,
            in_w,
            in_c,
            out_addr,
            bs_h,
            bs_w,
            elem_bytes,
        } => {
            if in_c % (bs_h * bs_w) != 0 {
                return Err(SwError::D2sNotDivisible);
            }
            // SAFETY: as above.
            match elem_bytes {
                1 => unsafe {
                    depth::depth_to_space::<1>(in_addr, in_h, in_w, in_c, out_addr, bs_h, bs_w)
                },
                2 => unsafe {
                    depth::depth_to_space::<2>(in_addr, in_h, in_w, in_c, out_addr, bs_h, bs_w)
                },
                got => return Err(SwError::UnsupportedElemBytes { got }),
            }
            Ok(())
        }

        SwOp::ResizeLinearHp {
            in_addr,
            in_h,
            in_w,
            in_c,
            out_addr,
            out_h,
            out_w,
            is_signed,
        } => {
            // Checked here rather than left to the operator's internal
            // bail-out: exceeding the table size used to mean returning
            // without writing the output, which leaves stale arena contents
            // downstream and looks like a plausible inference.
            let extent = out_w.max(out_h);
            if extent as usize > MAX_RESIZE_EXTENT {
                return Err(SwError::ResizeTooLarge { extent });
            }
            // SAFETY: shapes come from the compiled schedule; extent is
            // bounded above.
            unsafe {
                resize::resize_linear_hp(
                    in_addr, in_h, in_w, in_c, out_addr, out_h, out_w, is_signed,
                );
            }
            Ok(())
        }
    }
}
