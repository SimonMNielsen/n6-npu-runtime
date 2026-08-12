//! Bilinear `Resize` with the `half_pixel` coordinate transform — a port of
//! ST's `ll_sw_forward_resize_integer`.
//!
//! Two of these appear in the DeepLab schedule: the ASPP image-pool broadcast
//! (1×1 → 16×16) and the final mask upsample (16×16 → 256×256). Neural-ART has
//! no resize primitive, so both run on the M55.
//!
//! # Coordinate transform
//!
//! Standard ONNX `Resize` with `coordinate_transformation_mode=half_pixel`,
//! `mode=linear`:
//!
//! ```text
//! src = (dst + 0.5) * (src_extent / dst_extent) - 0.5
//! i0  = clamp(floor(src), 0, extent-1);  i1 = clamp(i0 + 1, 0, extent-1)
//! f   = clamp(src - i0, 0, 1)
//! out = (1-fx)(1-fy)·p00 + fx(1-fy)·p01 + (1-fx)fy·p10 + fx·fy·p11
//! ```
//!
//! # Quantisation
//!
//! Both DeepLab resizes have identical input and output int8 quantisation
//! (`is == os`, `izp == ozp`), so the affine requant is the identity and the
//! interpolation runs directly on the raw int8 codes. **This is an assumption
//! about the schedule, not a general truth** — a future model whose resize
//! changes scale would need scale/zero-point exported by the generator and a
//! real requant path here. There is currently nothing that would detect that
//! case automatically.
//!
//! # Arithmetic
//!
//! Per-column `i0`/`i1`/weight triples are precomputed in Q15, then combined
//! with the row weight into Q30 so the inner loop is four multiplies and an
//! add per output element, with a single rounding shift.

use super::MAX_RESIZE_EXTENT;

/// # Safety
/// `in_addr`/`out_addr` are valid, non-overlapping regions of exactly
/// `in_h*in_w*in_c` and `out_h*out_w*in_c` bytes.
#[inline]
pub(crate) unsafe fn resize_linear_hp(
    in_addr: u32,
    in_h: u32,
    in_w: u32,
    in_c: u32,
    out_addr: u32,
    out_h: u32,
    out_w: u32,
    is_signed: bool,
) {
    if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 || in_c == 0 {
        return;
    }

    // Degenerate 1x1 source (the ASPP image pool): every output pixel is a
    // byte-for-byte copy of the single input pixel. Worth special-casing —
    // it skips ~1e6 interpolation steps that would all produce the same
    // answer.
    if in_h == 1 && in_w == 1 {
        let c_bytes = in_c as usize;
        let src = in_addr as *const u8;
        let dst_base = out_addr as *mut u8;
        unsafe {
            for i in 0..((out_h * out_w) as usize) {
                core::ptr::copy_nonoverlapping(src, dst_base.add(i * c_bytes), c_bytes);
            }
        }
        return;
    }

    let in_w_i32 = in_w as i32;
    let in_h_i32 = in_h as i32;
    let in_c_us = in_c as usize;
    let in_row_b = (in_w as usize) * in_c_us;
    let out_row_b = (out_w as usize) * in_c_us;

    // The per-column tables are fixed-size stack arrays, so an oversized
    // output would overflow them. The caller checks this before dispatch
    // (see `sw::run`); this is a belt-and-braces bail-out.
    if (out_w as usize) > MAX_RESIZE_EXTENT || (out_h as usize) > MAX_RESIZE_EXTENT {
        return;
    }
    let mut x0_arr = [0i32; MAX_RESIZE_EXTENT];
    let mut x1_arr = [0i32; MAX_RESIZE_EXTENT];
    let mut wx_arr = [0i32; MAX_RESIZE_EXTENT];

    let scale_x = (in_w as f32) / (out_w as f32);
    for x in 0..(out_w as usize) {
        let s = ((x as f32) + 0.5) * scale_x - 0.5;
        let x0f = libm::floorf(s);
        let mut x0 = x0f as i32;
        let mut fx = s - x0f;
        if x0 < 0 {
            x0 = 0;
            fx = 0.0;
        }
        if x0 >= in_w_i32 - 1 {
            x0 = in_w_i32 - 1;
            fx = 0.0;
        }
        let x1 = if x0 + 1 < in_w_i32 { x0 + 1 } else { x0 };
        let wx = (fx * 32768.0 + 0.5) as i32;
        x0_arr[x] = x0;
        x1_arr[x] = x1;
        wx_arr[x] = wx.clamp(0, 32768);
    }

    let scale_y = (in_h as f32) / (out_h as f32);
    let in_base = in_addr as *const u8;
    let out_base = out_addr as *mut u8;

    for y in 0..(out_h as usize) {
        let s = ((y as f32) + 0.5) * scale_y - 0.5;
        let y0f = libm::floorf(s);
        let mut y0 = y0f as i32;
        let mut fy = s - y0f;
        if y0 < 0 {
            y0 = 0;
            fy = 0.0;
        }
        if y0 >= in_h_i32 - 1 {
            y0 = in_h_i32 - 1;
            fy = 0.0;
        }
        let y1 = if y0 + 1 < in_h_i32 { y0 + 1 } else { y0 };
        let wy = ((fy * 32768.0 + 0.5) as i32).clamp(0, 32768);
        let iwy = 32768 - wy;

        // SAFETY: `y0`/`y1` are clamped to the source extent and `y` is bounded
        // by `out_h`, so every offset below stays inside the caller's regions.
        unsafe {
            let row_top: *const u8 = in_base.add((y0 as usize) * in_row_b);
            let row_bot: *const u8 = in_base.add((y1 as usize) * in_row_b);
            let out_row = out_base.add(y * out_row_b);

            for x in 0..(out_w as usize) {
                let x0 = x0_arr[x] as usize;
                let x1 = x1_arr[x] as usize;
                let wx = wx_arr[x];
                let iwx = 32768 - wx;

                let p_tl = row_top.add(x0 * in_c_us);
                let p_tr = row_top.add(x1 * in_c_us);
                let p_bl = row_bot.add(x0 * in_c_us);
                let p_br = row_bot.add(x1 * in_c_us);
                let p_out = out_row.add(x * in_c_us);

                // Four-corner weights in Q30 (Q15 x Q15). They sum to 2^30, so
                // the accumulator is shifted right by 30 with rounding.
                let w_tl = (iwx as i64) * (iwy as i64);
                let w_tr = (wx as i64) * (iwy as i64);
                let w_bl = (iwx as i64) * (wy as i64);
                let w_br = (wx as i64) * (wy as i64);

                if is_signed {
                    for c in 0..in_c_us {
                        let s_tl = *(p_tl.add(c) as *const i8) as i64;
                        let s_tr = *(p_tr.add(c) as *const i8) as i64;
                        let s_bl = *(p_bl.add(c) as *const i8) as i64;
                        let s_br = *(p_br.add(c) as *const i8) as i64;
                        let acc = w_tl * s_tl + w_tr * s_tr + w_bl * s_bl + w_br * s_br;
                        // Round to nearest, ties away from zero.
                        let biased = if acc >= 0 {
                            acc + (1i64 << 29)
                        } else {
                            acc - (1i64 << 29)
                        };
                        let out_val = (biased >> 30) as i32;
                        *(p_out.add(c) as *mut i8) = out_val.clamp(-128, 127) as i8;
                    }
                } else {
                    for c in 0..in_c_us {
                        let s_tl = *(p_tl.add(c)) as i64;
                        let s_tr = *(p_tr.add(c)) as i64;
                        let s_bl = *(p_bl.add(c)) as i64;
                        let s_br = *(p_br.add(c)) as i64;
                        let acc = w_tl * s_tl + w_tr * s_tr + w_bl * s_bl + w_br * s_br;
                        let out_val = ((acc + (1i64 << 29)) >> 30) as i32;
                        *(p_out.add(c)) = out_val.clamp(0, 255) as u8;
                    }
                }
            }
        }
    }
}
