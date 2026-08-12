//! `SpaceToDepth` / `DepthToSpace` — bit-exact ports of ST's
//! `LL_ATON_LIB_DMA_SpaceToDepth` and `LL_ATON_LIB_DMA_DepthToSpace`.
//!
//! These exist because the Neural-ART hardware cannot perform dilated
//! convolution directly. `stedgeai` rewrites each dilated conv as a
//! `SpaceToDepth → Conv → DepthToSpace` triple: the rearrangement turns a
//! dilated kernel over a sparse neighbourhood into a dense kernel over an
//! interleaved tensor, which the hardware can do. The rearrangement itself is
//! pure data movement and lands on the M55.
//!
//! Both tensors are ATON canonical NHWC (`chpos = CHPos_Last`), so element
//! `(h, w, c)` of a `(H, W, C)` tensor at `addr` sits at
//! `addr + ((h * W + w) * C + c) * elem_bytes`.
//!
//! All access is through raw pointers rather than slices. The regions live in
//! the NPU activation arena, which the NPU also touches; constructing a `&mut
//! [u8]` over them would assert an exclusivity that does not hold.

/// `SpaceToDepth`, DCR-canonical block interleave.
///
/// ```text
/// output[b, oh, ow, hb*bs_w*C + wb*C + c] = input[b, oh*bs_h + hb, ow*bs_w + wb, c]
/// ```
///
/// Iterating over output positions in memory order lets the inner step copy a
/// whole `C`-element run per `(oh, ow, hb, wb)` tuple.
///
/// # Safety
/// * `in_addr`/`out_addr` are valid, non-overlapping tensor regions with
///   exactly the byte count implied by their shapes.
/// * `in_h % bs_h == 0` and `in_w % bs_w == 0`.
#[inline]
pub(crate) unsafe fn space_to_depth<const B: usize>(
    in_addr: u32,
    in_h: u32,
    in_w: u32,
    in_c: u32,
    out_addr: u32,
    bs_h: u32,
    bs_w: u32,
) {
    let out_h_dim = in_h / bs_h;
    let out_w_dim = in_w / bs_w;
    let c_bytes = (in_c as usize) * B;
    let in_row = (in_w as usize) * c_bytes; // bytes per input h-row
    let out_row = (out_w_dim as usize) * (in_c as usize) * (bs_h as usize) * (bs_w as usize) * B;
    let block_row = (in_c as usize) * (bs_w as usize) * B; // bytes per hb row in out block

    let in_base: *const u8 = in_addr as *const u8;
    let out_base: *mut u8 = out_addr as *mut u8;

    unsafe {
        for out_h in 0..out_h_dim {
            for out_w in 0..out_w_dim {
                let out_block = out_base.add(
                    (out_h as usize) * out_row + (out_w as usize) * (block_row * (bs_h as usize)),
                );
                for hb in 0..bs_h {
                    let in_h_idx = out_h * bs_h + hb;
                    let in_row_p = in_base.add((in_h_idx as usize) * in_row);
                    let out_row_p = out_block.add((hb as usize) * block_row);
                    for wb in 0..bs_w {
                        let in_w_idx = out_w * bs_w + wb;
                        let src = in_row_p.add((in_w_idx as usize) * c_bytes);
                        let dst = out_row_p.add((wb as usize) * c_bytes);
                        core::ptr::copy_nonoverlapping(src, dst, c_bytes);
                    }
                }
            }
        }
    }
}

/// `DepthToSpace`, DCR mode — the inverse of [`space_to_depth`].
///
/// ```text
/// output[b, ih*bs_h + hb, iw*bs_w + wb, oc] = input[b, ih, iw, hb*bs_w*max_c + wb*max_c + oc]
/// ```
/// with `max_c = in_c / (bs_h * bs_w)`.
///
/// Iterating in input order lets the inner step copy a whole `max_c` run per
/// `(ih, iw, hb, wb)` tuple.
///
/// # Safety
/// * `in_addr`/`out_addr` are valid, non-overlapping tensor regions with
///   exactly the byte count implied by their shapes.
/// * `in_c % (bs_h * bs_w) == 0`.
#[inline]
pub(crate) unsafe fn depth_to_space<const B: usize>(
    in_addr: u32,
    in_h: u32,
    in_w: u32,
    in_c: u32,
    out_addr: u32,
    bs_h: u32,
    bs_w: u32,
) {
    let max_c = in_c / (bs_h * bs_w);
    let out_w_d = in_w * bs_w;
    let inc_b = (in_c as usize) * B;
    let maxc_b = (max_c as usize) * B;
    let in_row_b = (in_w as usize) * inc_b;
    let out_c_b = (max_c as usize) * B; // == maxc_b
    let out_row_b = (out_w_d as usize) * out_c_b;

    let in_base: *const u8 = in_addr as *const u8;
    let out_base: *mut u8 = out_addr as *mut u8;

    unsafe {
        for h_in in 0..in_h {
            let in_row_p = in_base.add((h_in as usize) * in_row_b);
            for w_in in 0..in_w {
                let in_pix = in_row_p.add((w_in as usize) * inc_b);
                for hb in 0..bs_h {
                    let out_h_pos = h_in * bs_h + hb;
                    let out_row_p = out_base.add((out_h_pos as usize) * out_row_b);
                    for wb in 0..bs_w {
                        let out_w_pos = w_in * bs_w + wb;
                        let src_off = ((hb as usize) * (bs_w as usize) + (wb as usize)) * maxc_b;
                        let dst = out_row_p.add((out_w_pos as usize) * out_c_b);
                        let src = in_pix.add(src_off);
                        core::ptr::copy_nonoverlapping(src, dst, maxc_b);
                    }
                }
            }
        }
    }
}
