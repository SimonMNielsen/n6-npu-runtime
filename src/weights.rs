//! Weight-image validation against the NOR manifest.
//!
//! # Why this exists
//!
//! On 2026-08-11 the DeepLab demo produced a uniformly blank mask with no error
//! of any kind. The NPU ran, every epoch completed, the schedule was correct —
//! and the weight image had been erased out from under it, so the NPU dutifully
//! convolved 900 KiB of `0xFF`. Nothing in the firmware was in a position to
//! notice, because nothing ever looked at the weights: the CPU hands the NPU a
//! base address and the NPU fetches through XSPI2 on its own.
//!
//! Erasure is not exotic. The weight slots share a 128 MiB part with the
//! bootloader, the app and the DFU staging area, sector erases round outward to
//! 64 KiB, and the flash layout has historically had partitions overlapping a
//! slot outright. Any of flashing the wrong target, an interrupted `flash.py`,
//! or eventually an OTA swap will reproduce it.
//!
//! With several models sharing one device the stakes go up rather than down:
//! each model has its own slot, and a partial re-flash can leave one image
//! valid and its neighbour blank. Verification is per-model for that reason.
//!
//! # Manifest
//!
//! A single erase sector, written by `flash.py` alongside each weight image. It
//! is a separate sector rather than a header prepended to the image because
//! shifting an image would invalidate the base address `stedgeai` baked into
//! the compiled EC blob.
//!
//! ```text
//! offset  size  field
//! 0x00    4     magic     "N6MF" little-endian
//! 0x04    4     version   currently 1
//! 0x08    4     count     number of entries
//! 0x0C    4     crc32     CRC-32 of the entry array (count * 32 bytes)
//! 0x10    ..    entries
//!
//! entry (32 bytes)
//! 0x00    16    id        model id, ASCII, NUL-padded
//! 0x10    4     base      NOR address of the weight image
//! 0x14    4     len       image length in bytes
//! 0x18    4     crc32     CRC-32 of the image
//! 0x1C    4     flags     bit 0: resident (copied to SRAM at boot)
//! ```
//!
//! The application supplies the sector's location via [`ManifestLocation`];
//! this crate hardcodes no addresses. In the deeplab demo those values come
//! from `memory.toml` by way of the generated memory map.
//!
//! # Ordering
//!
//! Every read here goes through the XSPI2 memory-mapped window, so the caller
//! must not run before that window is live. Booting from flash, the FSBL has
//! already enabled it; in a RAM-app run the application must map it first.

use defmt::{info, warn};

/// `"N6MF"` little-endian.
const MAGIC: u32 = 0x464D_364E;
const VERSION: u32 = 1;
const HEADER_BYTES: u32 = 16;
const ENTRY_BYTES: u32 = 32;

/// Bytes between samples when scanning for an erased image.
const SCAN_STRIDE: u32 = 4096;

/// Where the manifest lives and how coarsely the part erases.
#[derive(Clone, Copy, defmt::Format)]
pub struct ManifestLocation {
    pub base: u32,
    /// Size of the reserved region, used only to bound `count`.
    pub size: u32,
    /// Erase granularity of the NOR part. Determines the unit at which
    /// [`verify`] judges a span to be blank.
    pub erase_sector: u32,
}

#[derive(Clone, Copy, defmt::Format)]
pub struct WeightInfo {
    pub base: u32,
    pub len: u32,
    pub crc32: u32,
    pub resident: bool,
}

#[derive(Clone, Copy, defmt::Format)]
pub enum WeightError {
    /// The manifest sector is blank. Nothing has ever written it.
    ManifestErased,
    /// Present but not ours, or a version we cannot read.
    ManifestBadMagic { magic: u32 },
    ManifestBadVersion { version: u32 },
    /// `count` is impossible for the sector size.
    ManifestBadCount { count: u32 },
    /// The entry array does not match its own CRC — a torn write.
    ManifestCorrupt { stored: u32, computed: u32 },
    /// No entry for this model id.
    ModelNotFound,
    /// The manifest disagrees with the build about where the image lives.
    SlotMismatch { manifest: u32, expected: u32 },
    /// An erase sector of the image reads back as blank flash. `at` is the
    /// address of the first such sector, so a prefix erase is distinguishable
    /// from a whole-slot one.
    ImageErased { at: u32 },
    /// The image is present but does not match its recorded CRC.
    ImageCorrupt { stored: u32, computed: u32 },
}

impl WeightError {
    /// A one-line remedy, because the failure is always operator-facing.
    pub fn hint(&self) -> &'static str {
        match self {
            WeightError::ManifestErased
            | WeightError::ManifestBadMagic { .. }
            | WeightError::ManifestBadVersion { .. }
            | WeightError::ManifestBadCount { .. }
            | WeightError::ManifestCorrupt { .. }
            | WeightError::ModelNotFound => {
                "re-run flash.py for this model's weight target — it writes the image \
                 and the manifest entry describing it in one session"
            }
            WeightError::SlotMismatch { .. } => {
                "the flashed image predates a memory.toml change — re-flash, and if the \
                 slot base moved, regenerate the model (the base is baked into the EC blob)"
            }
            WeightError::ImageErased { .. } => {
                "the weight slot has been erased — most likely another flash target \
                 overlapped it; check the partition table in memory.toml, then re-flash"
            }
            WeightError::ImageCorrupt { .. } => "the weight image is damaged — re-flash",
        }
    }
}

// ── CRC-32 (IEEE 802.3, reflected) ───────────────────────────────────────────
//
// Matches Python's `zlib.crc32`, which is what flash.py uses. Nibble-wise so
// the table is 64 bytes instead of 1 KiB; the loop is bounded by XSPI2 read
// latency, not by the arithmetic.

const CRC_TABLE: [u32; 16] = [
    0x0000_0000, 0x1DB7_1064, 0x3B6E_20C8, 0x26D9_30AC,
    0x76DC_4190, 0x6B6B_51F4, 0x4DB2_6158, 0x5005_713C,
    0xEDB8_8320, 0xF00F_9344, 0xD6D6_A3E8, 0xCB61_B38C,
    0x9B64_C2B0, 0x86D3_D2D4, 0xA00A_E278, 0xBDBD_F21C,
];

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc = CRC_TABLE[((crc ^ b as u32) & 0x0F) as usize] ^ (crc >> 4);
        crc = CRC_TABLE[((crc ^ (b as u32 >> 4)) & 0x0F) as usize] ^ (crc >> 4);
    }
    !crc
}

/// # Safety
/// `addr..addr+len` must be a readable mapped range. Callers pass NOR XIP
/// addresses in a window the caller has already brought into memory-mapped
/// mode — see the module-level note on ordering.
unsafe fn nor_slice(addr: u32, len: u32) -> &'static [u8] {
    unsafe { core::slice::from_raw_parts(addr as *const u8, len as usize) }
}

fn rd32(addr: u32) -> u32 {
    unsafe { core::ptr::read_volatile(addr as *const u32) }
}

/// Cheap erased-image detector.
///
/// A full CRC over ~900 KiB of single-SPI XIP NOR costs real boot time, so the
/// default path samples instead: one word every [`SCAN_STRIDE`] bytes, judged
/// per erase sector. If *any* sector reads back entirely as `0xFF`, the image
/// is reported as erased.
///
/// Per-sector rather than whole-image, because erasure is per-sector. The
/// failure that actually happened erased the whole slot, but the more likely
/// future one — an overlapping partition, an interrupted `flash.py` — erases a
/// *prefix*. A whole-image test would sail straight past that, since the
/// surviving tail reads back as ordinary data.
///
/// This detects erasure, not corruption; that is what `full_crc` is for. The
/// residual false positive is a model with a genuinely blank sector-sized span,
/// which for int8 weights would mean tens of thousands of consecutive `-1`
/// coefficients. If that ever happens the CRC path is the answer, not a wider
/// stride.
fn scan_erased(base: u32, len: u32, erase_sector: u32) -> Option<u32> {
    if len < 4 {
        return Some(base);
    }

    let mut sector = 0u32;
    while sector < len {
        let end = core::cmp::min(sector + erase_sector, len);
        let mut blank = true;

        let mut off = sector;
        while off + 4 <= end {
            if rd32(base + off) != 0xFFFF_FFFF {
                blank = false;
                break;
            }
            off += SCAN_STRIDE;
        }
        // Always sample the last word of the sector — a stride that does not
        // divide the sector would otherwise leave the tail unexamined.
        if blank && end >= 4 && rd32(base + end - 4) != 0xFFFF_FFFF {
            blank = false;
        }

        if blank {
            return Some(base + sector);
        }
        sector += erase_sector;
    }
    None
}

/// Look up `id` in the NOR manifest and verify its weight image.
///
/// `expected_base` is the slot address the build was compiled against; a
/// disagreement means the flashed image is stale relative to the source of
/// truth.
///
/// `full_crc` reads the entire image. Leave it off for normal boots — see
/// [`scan_erased`] for what the cheap path does and does not cover.
pub fn verify(
    loc: &ManifestLocation,
    id: &str,
    expected_base: u32,
    full_crc: bool,
) -> Result<WeightInfo, WeightError> {
    // Refuse absurd counts before trusting `count` for a length calculation —
    // a blank sector reads back as 0xFFFFFFFF.
    let max_entries = (loc.size - HEADER_BYTES) / ENTRY_BYTES;

    let magic = rd32(loc.base);
    if magic == 0xFFFF_FFFF {
        return Err(WeightError::ManifestErased);
    }
    if magic != MAGIC {
        return Err(WeightError::ManifestBadMagic { magic });
    }
    let version = rd32(loc.base + 4);
    if version != VERSION {
        return Err(WeightError::ManifestBadVersion { version });
    }
    let count = rd32(loc.base + 8);
    if count == 0 || count > max_entries {
        return Err(WeightError::ManifestBadCount { count });
    }
    let stored = rd32(loc.base + 12);

    let entries = unsafe { nor_slice(loc.base + HEADER_BYTES, count * ENTRY_BYTES) };
    let computed = crc32(entries);
    if computed != stored {
        return Err(WeightError::ManifestCorrupt { stored, computed });
    }

    for i in 0..count as usize {
        let e = &entries[i * ENTRY_BYTES as usize..][..ENTRY_BYTES as usize];
        let name_len = e[..16].iter().position(|&c| c == 0).unwrap_or(16);
        if &e[..name_len] != id.as_bytes() {
            continue;
        }

        let base = u32::from_le_bytes([e[16], e[17], e[18], e[19]]);
        let len = u32::from_le_bytes([e[20], e[21], e[22], e[23]]);
        let crc = u32::from_le_bytes([e[24], e[25], e[26], e[27]]);
        let flags = u32::from_le_bytes([e[28], e[29], e[30], e[31]]);

        if base != expected_base {
            return Err(WeightError::SlotMismatch {
                manifest: base,
                expected: expected_base,
            });
        }
        if let Some(at) = scan_erased(base, len, loc.erase_sector) {
            return Err(WeightError::ImageErased { at });
        }
        if full_crc {
            let computed = crc32(unsafe { nor_slice(base, len) });
            if computed != crc {
                return Err(WeightError::ImageCorrupt { stored: crc, computed });
            }
        }

        return Ok(WeightInfo {
            base,
            len,
            crc32: crc,
            resident: flags & 1 != 0,
        });
    }

    Err(WeightError::ModelNotFound)
}

/// Verify and log the outcome.
///
/// Kept separate from [`verify`] so the decision to log — and to log at `warn`
/// with a remedy attached — stays out of the error path itself.
pub fn verify_and_log(
    loc: &ManifestLocation,
    id: &str,
    expected_base: u32,
    full_crc: bool,
) -> Result<WeightInfo, WeightError> {
    match verify(loc, id, expected_base, full_crc) {
        Ok(info) => {
            info!(
                "weights '{=str}': {=u32} B at {=u32:#010x}, crc {=u32:#010x}{=str}",
                id,
                info.len,
                info.base,
                info.crc32,
                if full_crc {
                    " (verified)"
                } else {
                    " (not verified — erased-scan only)"
                },
            );
            Ok(info)
        }
        Err(e) => {
            warn!("weights '{=str}': REFUSING TO RUN — {}", id, e);
            warn!("  {=str}", e.hint());
            Err(e)
        }
    }
}
