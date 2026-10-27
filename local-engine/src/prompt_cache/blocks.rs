//! The checked block layout of a snapshot's payload, and its parallel reader.
//!
//! A snapshot written in this layout starts its payload, and each of its five
//! sections, on an [`ALIGN`] boundary, and its header carries the CRC-32C of
//! every [`BLOCK`] of every section. A load then reads the blocks with
//! [`READERS`] positioned reads at once, uncached, straight into the
//! section's pages, and checks each block's checksum on the reader that just
//! read it, while the block is still in that core's cache: verification
//! overlaps the read instead of following it, and the read is no longer
//! bound by one core hashing the whole payload with SHA-256.
//!
//! CRC-32C detects what the checksum is for, a snapshot damaged on disk or
//! cut short, as well as SHA-256 did (see `docs/BONSAI.md`, Prompt cache):
//! neither is a defence against a forged file, since the digest sits in the
//! same unkeyed file anyone able to write it could rewrite.

use std::fs::File;
use std::os::unix::fs::FileExt as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use crate::bonsai_native::PageBytes;

/// Alignment of the payload and of each section in the file: the page size,
/// so an uncached read lands straight in the destination's pages.
pub const ALIGN: usize = 16 * 1024;

/// Bytes covered by one checksum, and read by one `pread`.
pub const BLOCK: usize = 1024 * 1024;

/// Positioned reads in flight at once.
///
/// Measured on an M4 Pro reading a 1.3 GB file in 1 MiB blocks with
/// `F_NOCACHE`, out of the page cache: 996 ms with one reader, 327 ms with
/// four, 251 ms with eight, 217 ms with sixteen and 218 ms with 32.
const READERS: usize = 16;

/// Hex digits each block's checksum takes in the header.
const DIGITS: usize = 8;

/// Section lengths padded to [`ALIGN`], as they lie in the file.
pub const fn padded(length: usize) -> usize {
    length.next_multiple_of(ALIGN)
}

/// Blocks in a section of `length` bytes.
const fn blocks(length: usize) -> usize {
    length.div_ceil(BLOCK)
}

/// Hex digits the checksums of sections of these lengths take in the header.
pub fn checksum_digits(lengths: impl IntoIterator<Item = usize>) -> usize {
    lengths.into_iter().map(blocks).sum::<usize>() * DIGITS
}

/// The checksums of every block of `sections`, in file order, as the header
/// stores them; `None` once `abort` is set, checked every `piece` bytes.
pub fn checksums(sections: &[&[u8]], piece: usize, abort: &AtomicBool) -> Option<String> {
    let lengths = sections.iter().map(|section| section.len());
    let mut digits = String::with_capacity(checksum_digits(lengths));
    let per_piece = (piece / BLOCK).max(1);
    for section in sections {
        for (index, block) in section.chunks(BLOCK).enumerate() {
            if index % per_piece == 0 && abort.load(Ordering::Acquire) {
                return None;
            }
            push_hex(&mut digits, crc32c(block));
        }
    }
    Some(digits)
}

fn push_hex(digits: &mut String, value: u32) {
    use std::fmt::Write as _;
    let _ = write!(digits, "{value:08x}");
}

fn parse_checksums(digits: &str, count: usize) -> crate::Result<Vec<u32>> {
    let invalid = || crate::Error::InvalidFormat("invalid prompt cache checksums".into());
    if digits.len() != count * DIGITS || !digits.is_ascii() {
        return Err(invalid());
    }
    digits
        .as_bytes()
        .chunks(DIGITS)
        .map(|chunk| {
            std::str::from_utf8(chunk)
                .ok()
                .and_then(|text| u32::from_str_radix(text, 16).ok())
                .ok_or_else(invalid)
        })
        .collect()
}

/// One block to read: where it lies in the file, where it goes, and the
/// checksum it must have.
struct Job<'a> {
    offset: u64,
    bytes: &'a mut [u8],
    checksum: u32,
}

/// Read and check sections of `lengths` laid out from `start` in `file` (of
/// `file_length` bytes), against the header's checksum `digits`.
///
/// Anything but a file of exactly the expected length, every block matching
/// its checksum, is an error, so a snapshot cut short, extended or damaged is
/// never returned. Gives up within one block per reader once `abort` is set.
pub fn read<const N: usize>(
    file: &File,
    start: u64,
    file_length: u64,
    lengths: [usize; N],
    digits: &str,
    abort: &AtomicBool,
) -> crate::Result<[PageBytes; N]> {
    let align = ALIGN as u64;
    // Checked: the lengths come from the header, not yet known to be intact.
    let expected = lengths.iter().try_fold(start, |end, &length| {
        end.checked_add(length.checked_next_multiple_of(ALIGN)? as u64)
    });
    if !start.is_multiple_of(align) || expected != Some(file_length) {
        return Err(crate::Error::InvalidFormat(
            "truncated or misaligned prompt cache section".into(),
        ));
    }
    let checksums = parse_checksums(digits, lengths.iter().copied().map(blocks).sum())?;
    let mut sections = lengths
        .iter()
        .map(|&length| PageBytes::zeroed(length))
        .collect::<std::io::Result<Vec<_>>>()?;
    let mut jobs = Vec::with_capacity(checksums.len());
    let mut checksums = checksums.into_iter();
    let mut offset = start;
    for section in &mut sections {
        let length = padded(section.len()) as u64;
        for (index, bytes) in section.chunks_mut(BLOCK).enumerate() {
            // As many as there are blocks: `parse_checksums` counted them.
            let checksum = checksums.next().unwrap_or_default();
            jobs.push(Job {
                offset: offset + (index * BLOCK) as u64,
                bytes,
                checksum,
            });
        }
        offset += length;
    }
    uncached(file);
    let readers = READERS.min(jobs.len());
    let jobs = Mutex::new(jobs.into_iter());
    let failed = AtomicBool::new(false);
    let work = || drain(file, &jobs, &failed, abort);
    std::thread::scope(|scope| {
        let mut helpers = Vec::with_capacity(readers);
        for _ in 1..readers {
            if let Ok(helper) = std::thread::Builder::new()
                .name("prompt-cache-read".into())
                .spawn_scoped(scope, work)
            {
                helpers.push(helper);
            }
        }
        // This thread reads too, so a load still completes when no helper
        // could be spawned.
        let mine = work();
        helpers.into_iter().fold(mine, |result, helper| {
            let theirs = helper.join().unwrap_or_else(|_| {
                Err(crate::Error::Io(std::io::Error::other(
                    "prompt cache reader panicked",
                )))
            });
            result.and(theirs)
        })
    })?;
    sections
        .try_into()
        .map_err(|_| crate::Error::InvalidFormat("prompt cache section count mismatch".into()))
}

/// Read and check jobs until none is left, one fails (here or on another
/// reader), or `abort` is set.
fn drain(
    file: &File,
    jobs: &Mutex<std::vec::IntoIter<Job<'_>>>,
    failed: &AtomicBool,
    abort: &AtomicBool,
) -> crate::Result<()> {
    loop {
        if abort.load(Ordering::Acquire) {
            return Err(crate::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "prompt cache load abandoned",
            )));
        }
        if failed.load(Ordering::Relaxed) {
            // The reader that failed reports why.
            return Ok(());
        }
        let Some(job) = jobs.lock().unwrap_or_else(PoisonError::into_inner).next() else {
            return Ok(());
        };
        let checked = file
            .read_exact_at(job.bytes, job.offset)
            .map_err(crate::Error::from)
            .and_then(|()| {
                if crc32c(job.bytes) == job.checksum {
                    Ok(())
                } else {
                    Err(crate::Error::InvalidFormat(
                        "prompt cache checksum mismatch".into(),
                    ))
                }
            });
        if checked.is_err() {
            failed.store(true, Ordering::Relaxed);
            return checked;
        }
    }
}

/// Read `file` around the unified buffer cache from here on.
///
/// Measured on an M4 Pro, sixteen readers of a 1.3 GB snapshot out of the
/// page cache took 209 ms uncached against 244 ms through the cache, and
/// equally long (143 ms) when the file was already cached, since a cached
/// page is still served from memory. Reading through the cache would also
/// evict up to a whole snapshot's worth of other pages, the demand-paged
/// checkpoint's among them.
#[cfg(target_os = "macos")]
#[expect(unsafe_code)]
fn uncached(file: &File) {
    use std::os::fd::AsRawFd as _;
    // SAFETY: `F_NOCACHE` takes an integer argument and only changes how
    // reads of this open file use the cache; the descriptor outlives the call.
    unsafe {
        libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
    }
}

#[cfg(not(target_os = "macos"))]
const fn uncached(_file: &File) {}

/// CRC-32C (Castagnoli) of `bytes`, with the 64-bit Arm CRC instructions.
///
/// One dependency chain of `crc32cx` checked 1 GiB in 213 ms on an M4 Pro
/// core, where `sha2`'s SHA-256 took 806 ms, and each reader checks only its
/// own blocks.
#[cfg(all(target_arch = "aarch64", target_feature = "crc"))]
#[expect(unsafe_code)]
pub fn crc32c(bytes: &[u8]) -> u32 {
    use std::arch::aarch64::{__crc32cb, __crc32cd};
    let (words, rest) = bytes.as_chunks::<8>();
    let mut crc = u32::MAX;
    // SAFETY: the `crc` target feature is enabled for this build.
    unsafe {
        for word in words {
            crc = __crc32cd(crc, u64::from_le_bytes(*word));
        }
        for &byte in rest {
            crc = __crc32cb(crc, byte);
        }
    }
    !crc
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "crc")))]
pub fn crc32c(bytes: &[u8]) -> u32 {
    crc32c_bitwise(bytes)
}

/// The reference CRC-32C, one bit at a time: the fallback without the CRC
/// instructions, and the oracle the hardware one is tested against.
#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "crc"))))]
pub fn crc32c_bitwise(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0x82F6_3B78 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}
