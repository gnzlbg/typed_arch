//! Supplemental Streaming SIMD Extensions 3 (SSSE3)

use mem::transmute;
use simd::*;

/// Shuffle bytes from `a` according to the content of `b`.
///
/// The last 4 bits of each byte of `b` are used as addresses
/// into the 16 bytes of `a`.
///
/// In addition, if the highest significant bit of a byte of `b`
/// is set, the respective destination byte is set to 0.
#[inline]
#[target_feature(enable = "ssse3")]
pub unsafe fn _mm_shuffle_epi8(a: u8x16, b: u8x16) -> u8x16 {
    transmute(::arch::_mm_shuffle_epi8(
        transmute(a),
        transmute(b),
    ))
}

/// Horizontally add the adjacent pairs of values contained in 2 packed
/// 128-bit vectors of [8 x i16]. Positive sums greater than 7FFFh are
/// saturated to 7FFFh. Negative sums less than 8000h are saturated to 8000h.
#[inline]
#[target_feature(enable = "ssse3")]
pub unsafe fn _mm_hadds_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_hadds_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Horizontally subtract the adjacent pairs of values contained in 2
/// packed 128-bit vectors of [8 x i16]. Positive differences greater than
/// 7FFFh are saturated to 7FFFh. Negative differences less than 8000h are
/// saturated to 8000h.
#[inline]
#[target_feature(enable = "ssse3")]
pub unsafe fn _mm_hsubs_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_hsubs_epi16(
        transmute(a),
        transmute(b),
    ))
}
