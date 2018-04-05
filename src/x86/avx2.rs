//! Advanced Vector Extensions 2 (AVX2)

use mem::transmute;
use simd::*;

/// Blend packed 8-bit integers from `a` and `b` using `mask`.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_blendv_epi8(
    a: i8x32,
    b: i8x32,
    mask: i8x32, // FIXME: should be m8x32
) -> i8x32 {
    transmute(::arch::_mm256_blendv_epi8(
        transmute(a),
        transmute(b),
        transmute(mask),
    ))
}
