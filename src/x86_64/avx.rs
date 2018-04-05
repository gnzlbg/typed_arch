//! Advanced Vector Extensions (AVX)

use mem::transmute;
use simd::i64x4;

/// Copy `a` to result, and insert the 64-bit integer `i` into result
/// at the location specified by `index`.
#[inline]
#[target_feature(enable = "avx")]
pub unsafe fn _mm256_insert_epi64(a: i64x4, i: i64, index: i32) -> i64x4 {
    match index {
        0 => transmute(::arch::_mm256_insert_epi64(transmute(a), i, 0)),
        1 => transmute(::arch::_mm256_insert_epi64(transmute(a), i, 1)),
        2 => transmute(::arch::_mm256_insert_epi64(transmute(a), i, 2)),
        3 => transmute(::arch::_mm256_insert_epi64(transmute(a), i, 3)),
        _ => unreachable!(),
    }
}
