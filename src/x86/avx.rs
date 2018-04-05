//! Advanced Vector Extensions (AVX)

use mem::transmute;
use simd::*;

/// Add
#[inline]
#[target_feature(enable = "avx")]
pub unsafe fn _mm256_add_pd(a: f64x4, b: f64x4) -> f64x4 {
    transmute(::arch::_mm256_add_pd(
        transmute(a),
        transmute(b),
    ))
}

/// Square root
#[inline]
#[target_feature(enable = "avx")]
pub unsafe fn _mm256_sqrt_pd(a: f64x4) -> f64x4 {
    transmute(::arch::_mm256_sqrt_pd(transmute(a)))
}
