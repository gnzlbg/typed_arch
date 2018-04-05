//! Advanced Vector Extensions (AVX)

use mem::transmute;
use simd::f64x4;

/// Adds two 256-bit f64x4 vectors.
#[inline]
#[target_feature(enable = "avx")]
pub unsafe fn _mm256_add_pd(a: f64x4, b: f64x4) -> f64x4 {
    transmute(::arch::_mm256_add_pd(transmute(a), transmute(b)))
}
