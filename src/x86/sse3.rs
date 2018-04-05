//! Streaming SIMD Extensions 3 (SSE3)

use mem::transmute;
use simd::*;

/// Alternatively add and subtract packed single-precision (32-bit)
/// floating-point elements in `a` to/from packed elements in `b`.
#[inline]
#[target_feature(enable = "sse3")]
pub unsafe fn _mm_addsub_ps(a: f32x4, b: f32x4) -> f32x4 {
    transmute(::arch::_mm_addsub_ps(
        transmute(a),
        transmute(b),
    ))
}

/// Alternatively add and subtract packed double-precision (64-bit)
/// floating-point elements in `a` to/from packed elements in `b`.
#[inline]
#[target_feature(enable = "sse3")]
pub unsafe fn _mm_addsub_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_addsub_pd(
        transmute(a),
        transmute(b),
    ))
}

/// Horizontally add adjacent pairs of single-precision (32-bit)
/// floating-point elements in `a` and `b`, and pack the results.
#[inline]
#[target_feature(enable = "sse3")]
pub unsafe fn _mm_hadd_ps(a: f32x4, b: f32x4) -> f32x4 {
    transmute(::arch::_mm_hadd_ps(transmute(a), transmute(b)))
}

/// Horizontally add adjacent pairs of double-precision (64-bit)
/// floating-point elements in `a` and `b`, and pack the results.
#[inline]
#[target_feature(enable = "sse3")]
pub unsafe fn _mm_hadd_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_hadd_pd(transmute(a), transmute(b)))
}

/// Horizontally add adjacent pairs of single-precision (32-bit)
/// floating-point elements in `a` and `b`, and pack the results.
#[inline]
#[target_feature(enable = "sse3")]
pub unsafe fn _mm_hsub_ps(a: f32x4, b: f32x4) -> f32x4 {
    transmute(::arch::_mm_hsub_ps(transmute(a), transmute(b)))
}

/// Horizontally subtract adjacent pairs of double-precision (64-bit)
/// floating-point elements in `a` and `b`, and pack the results.
#[inline]
#[target_feature(enable = "sse3")]
pub unsafe fn _mm_hsub_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_hsub_pd(transmute(a), transmute(b)))
}
