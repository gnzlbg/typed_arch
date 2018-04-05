//! Streaming SIMD Extensions 2 (SSE2)

pub use arch::_mm_stream_si64;
use mem::transmute;
use simd::*;

/// Convert the lower double-precision (64-bit) floating-point element in a to
/// a 64-bit integer.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsd_si64(a: f64x2) -> i64 {
    ::arch::_mm_cvtsd_si64(transmute(a))
}

/// Alias for `_mm_cvtsd_si64`
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsd_si64x(a: f64x2) -> i64 {
    ::arch::_mm_cvtsd_si64x(transmute(a))
}

/// Convert the lower double-precision (64-bit) floating-point element in `a`
/// to a 64-bit integer with truncation.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvttsd_si64(a: f64x2) -> i64 {
    ::arch::_mm_cvttsd_si64(transmute(a))
}

/// Alias for `_mm_cvttsd_si64`
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvttsd_si64x(a: f64x2) -> i64 {
    ::arch::_mm_cvttsd_si64x(transmute(a))
}

/// Return a vector whose lowest element is `a` and all higher elements are `0`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsi64_si128(a: i64) -> i64x2 {
    transmute(::arch::_mm_cvtsi64_si128(a))
}

/// Return a vector whose lowest element is `a` and all higher elements are
/// `0`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsi64x_si128(a: i64) -> i64x2 {
    transmute(::arch::_mm_cvtsi64x_si128(a))
}

/// Return the lowest element of `a`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsi128_si64(a: i64x2) -> i64 {
    ::arch::_mm_cvtsi128_si64(transmute(a))
}

/// Return the lowest element of `a`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsi128_si64x(a: i64x2) -> i64 {
    ::arch::_mm_cvtsi128_si64(transmute(a))
}

/// Return `a` with its lower element replaced by `b` after converting it to
/// an `f64`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsi64_sd(a: f64x2, b: i64) -> f64x2 {
    transmute(::arch::_mm_cvtsi64_sd(transmute(a), b))
}

/// Return `a` with its lower element replaced by `b` after converting it to
/// an `f64`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsi64x_sd(a: f64x2, b: i64) -> f64x2 {
    transmute(::arch::_mm_cvtsi64x_sd(transmute(a), b))
}
