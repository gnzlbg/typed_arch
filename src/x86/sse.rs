//! Streaming SIMD Extensions (SSE)

use mem::transmute;
use simd::*;

/// Reciprocal square root (approximate).
#[inline]
#[target_feature(enable = "sse")]
pub unsafe fn _mm_rsqrt_ps(a: f32x4) -> f32x4 {
    transmute(::arch::_mm_rsqrt_ps(transmute(a)))
}

/// Reciprocal (approximate)
#[inline]
#[target_feature(enable = "sse")]
pub unsafe fn _mm_rcp_ps(a: f32x4) -> f32x4 {
    transmute(::arch::_mm_rcp_ps(transmute(a)))
}

/// Square root.
#[inline]
#[target_feature(enable = "sse")]
pub unsafe fn _mm_sqrt_ps(a: f32x4) -> f32x4 {
    transmute(::arch::_mm_sqrt_ps(transmute(a)))
}

/// Bitwise AND of packed single-precision (32-bit) floating-point elements.
#[inline]
#[target_feature(enable = "sse")]
pub unsafe fn _mm_and_ps(a: f32x4, b: f32x4) -> f32x4 {
    transmute(::arch::_mm_and_ps(transmute(a), transmute(b)))
}

/// Shuffle elements in `a` and `b` using `mask`.
///
/// The lower half of result takes values from `a` and the higher half from
/// `b`. Mask is split to 2 control bits each to index the element from inputs.
#[inline]
#[target_feature(enable = "sse")]
pub unsafe fn _mm_shuffle_ps(a: f32x4, b: f32x4, mask: u8) -> f32x4 {
    let a = transmute(a);
    let b = transmute(b);

    macro_rules! call {
        ($i:expr) => {
            ::arch::_mm_shuffle_ps(a, b, $i)
        };
    }

    let v = constify_imm8!(mask, call);
    transmute(v)
}
