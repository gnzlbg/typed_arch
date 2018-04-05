//! Streaming SIMD Extensions 4.1 (SSE4.1)

use mem::transmute;
use simd::*;

pub struct Round(i32);

impl Round {
    /// Round to nearest integer
    pub fn nearest() -> Self {
        Round(::arch::_MM_FROUND_TO_NEAREST_INT)
    }
    /// Round down
    pub fn down() -> Self {
        Round(::arch::_MM_FROUND_TO_NEG_INF)
    }
    /// Round up
    pub fn up() -> Self {
        Round(::arch::_MM_FROUND_TO_POS_INF)
    }
    /// Round towards zero (truncate)
    pub fn zero() -> Self {
        Round(::arch::_MM_FROUND_TO_ZERO)
    }
    /// Use current `MXCSR` setting.
    pub fn current() -> Self {
        Round(::arch::_MM_FROUND_CUR_DIRECTION)
    }
    /// Round down and do not supress exceptions
    pub fn floor() -> Self {
        Round(::arch::_MM_FROUND_FLOOR)
    }
    /// Round up and do not supress exceptions
    pub fn ceil() -> Self {
        Round(::arch::_MM_FROUND_CEIL)
    }
    /// Round towards zero (trunacate) and do not supress exceptions
    pub fn trunc() -> Self {
        Round(::arch::_MM_FROUND_TRUNC)
    }
    /// Use current `MXCSR` setting and do not supress exceptions
    pub fn rint() -> Self {
        Round(::arch::_MM_FROUND_RINT)
    }
    /// Use current `MXCSR` setting and supress exceptions
    pub fn nearby_int() -> Self {
        Round(::arch::_MM_FROUND_NEARBYINT)
    }
    #[must_use = "this operation returns a new value"]
    pub fn supress_exceptions(self) -> Round {
        Round(self.0 | ::arch::_MM_FROUND_NO_EXC)
    }
    #[must_use = "this operation returns a new value"]
    pub fn raise_exceptions(self) -> Round {
        Round(self.0 | ::arch::_MM_FROUND_RAISE_EXC)
    }
}

/// Blend packed 8-bit integers from `a` and `b` using `mask`
///
/// The high bit of each corresponding mask byte determines the selection.
/// If the high bit is set the element of `a` is selected. The element
/// of `b` is selected otherwise.
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_blendv_epi8(
    a: i8x16,
    b: i8x16,
    mask: i8x16, // FIXME: mask should be m8x16
) -> i8x16 {
    transmute(::arch::_mm_blendv_epi8(
        transmute(a),
        transmute(b),
        transmute(mask),
    ))
}

/// Round the elements in `a` using the `rounding` parameter.
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_round_ps(a: f32x4, rounding: Round) -> f32x4 {
    let a = transmute(a);
    macro_rules! call {
        ($i:expr) => {
            ::arch::_mm_round_ps(a, $i)
        };
    }
    let v = constify_imm4!(rounding.0, call);
    transmute(v)
}

/// Round the elements in `a` up to an integer value.
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_ceil_ps(a: f32x4) -> f32x4 {
    transmute(::arch::_mm_ceil_ps(transmute(a)))
}

/// Round the elements in `a` down to an integer value.
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_floor_ps(a: f32x4) -> f32x4 {
    transmute(::arch::_mm_floor_ps(transmute(a)))
}

/// Zero extend packed unsigned 8-bit integers in `a` to packed 16-bit integers
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cvtepu8_epi16(a: u8x16) -> i16x8 {
    transmute(::arch::_mm_cvtepu8_epi16(transmute(a)))
}
