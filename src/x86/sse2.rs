//! Streaming SIMD Extensions 2 (SSE2)

pub use arch::{_mm_clflush, _mm_lfence, _mm_mfence, _mm_pause, _mm_add_si64,
               _mm_and_si128, _mm_andnot_si128, _mm_bslli_si128,
               _mm_bsrli_si128, _mm_load_si128, _mm_loadu_si128,
               _mm_or_si128, _mm_setzero_si128, _mm_slli_si128,
               _mm_srli_si128, _mm_store_si128, _mm_storeu_si128,
               _mm_stream_si128, _mm_stream_si32, _mm_sub_si64,
               _mm_undefined_si128, _mm_xor_si128};
use mem::transmute;
pub use simd::*;

/// Add
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_add_epi8(a: i8x16, b: i8x16) -> i8x16 {
    transmute(::arch::_mm_add_epi8(transmute(a), transmute(b)))
}

/// Add
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_add_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_add_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Add
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_add_epi32(a: i32x4, b: i32x4) -> i32x4 {
    transmute(::arch::_mm_add_epi32(
        transmute(a),
        transmute(b),
    ))
}

/// Add
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_add_epi64(a: i64x2, b: i64x2) -> i64x2 {
    transmute(::arch::_mm_add_epi64(
        transmute(a),
        transmute(b),
    ))
}

/// Saturated add
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_adds_epi8(a: i8x16, b: i8x16) -> i8x16 {
    transmute(::arch::_mm_adds_epi8(
        transmute(a),
        transmute(b),
    ))
}

/// Saturated add
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_adds_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_adds_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Saturated add
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_adds_epu8(a: u8x16, b: u8x16) -> u8x16 {
    transmute(::arch::_mm_adds_epu8(
        transmute(a),
        transmute(b),
    ))
}

/// Saturated add
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_adds_epu16(a: u16x8, b: u16x8) -> u16x8 {
    transmute(::arch::_mm_adds_epu16(
        transmute(a),
        transmute(b),
    ))
}

/// Average
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_avg_epu8(a: u8x16, b: u8x16) -> u8x16 {
    transmute(::arch::_mm_avg_epu8(transmute(a), transmute(b)))
}

/// Average
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_avg_epu16(a: u16x8, b: u16x8) -> u16x8 {
    transmute(::arch::_mm_avg_epu16(
        transmute(a),
        transmute(b),
    ))
}

/// Multiply and horizontally add
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_madd_epi16(a: i16x8, b: i16x8) -> i32x4 {
    transmute(::arch::_mm_madd_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Max
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_max_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_max_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Max
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_max_epu8(a: u8x16, b: u8x16) -> u8x16 {
    transmute(::arch::_mm_max_epu8(transmute(a), transmute(b)))
}

/// Min
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_min_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_min_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Min
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_min_epu8(a: u8x16, b: u8x16) -> u8x16 {
    transmute(::arch::_mm_min_epu8(transmute(a), transmute(b)))
}

/// Multiply returning high 16 bits of the result.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_mulhi_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_mulhi_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Multiply returning high 16 bits of the result.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_mulhi_epu16(a: u16x8, b: u16x8) -> u16x8 {
    transmute(::arch::_mm_mulhi_epu16(
        transmute(a),
        transmute(b),
    ))
}

/// Multiply returning low 16 bits of the result.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_mullo_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_mullo_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Multiply the low unsigned 32-bit integers from each packed 64-bit element
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_mul_epu32(a: u32x4, b: u32x4) -> u64x2 {
    transmute(::arch::_mm_mul_epu32(
        transmute(a),
        transmute(b),
    ))
}

/// Sum absolute differences
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sad_epu8(a: u8x16, b: u8x16) -> u64x2 {
    transmute(::arch::_mm_sad_epu8(transmute(a), transmute(b)))
}

/// Subtract
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sub_epi8(a: i8x16, b: i8x16) -> i8x16 {
    transmute(::arch::_mm_sub_epi8(transmute(a), transmute(b)))
}

/// Subtract
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sub_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_sub_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Subtract
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sub_epi32(a: i32x4, b: i32x4) -> i32x4 {
    transmute(::arch::_mm_sub_epi32(
        transmute(a),
        transmute(b),
    ))
}

/// Subtract
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sub_epi64(a: i64x2, b: i64x2) -> i64x2 {
    transmute(::arch::_mm_sub_epi64(
        transmute(a),
        transmute(b),
    ))
}

/// Saturated subtract
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_subs_epi8(a: i8x16, b: i8x16) -> i8x16 {
    transmute(::arch::_mm_subs_epi8(
        transmute(a),
        transmute(b),
    ))
}

/// Saturated subtract
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_subs_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_subs_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Saturated subtract
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_subs_epu8(a: u8x16, b: u8x16) -> u8x16 {
    transmute(::arch::_mm_subs_epu8(
        transmute(a),
        transmute(b),
    ))
}

/// Saturated subtract
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_subs_epu16(a: u16x8, b: u16x8) -> u16x8 {
    transmute(::arch::_mm_subs_epu16(
        transmute(a),
        transmute(b),
    ))
}

/// Left shift by `n` while shifting in zeros.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_slli_epi16(a: i16x8, n: i32) -> i16x8 {
    let a: ::arch::__m128i = transmute(a);
    let v = match n {
        0 => ::arch::_mm_slli_epi16(a, 0),
        1 => ::arch::_mm_slli_epi16(a, 1),
        2 => ::arch::_mm_slli_epi16(a, 2),
        3 => ::arch::_mm_slli_epi16(a, 3),
        4 => ::arch::_mm_slli_epi16(a, 4),
        5 => ::arch::_mm_slli_epi16(a, 5),
        6 => ::arch::_mm_slli_epi16(a, 6),
        7 => ::arch::_mm_slli_epi16(a, 7),
        _ => ::arch::_mm_slli_epi16(a, 8),
    };
    transmute(v)
}

/// Left shift by `n` while shifting in zeros.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_slli_epi32(a: i32x4, n: i32) -> i32x4 {
    let a: ::arch::__m128i = transmute(a);
    let v = match n {
        0 => ::arch::_mm_slli_epi32(a, 0),
        1 => ::arch::_mm_slli_epi32(a, 1),
        2 => ::arch::_mm_slli_epi32(a, 2),
        3 => ::arch::_mm_slli_epi32(a, 3),
        _ => ::arch::_mm_slli_epi32(a, 4),
    };
    transmute(v)
}

/// Left shift by `n` while shifting in zeros.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_slli_epi64(a: i64x2, n: i32) -> i64x2 {
    let a: ::arch::__m128i = transmute(a);
    let v = match n {
        0 => ::arch::_mm_slli_epi64(a, 0),
        1 => ::arch::_mm_slli_epi64(a, 1),
        _ => ::arch::_mm_slli_epi64(a, 2),
    };
    transmute(v)
}

/// Left shift (shifting in zeros).
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sll_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_sll_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Left shift (shifting in zeros).
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sll_epi32(a: i32x4, b: i32x4) -> i32x4 {
    transmute(::arch::_mm_sll_epi32(
        transmute(a),
        transmute(b),
    ))
}

/// Left shift (shifting in zeros).
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sll_epi64(a: i64x2, b: i64x2) -> i64x2 {
    transmute(::arch::_mm_sll_epi64(
        transmute(a),
        transmute(b),
    ))
}

/// Right shift by `n` while shifting in sign bits.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_srai_epi16(a: i16x8, n: u8) -> i16x8 {
    let a: ::arch::__m128i = transmute(a);
    let v = match n {
        0 => ::arch::_mm_srai_epi16(a, 0),
        1 => ::arch::_mm_srai_epi16(a, 1),
        2 => ::arch::_mm_srai_epi16(a, 2),
        3 => ::arch::_mm_srai_epi16(a, 3),
        4 => ::arch::_mm_srai_epi16(a, 4),
        5 => ::arch::_mm_srai_epi16(a, 5),
        6 => ::arch::_mm_srai_epi16(a, 6),
        7 => ::arch::_mm_srai_epi16(a, 7),
        _ => ::arch::_mm_srai_epi16(a, 8),
    };
    transmute(v)
}

/// Right shift by `n` while shifting in sign bits.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_srai_epi32(a: i32x4, n: u8) -> i32x4 {
    let a: ::arch::__m128i = transmute(a);
    let v = match n {
        0 => ::arch::_mm_srai_epi32(a, 0),
        1 => ::arch::_mm_srai_epi32(a, 1),
        2 => ::arch::_mm_srai_epi32(a, 2),
        3 => ::arch::_mm_srai_epi32(a, 3),
        _ => ::arch::_mm_srai_epi32(a, 4),
    };
    transmute(v)
}

/// Right shift (shifting in sign bits).
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sra_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_sra_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Right shift (shifting in sign bits).
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sra_epi32(a: i32x4, b: i32x4) -> i32x4 {
    transmute(::arch::_mm_sra_epi32(
        transmute(a),
        transmute(b),
    ))
}

/// Right shift (shifting in zeros).
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_srl_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_srl_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Right shift (shifting in zeros).
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_srl_epi32(a: i32x4, b: i32x4) -> i32x4 {
    transmute(::arch::_mm_srl_epi32(
        transmute(a),
        transmute(b),
    ))
}

/// Right shift (shifting in zeros).
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_srl_epi64(a: i64x2, b: i64x2) -> i64x2 {
    transmute(::arch::_mm_srl_epi64(
        transmute(a),
        transmute(b),
    ))
}

/// Right shift by `n` while shifting in zeros.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_srli_epi16(a: i16x8, n: u8) -> i16x8 {
    let a: ::arch::__m128i = transmute(a);
    let v = match n {
        0 => ::arch::_mm_srli_epi16(a, 0),
        1 => ::arch::_mm_srli_epi16(a, 1),
        2 => ::arch::_mm_srli_epi16(a, 2),
        3 => ::arch::_mm_srli_epi16(a, 3),
        4 => ::arch::_mm_srli_epi16(a, 4),
        5 => ::arch::_mm_srli_epi16(a, 5),
        6 => ::arch::_mm_srli_epi16(a, 6),
        7 => ::arch::_mm_srli_epi16(a, 7),
        _ => ::arch::_mm_srli_epi16(a, 8),
    };
    transmute(v)
}

/// Right shift by `n` while shifting in zeros.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_srli_epi32(a: i32x4, n: u8) -> i32x4 {
    let a: ::arch::__m128i = transmute(a);
    let v = match n {
        0 => ::arch::_mm_srli_epi32(a, 0),
        1 => ::arch::_mm_srli_epi32(a, 1),
        2 => ::arch::_mm_srli_epi32(a, 2),
        3 => ::arch::_mm_srli_epi32(a, 3),
        _ => ::arch::_mm_srli_epi32(a, 4),
    };
    transmute(v)
}

/// Right shift by `n` while shifting in zeros.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_srli_epi64(a: i64x2, n: u8) -> i64x2 {
    let a: ::arch::__m128i = transmute(a);
    let v = match n {
        0 => ::arch::_mm_srli_epi64(a, 0),
        1 => ::arch::_mm_srli_epi64(a, 1),
        _ => ::arch::_mm_srli_epi64(a, 2),
    };
    transmute(v)
}

/// Equal
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpeq_epi8(a: i8x16, b: i8x16) -> i8x16 {
    transmute(::arch::_mm_cmpeq_epi8(
        transmute(a),
        transmute(b),
    ))
}

/// Equal
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpeq_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_cmpeq_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Equal
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpeq_epi32(a: i32x4, b: i32x4) -> i32x4 {
    transmute(::arch::_mm_cmpeq_epi32(
        transmute(a),
        transmute(b),
    ))
}

/// Greater-than
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpgt_epi8(a: i8x16, b: i8x16) -> i8x16 {
    transmute(::arch::_mm_cmpgt_epi8(
        transmute(a),
        transmute(b),
    ))
}

/// Greater-than
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpgt_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_cmpgt_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Greater-than
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpgt_epi32(a: i32x4, b: i32x4) -> i32x4 {
    transmute(::arch::_mm_cmpgt_epi32(
        transmute(a),
        transmute(b),
    ))
}

/// Less-than
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmplt_epi8(a: i8x16, b: i8x16) -> i8x16 {
    transmute(::arch::_mm_cmplt_epi8(
        transmute(a),
        transmute(b),
    ))
}

/// Less-than
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmplt_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_cmplt_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Less-than
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmplt_epi32(a: i32x4, b: i32x4) -> i32x4 {
    transmute(::arch::_mm_cmplt_epi32(
        transmute(a),
        transmute(b),
    ))
}

/// Converts lower two packed 32-bit integers in `a` to `f64`s.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtepi32_pd(a: i32x4) -> f64x2 {
    transmute(::arch::_mm_cvtepi32_pd(transmute(a)))
}

/// Replaces lower element of `a` with `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsi32_sd(a: f64x2, b: i32) -> f64x2 {
    transmute(::arch::_mm_cvtsi32_sd(transmute(a), b))
}

/// Conversion
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtepi32_ps(a: i32x4) -> f32x4 {
    transmute(::arch::_mm_cvtepi32_ps(transmute(a)))
}

/// Conversion
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtps_epi32(a: f32x4) -> i32x4 {
    transmute(::arch::_mm_cvtps_epi32(transmute(a)))
}

/// Instantiates `[a, 0, 0, 0]`
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsi32_si128(a: i32) -> i32x4 {
    transmute(::arch::_mm_cvtsi32_si128(transmute(a)))
}

/// Extracts lowest element of `a`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsi128_si32(a: i32x4) -> i32 {
    ::arch::_mm_cvtsi128_si32(transmute(a))
}

/// Instantiate
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set_epi64x(e1: i64, e0: i64) -> i64x2 {
    transmute(::arch::_mm_set_epi64x(e0, e1))
}

/// Instantiate
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> i32x4 {
    transmute(::arch::_mm_set_epi32(e0, e1, e2, e3))
}

/// Instantiate
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set_epi16(
    e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16
) -> i16x8 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    transmute(::arch::_mm_set_epi16(
        e0, e1, e2, e3, e4, e5, e6, e7,
    ))
}

/// Instantiate
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set_epi8(
    e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8,
    e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8,
) -> i8x16 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    transmute(::arch::_mm_set_epi8(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
    ))
}

/// Broadcast `a` to all elements.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set1_epi64x(a: i64) -> i64x2 {
    transmute(::arch::_mm_set1_epi64x(a))
}

/// Broadcast `a` to all elements.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set1_epi32(a: i32) -> i32x4 {
    transmute(::arch::_mm_set1_epi32(a))
}

/// Broadcast `a` to all elements.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set1_epi16(a: i16) -> i16x8 {
    transmute(::arch::_mm_set1_epi16(a))
}

/// Broadcast `a` to all elements.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set1_epi8(a: i8) -> i8x16 {
    transmute(::arch::_mm_set1_epi8(a))
}

/// Instantiate with values in reverse order
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_setr_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> i32x4 {
    transmute(::arch::_mm_setr_epi32(e0, e1, e2, e3))
}

/// Instantiate with values in reverse order
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_setr_epi16(
    e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16
) -> i16x8 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    transmute(::arch::_mm_setr_epi16(
        e0, e1, e2, e3, e4, e5, e6, e7,
    ))
}

/// Instantiate with values in reverse order
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_setr_epi8(
    e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8,
    e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8,
) -> i8x16 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    transmute(::arch::_mm_setr_epi8(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
    ))
}

/// Load 64-bit integer from memory into first element of returned vector.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_loadl_epi64(mem_addr: *const i64x2) -> i64x2 {
    transmute(::arch::_mm_loadl_epi64(
        mem_addr as *const ::arch::__m128i,
    ))
}

/// Conditionally store elements from `a` into memory using `mask`.
///
/// Elements are not stored when the highest bit is not set in the
/// corresponding element.
///
/// `mem_addr` should correspond to a 128-bit memory location and does not need
/// to be aligned on any particular boundary.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_maskmoveu_si128(
    // FIXME: mask should be of type m8x16
    a: i8x16,
    mask: i8x16,
    mem_addr: *mut i8,
) {
    transmute(::arch::_mm_maskmoveu_si128(
        transmute(a),
        transmute(mask),
        mem_addr,
    ))
}

/// Store the lower integer of `a` to a memory location.
///
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_storel_epi64(mem_addr: *mut i64x2, a: i64x2) {
    transmute(::arch::_mm_storel_epi64(
        transmute(mem_addr),
        transmute(a),
    ))
}

/// Instantiate vector with the low element extracted from `a` and its upper
/// element is zero.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_move_epi64(a: i64x2) -> i64x2 {
    transmute(::arch::_mm_move_epi64(transmute(a)))
}

/// Convert elements of `a` and `b` to 8-bit integers using signed saturation.
///
/// The converted elements of `a` and `b` are stored to the lower and upper
/// halves of the result, respectively.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_packs_epi16(a: i16x8, b: i16x8) -> i8x16 {
    transmute(::arch::_mm_packs_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Convert elements of `a` and `b` to 16-bit integers using signed saturation.
///
/// The converted elements of `a` and `b` are stored to the lower and upper
/// halves of the result, respectively.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_packs_epi32(a: i32x4, b: i32x4) -> i16x8 {
    transmute(::arch::_mm_packs_epi32(
        transmute(a),
        transmute(b),
    ))
}

/// Convert elements of `a` and `b` to 8-bit integers using unsigned saturation.
///
/// The converted elements of `a` and `b` are stored to the lower and upper
/// halves of the result, respectively.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_packus_epi16(a: i16x8, b: i16x8) -> u8x16 {
    transmute(::arch::_mm_packus_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Return the `i`-th element of `a`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_extract_epi16(a: i16x8, i: u8) -> i16 {
    let a = transmute(a);
    let v = match i {
        0 => ::arch::_mm_extract_epi16(a, 0),
        1 => ::arch::_mm_extract_epi16(a, 1),
        2 => ::arch::_mm_extract_epi16(a, 2),
        3 => ::arch::_mm_extract_epi16(a, 3),
        4 => ::arch::_mm_extract_epi16(a, 4),
        5 => ::arch::_mm_extract_epi16(a, 5),
        6 => ::arch::_mm_extract_epi16(a, 6),
        7 => ::arch::_mm_extract_epi16(a, 7),
        _ => unreachable!(),
    };
    v as i16
}

/// Return a new vector where the `i`-th element of `a` is replaced with `v`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_insert_epi16(a: i16x8, v: i16, i: u8) -> i16x8 {
    let a = transmute(a);
    let v = v as i32;
    let v = match i {
        0 => ::arch::_mm_insert_epi16(a, v, 0),
        1 => ::arch::_mm_insert_epi16(a, v, 1),
        2 => ::arch::_mm_insert_epi16(a, v, 2),
        3 => ::arch::_mm_insert_epi16(a, v, 3),
        4 => ::arch::_mm_insert_epi16(a, v, 4),
        5 => ::arch::_mm_insert_epi16(a, v, 5),
        6 => ::arch::_mm_insert_epi16(a, v, 6),
        7 => ::arch::_mm_insert_epi16(a, v, 7),
        _ => unreachable!(),
    };
    transmute(v)
}

/// Return a mask of the most significant bit of each element in `a`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_movemask_epi8(a: i8x16) -> i16 {
    // FIXME: use m1x16
    ::arch::_mm_movemask_epi8(transmute(a)) as i16
}

/// Shuffle `a` using the `control`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_shuffle_epi32(a: i32x4, control: u8) -> i32x4 {
    let a = transmute(a);
    macro_rules! call {
        ($i:expr) => {
            ::arch::_mm_shuffle_epi32(a, $i)
        };
    }
    let v = constify_imm8!(control, call);
    transmute(v)
}

/// Shuffle integers in the high 64 bits of `a` using the `control`
///
/// Put the results in the high 64 bits of the returned vector, with the low 64
/// bits being copied from from `a`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_shufflehi_epi16(a: i16x8, control: u8) -> i16x8 {
    let a = transmute(a);
    macro_rules! call {
        ($i:expr) => {
            ::arch::_mm_shufflehi_epi16(a, $i)
        };
    }
    let v = constify_imm8!(control, call);
    transmute(v)
}

/// Shuffle integers in the low 64 bits of `a` using the `control`
///
/// Put the results in the low 64 bits of the returned vector, with the high 64
/// bits being copied from from `a`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_shufflelo_epi16(a: i16x8, control: u8) -> i16x8 {
    let a = transmute(a);
    macro_rules! call {
        ($i:expr) => {
            ::arch::_mm_shufflelo_epi16(a, $i)
        };
    }
    let v = constify_imm8!(control, call);
    transmute(v)
}

/// Unpack and interleave integers from the high half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_unpackhi_epi8(a: i8x16, b: i8x16) -> i8x16 {
    transmute(::arch::_mm_unpackhi_epi8(
        transmute(a),
        transmute(b),
    ))
}

/// Unpack and interleave integers from the high half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_unpackhi_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_unpackhi_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Unpack and interleave integers from the high half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_unpackhi_epi32(a: i32x4, b: i32x4) -> i32x4 {
    transmute(::arch::_mm_unpackhi_epi32(
        transmute(a),
        transmute(b),
    ))
}

/// Unpack and interleave integers from the high half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_unpackhi_epi64(a: i64x2, b: i64x2) -> i64x2 {
    transmute(::arch::_mm_unpackhi_epi64(
        transmute(a),
        transmute(b),
    ))
}

/// Unpack and interleave integers from the low half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_unpacklo_epi8(a: i8x16, b: i8x16) -> i8x16 {
    transmute(::arch::_mm_unpacklo_epi8(
        transmute(a),
        transmute(b),
    ))
}

/// Unpack and interleave integers from the low half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_unpacklo_epi16(a: i16x8, b: i16x8) -> i16x8 {
    transmute(::arch::_mm_unpacklo_epi16(
        transmute(a),
        transmute(b),
    ))
}

/// Unpack and interleave integers from the low half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_unpacklo_epi32(a: i32x4, b: i32x4) -> i32x4 {
    transmute(::arch::_mm_unpacklo_epi32(
        transmute(a),
        transmute(b),
    ))
}

/// Unpack and interleave integers from the low half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_unpacklo_epi64(a: i64x2, b: i64x2) -> i64x2 {
    transmute(::arch::_mm_unpacklo_epi64(
        transmute(a),
        transmute(b),
    ))
}

/// Return a new vector with the low element of `a` replaced by the sum of the
/// low elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_add_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_add_sd(transmute(a), transmute(b)))
}

/// Add packed double-precision (64-bit) floating-point elements in `a` and
/// `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_add_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_add_pd(transmute(a), transmute(b)))
}

/// Return a new vector with the low element of `a` replaced by the result of
/// diving the lower element of `a` by the lower element of `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_div_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_div_sd(transmute(a), transmute(b)))
}

/// Divide packed double-precision (64-bit) floating-point elements in `a` by
/// packed elements in `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_div_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_div_pd(transmute(a), transmute(b)))
}

/// Return a new vector with the low element of `a` replaced by the maximum
/// of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_max_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_max_sd(transmute(a), transmute(b)))
}

/// Return a new vector with the maximum values from corresponding elements in
/// `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_max_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_max_pd(transmute(a), transmute(b)))
}

/// Return a new vector with the low element of `a` replaced by the minimum
/// of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_min_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_min_sd(transmute(a), transmute(b)))
}

/// Return a new vector with the minimum values from corresponding elements in
/// `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_min_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_min_pd(transmute(a), transmute(b)))
}

/// Return a new vector with the low element of `a` replaced by multiplying the
/// low elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_mul_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_mul_sd(transmute(a), transmute(b)))
}

/// Multiply packed double-precision (64-bit) floating-point elements in `a`
/// and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_mul_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_mul_pd(transmute(a), transmute(b)))
}

/// Return a new vector with the low element of `a` replaced by the square
/// root of the lower element `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sqrt_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_sqrt_sd(transmute(a), transmute(b)))
}

/// Return a new vector with the square root of each of the values in `a`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sqrt_pd(a: f64x2) -> f64x2 {
    transmute(::arch::_mm_sqrt_pd(transmute(a)))
}

/// Return a new vector with the low element of `a` replaced by subtracting the
/// low element by `b` from the low element of `a`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sub_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_sub_sd(transmute(a), transmute(b)))
}

/// Subtract packed double-precision (64-bit) floating-point elements in `b`
/// from `a`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sub_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_sub_pd(transmute(a), transmute(b)))
}

/// Compute the bitwise AND of packed double-precision (64-bit) floating-point
/// elements in `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_and_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_and_pd(transmute(a), transmute(b)))
}

/// Compute the bitwise NOT of `a` and then AND with `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_andnot_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_andnot_pd(
        transmute(a),
        transmute(b),
    ))
}

/// Compute the bitwise OR of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_or_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_or_pd(transmute(a), transmute(b)))
}

/// Compute the bitwise OR of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_xor_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_xor_pd(transmute(a), transmute(b)))
}

/// Return a new vector with the low element of `a` replaced by the equality
/// comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpeq_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpeq_sd(transmute(a), transmute(b)))
}

/// Return a new vector with the low element of `a` replaced by the less-than
/// comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmplt_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmplt_sd(transmute(a), transmute(b)))
}

/// Return a new vector with the low element of `a` replaced by the
/// less-than-or-equal comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmple_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmple_sd(transmute(a), transmute(b)))
}

/// Return a new vector with the low element of `a` replaced by the
/// greater-than comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpgt_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpgt_sd(transmute(a), transmute(b)))
}

/// Return a new vector with the low element of `a` replaced by the
/// greater-than-or-equal comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpge_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpge_sd(transmute(a), transmute(b)))
}

/// Return a new vector with the low element of `a` replaced by the result
/// of comparing both of the lower elements of `a` and `b` to `NaN`. If
/// neither are equal to `NaN` then `0xFFFFFFFFFFFFFFFF` is used and `0`
/// otherwise.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpord_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpord_sd(
        transmute(a),
        transmute(b),
    ))
}

/// Return a new vector with the low element of `a` replaced by the result of
/// comparing both of the lower elements of `a` and `b` to `NaN`. If either is
/// equal to `NaN` then `0xFFFFFFFFFFFFFFFF` is used and `0` otherwise.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpunord_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpunord_sd(
        transmute(a),
        transmute(b),
    ))
}

/// Return a new vector with the low element of `a` replaced by the not-equal
/// comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpneq_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpneq_sd(
        transmute(a),
        transmute(b),
    ))
}

/// Return a new vector with the low element of `a` replaced by the
/// not-less-than comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpnlt_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpnlt_sd(
        transmute(a),
        transmute(b),
    ))
}

/// Return a new vector with the low element of `a` replaced by the
/// not-less-than-or-equal comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpnle_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpnle_sd(
        transmute(a),
        transmute(b),
    ))
}

/// Return a new vector with the low element of `a` replaced by the
/// not-greater-than comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpngt_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpngt_sd(
        transmute(a),
        transmute(b),
    ))
}

/// Return a new vector with the low element of `a` replaced by the
/// not-greater-than-or-equal comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpnge_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpnge_sd(
        transmute(a),
        transmute(b),
    ))
}

/// Compare corresponding elements in `a` and `b` for equality.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpeq_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpeq_pd(transmute(a), transmute(b)))
}

/// Compare corresponding elements in `a` and `b` for less-than.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmplt_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmplt_pd(transmute(a), transmute(b)))
}

/// Compare corresponding elements in `a` and `b` for less-than-or-equal
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmple_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmple_pd(transmute(a), transmute(b)))
}

/// Compare corresponding elements in `a` and `b` for greater-than.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpgt_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpgt_pd(transmute(a), transmute(b)))
}

/// Compare corresponding elements in `a` and `b` for greater-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpge_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpge_pd(transmute(a), transmute(b)))
}

/// Compare corresponding elements in `a` and `b` to see if neither is `NaN`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpord_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpord_pd(
        transmute(a),
        transmute(b),
    ))
}

/// Compare corresponding elements in `a` and `b` to see if either is `NaN`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpunord_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpunord_pd(
        transmute(a),
        transmute(b),
    ))
}

/// Compare corresponding elements in `a` and `b` for not-equal.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpneq_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpneq_pd(
        transmute(a),
        transmute(b),
    ))
}

/// Compare corresponding elements in `a` and `b` for not-less-than.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpnlt_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpnlt_pd(
        transmute(a),
        transmute(b),
    ))
}

/// Compare corresponding elements in `a` and `b` for not-less-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpnle_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpnle_pd(
        transmute(a),
        transmute(b),
    ))
}

/// Compare corresponding elements in `a` and `b` for not-greater-than.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpngt_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpngt_pd(
        transmute(a),
        transmute(b),
    ))
}

/// Compare corresponding elements in `a` and `b` for
/// not-greater-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cmpnge_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_cmpnge_pd(
        transmute(a),
        transmute(b),
    ))
}

/// Compare the lower element of `a` and `b` for equality.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_comieq_sd(a: f64x2, b: f64x2) -> bool {
    ::arch::_mm_comieq_sd(transmute(a), transmute(b)) != 0
}

/// Compare the lower element of `a` and `b` for less-than.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_comilt_sd(a: f64x2, b: f64x2) -> bool {
    ::arch::_mm_comilt_sd(transmute(a), transmute(b)) != 0
}

/// Compare the lower element of `a` and `b` for less-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_comile_sd(a: f64x2, b: f64x2) -> bool {
    ::arch::_mm_comile_sd(transmute(a), transmute(b)) != 0
}

/// Compare the lower element of `a` and `b` for greater-than.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_comigt_sd(a: f64x2, b: f64x2) -> bool {
    ::arch::_mm_comigt_sd(transmute(a), transmute(b)) != 0
}

/// Compare the lower element of `a` and `b` for greater-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_comige_sd(a: f64x2, b: f64x2) -> bool {
    ::arch::_mm_comige_sd(transmute(a), transmute(b)) != 0
}

/// Compare the lower element of `a` and `b` for not-equal.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_comineq_sd(a: f64x2, b: f64x2) -> bool {
    ::arch::_mm_comineq_sd(transmute(a), transmute(b)) != 0
}

/// Compare the lower element of `a` and `b` for equality.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_ucomieq_sd(a: f64x2, b: f64x2) -> bool {
    ::arch::_mm_ucomieq_sd(transmute(a), transmute(b)) != 0
}

/// Compare the lower element of `a` and `b` for less-than.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_ucomilt_sd(a: f64x2, b: f64x2) -> bool {
    ::arch::_mm_ucomilt_sd(transmute(a), transmute(b)) != 0
}

/// Compare the lower element of `a` and `b` for less-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_ucomile_sd(a: f64x2, b: f64x2) -> bool {
    ::arch::_mm_ucomile_sd(transmute(a), transmute(b)) != 0
}

/// Compare the lower element of `a` and `b` for greater-than.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_ucomigt_sd(a: f64x2, b: f64x2) -> bool {
    ::arch::_mm_ucomigt_sd(transmute(a), transmute(b)) != 0
}

/// Compare the lower element of `a` and `b` for greater-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_ucomige_sd(a: f64x2, b: f64x2) -> bool {
    ::arch::_mm_ucomige_sd(transmute(a), transmute(b)) != 0
}

/// Compare the lower element of `a` and `b` for not-equal.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_ucomineq_sd(a: f64x2, b: f64x2) -> bool {
    ::arch::_mm_ucomineq_sd(transmute(a), transmute(b)) != 0
}

/// Convert packed double-precision (64-bit) floating-point elements in "a" to
/// packed single-precision (32-bit) floating-point elements
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtpd_ps(a: f64x2) -> f32x4 {
    transmute(::arch::_mm_cvtpd_ps(transmute(a)))
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to
/// packed double-precision (64-bit) floating-point elements.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtps_pd(a: f32x4) -> f64x2 {
    transmute(::arch::_mm_cvtps_pd(transmute(a)))
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to
/// packed 32-bit integers.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtpd_epi32(a: f64x2) -> i32x4 {
    transmute(::arch::_mm_cvtpd_epi32(transmute(a)))
}

/// Convert the lower double-precision (64-bit) floating-point element in a to
/// a 32-bit integer.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsd_si32(a: f64x2) -> i32 {
    transmute(::arch::_mm_cvtsd_si32(transmute(a)))
}

/// Convert the lower double-precision (64-bit) floating-point element in `b`
/// to a single-precision (32-bit) floating-point element, store the result in
/// the lower element of the return value, and copy the upper element from `a`
/// to the upper element the return value.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsd_ss(a: f32x4, b: f64x2) -> f32x4 {
    transmute(::arch::_mm_cvtsd_ss(transmute(a), transmute(b)))
}

/// Return the lower double-precision (64-bit) floating-point element of "a".
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsd_f64(a: f64x2) -> f64 {
    transmute(::arch::_mm_cvtsd_f64(transmute(a)))
}

/// Convert the lower single-precision (32-bit) floating-point element in `b`
/// to a double-precision (64-bit) floating-point element, store the result in
/// the lower element of the return value, and copy the upper element from `a`
/// to the upper element the return value.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtss_sd(a: f64x2, b: f32x4) -> f64x2 {
    transmute(::arch::_mm_cvtss_sd(transmute(a), transmute(b)))
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to
/// packed 32-bit integers with truncation.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvttpd_epi32(a: f64x2) -> i32x4 {
    transmute(::arch::_mm_cvttpd_epi32(transmute(a)))
}

/// Convert the lower double-precision (64-bit) floating-point element in `a`
/// to a 32-bit integer with truncation.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvttsd_si32(a: f64x2) -> i32 {
    transmute(::arch::_mm_cvttsd_si32(transmute(a)))
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to
/// packed 32-bit integers with truncation.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvttps_epi32(a: f32x4) -> i32x4 {
    transmute(::arch::_mm_cvttps_epi32(transmute(a)))
}

/// Copy double-precision (64-bit) floating-point element `a` to the lower
/// element of the packed 64-bit return value.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set_sd(a: f64) -> f64x2 {
    transmute(::arch::_mm_set_sd(a))
}

/// Broadcast double-precision (64-bit) floating-point value a to all elements
/// of the return value.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set1_pd(a: f64) -> f64x2 {
    transmute(::arch::_mm_set1_pd(a))
}

/// Broadcast double-precision (64-bit) floating-point value a to all elements
/// of the return value.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set_pd1(a: f64) -> f64x2 {
    transmute(::arch::_mm_set_pd1(a))
}

/// Set packed double-precision (64-bit) floating-point elements in the return
/// value with the supplied values.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set_pd(a: f64, b: f64) -> f64x2 {
    transmute(::arch::_mm_set_pd(a, b))
}

/// Set packed double-precision (64-bit) floating-point elements in the return
/// value with the supplied values in reverse order.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_setr_pd(a: f64, b: f64) -> f64x2 {
    transmute(::arch::_mm_setr_pd(a, b))
}

/// Returns packed double-precision (64-bit) floating-point elements with all
/// zeros.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_setzero_pd() -> f64x2 {
    transmute(::arch::_mm_setzero_pd())
}

/// Return a mask of the most significant bit of each element in `a`.
///
/// The mask is stored in the 2 least significant bits of the return value.
/// All other bits are set to `0`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_movemask_pd(a: f64x2) -> i32 {
    transmute(::arch::_mm_movemask_pd(transmute(a)))
}

/// Load 128-bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from memory into the returned vector.
/// `mem_addr` must be aligned on a 16-byte boundary or a general-protection
/// exception may be generated.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_load_pd(mem_addr: *const f64x2) -> f64x2 {
    transmute(::arch::_mm_load_pd(transmute(mem_addr)))
}

/// Loads a 64-bit double-precision value to the low element of a
/// 128-bit integer vector and clears the upper element.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_load_sd(mem_addr: *const f64) -> f64x2 {
    transmute(::arch::_mm_load_pd(mem_addr))
}

/// Loads a double-precision value into the high-order bits of a 128-bit
/// vector of [2 x double]. The low-order bits are copied from the low-order
/// bits of the first operand.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_loadh_pd(a: f64x2, mem_addr: *const f64) -> f64x2 {
    transmute(::arch::_mm_loadh_pd(transmute(a), mem_addr))
}

/// Loads a double-precision value into the low-order bits of a 128-bit
/// vector of [2 x double]. The high-order bits are copied from the
/// high-order bits of the first operand.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_loadl_pd(a: f64x2, mem_addr: *const f64) -> f64x2 {
    transmute(::arch::_mm_loadl_pd(transmute(a), mem_addr))
}

/// Stores a 128-bit floating point vector of [2 x double] to a 128-bit
/// aligned memory location.
/// To minimize caching, the data is flagged as non-temporal (unlikely to be
/// used again soon).
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_stream_pd(mem_addr: *mut f64x2, a: f64x2) {
    ::arch::_mm_stream_pd(transmute(mem_addr), transmute(a))
}

/// Stores the lower 64 bits of a 128-bit vector of [2 x double] to a
/// memory location.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_store_sd(mem_addr: *mut f64, a: f64x2) {
    ::arch::_mm_store_sd(mem_addr, transmute(a))
}

/// Store 128-bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory. `mem_addr` must be aligned
/// on a 16-byte boundary or a general-protection exception may be generated.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_store_pd(mem_addr: *mut f64x2, a: f64x2) {
    ::arch::_mm_store_pd(transmute(mem_addr), transmute(a))
}

/// Store 128-bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_storeu_pd(mem_addr: *mut f64, a: f64x2) {
    ::arch::_mm_storeu_pd(mem_addr, transmute(a))
}

/// Store the lower double-precision (64-bit) floating-point element from `a`
/// into 2 contiguous elements in memory. `mem_addr` must be aligned on a
/// 16-byte boundary or a general-protection exception may be generated.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_store1_pd(mem_addr: *mut f64x2, a: f64x2) {
    ::arch::_mm_store1_pd(transmute(mem_addr), transmute(a))
}

/// Store the lower double-precision (64-bit) floating-point element from `a`
/// into 2 contiguous elements in memory. `mem_addr` must be aligned on a
/// 16-byte boundary or a general-protection exception may be generated.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_store_pd1(mem_addr: *mut f64x2, a: f64x2) {
    ::arch::_mm_store1_pd(transmute(mem_addr), transmute(a))
}

/// Store 2 double-precision (64-bit) floating-point elements from `a` into
/// memory in reverse order.
/// `mem_addr` must be aligned on a 16-byte boundary or a general-protection
/// exception may be generated.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_storer_pd(mem_addr: *mut f64x2, a: f64x2) {
    ::arch::_mm_storer_pd(transmute(mem_addr), transmute(a))
}

/// Stores the upper 64 bits of a 128-bit vector of [2 x double] to a
/// memory location.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_storeh_pd(mem_addr: *mut f64, a: f64x2) {
    ::arch::_mm_storeh_pd(mem_addr, transmute(a))
}

/// Stores the lower 64 bits of a 128-bit vector of [2 x double] to a
/// memory location.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_storel_pd(mem_addr: *mut f64, a: f64x2) {
    ::arch::_mm_storel_pd(mem_addr, transmute(a))
}

/// Load a double-precision (64-bit) floating-point element from memory
/// into both elements of returned vector.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_load1_pd(mem_addr: *const f64) -> f64x2 {
    transmute(::arch::_mm_load1_pd(mem_addr))
}

/// Load a double-precision (64-bit) floating-point element from memory
/// into both elements of returned vector.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_load_pd1(mem_addr: *const f64) -> f64x2 {
    transmute(::arch::_mm_load_pd1(mem_addr))
}

/// Load 2 double-precision (64-bit) floating-point elements from memory into
/// the returned vector in reverse order. `mem_addr` must be aligned on a
/// 16-byte boundary or a general-protection exception may be generated.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_loadr_pd(mem_addr: *const f64x2) -> f64x2 {
    transmute(::arch::_mm_loadr_pd(transmute(mem_addr)))
}

/// Load 128-bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from memory into the returned vector.
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_loadu_pd(mem_addr: *const f64) -> f64x2 {
    transmute(::arch::_mm_loadu_pd(mem_addr))
}

/// Constructs a 128-bit floating-point vector of [2 x double] from two
/// 128-bit vector parameters of [2 x double], using the `control`.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_shuffle_pd(a: f64x2, b: f64x2, control: u8) -> f64x2 {
    let a = transmute(a);
    let b = transmute(b);
    let v = match control & 0b11 {
        0b00 => ::arch::_mm_shuffle_pd(a, b, 0b00),
        0b01 => ::arch::_mm_shuffle_pd(a, b, 0b01),
        0b10 => ::arch::_mm_shuffle_pd(a, b, 0b10),
        0b11 => ::arch::_mm_shuffle_pd(a, b, 0b11),
        _ => unreachable!(),
    };
    transmute(v)
}

/// Constructs a 128-bit floating-point vector of [2 x double]. The lower
/// 64 bits are set to the lower 64 bits of the second parameter. The upper
/// 64 bits are set to the upper 64 bits of the first parameter.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_move_sd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_move_sd(transmute(a), transmute(b)))
}

/// Casts a 128-bit floating-point vector of [2 x double] into a 128-bit
/// floating-point vector of [4 x float].
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_castpd_ps(a: f64x2) -> f32x4 {
    transmute(::arch::_mm_castpd_ps(transmute(a)))
}

/// Casts a 128-bit floating-point vector of [2 x double] into a 128-bit
/// integer vector.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_castpd_si128(a: f64x2) -> ::arch::__m128i {
    ::arch::_mm_castpd_si128(transmute(a))
}

/// Casts a 128-bit floating-point vector of [4 x float] into a 128-bit
/// floating-point vector of [2 x double].
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_castps_pd(a: f32x4) -> f64x2 {
    transmute(::arch::_mm_castps_pd(transmute(a)))
}

/// Casts a 128-bit floating-point vector of [4 x float] into a 128-bit
/// integer vector.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_castps_si128(a: f32x4) -> ::arch::__m128i {
    ::arch::_mm_castps_si128(transmute(a))
}

/// Casts a 128-bit integer vector into a 128-bit floating-point vector
/// of [2 x double].
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_castsi128_pd(a: ::arch::__m128i) -> f64x2 {
    transmute(::arch::_mm_castsi128_pd(a))
}

/// Casts a 128-bit integer vector into a 128-bit floating-point vector
/// of [4 x float].
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_castsi128_ps(a: ::arch::__m128i) -> f32x4 {
    transmute(::arch::_mm_castsi128_ps(a))
}

/// Return vector of type f64x2 with undefined elements.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_undefined_pd() -> f64x2 {
    transmute(::arch::_mm_undefined_pd())
}

/// The resulting `f64x2` element is composed by the low-order values of
/// the two `f64x2` interleaved input elements, i.e.:
///
/// * The [127:64] bits are copied from the [127:64] bits of the second input
/// * The [63:0] bits are copied from the [127:64] bits of the first input
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_unpackhi_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_unpackhi_pd(
        transmute(a),
        transmute(b),
    ))
}

/// The resulting `f64x2` element is composed by the high-order values of
/// the two `f64x2` interleaved input elements, i.e.:
///
/// * The [127:64] bits are copied from the [63:0] bits of the second input
/// * The [63:0] bits are copied from the [63:0] bits of the first input
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_unpacklo_pd(a: f64x2, b: f64x2) -> f64x2 {
    transmute(::arch::_mm_unpacklo_pd(
        transmute(a),
        transmute(b),
    ))
}

/// Multiplies 32-bit unsigned integer values contained in the lower bits
/// of the two 64-bit integer vectors and returns the 64-bit unsigned
/// product.
#[inline]
#[target_feature(enable = "sse2,mmx")]
pub unsafe fn _mm_mul_su32(a: u32x2, b: u32x2) -> u64 {
    transmute(::arch::_mm_mul_su32(transmute(a), transmute(b)))
}

/// Converts the two signed 32-bit integer elements of a 64-bit vector of
/// [2 x i32] into two double-precision floating-point values, returned in a
/// 128-bit vector of [2 x double].
#[inline]
#[target_feature(enable = "sse2,mmx")]
pub unsafe fn _mm_cvtpi32_pd(a: i32x2) -> f64x2 {
    transmute(::arch::_mm_cvtpi32_pd(transmute(a)))
}

/// Initializes both 64-bit values in a 128-bit vector of [2 x i64] with
/// the specified 64-bit integer values.
#[inline]
#[target_feature(enable = "sse2,mmx")]
pub unsafe fn _mm_set_epi64(e1: i64, e0: i64) -> i64x2 {
    transmute(::arch::_mm_set_epi64(
        transmute(e1),
        transmute(e0),
    ))
}

/// Initializes both values in a 128-bit vector of [2 x i64] with the
/// specified 64-bit value.
#[inline]
#[target_feature(enable = "sse2,mmx")]
pub unsafe fn _mm_set1_epi64(a: i64) -> i64x2 {
    transmute(::arch::_mm_set1_epi64(transmute(a)))
}

/// Constructs a 128-bit integer vector, initialized in reverse order
/// with the specified 64-bit integral values.
#[inline]
#[target_feature(enable = "sse2,mmx")]
pub unsafe fn _mm_setr_epi64(e1: i64, e0: i64) -> i64x2 {
    transmute(::arch::_mm_setr_epi64(
        transmute(e1),
        transmute(e0),
    ))
}

/// Returns the lower 64 bits of a 128-bit integer vector as a 64-bit
/// integer.
#[inline]
#[target_feature(enable = "sse2,mmx")]
pub unsafe fn _mm_movepi64_pi64(a: i64x2) -> i64 {
    transmute(::arch::_mm_movepi64_pi64(transmute(a)))
}

/// Moves the 64-bit operand to a 128-bit integer vector, zeroing the
/// upper bits.
#[inline]
#[target_feature(enable = "sse2,mmx")]
pub unsafe fn _mm_movpi64_epi64(a: i64) -> i64x2 {
    transmute(::arch::_mm_movpi64_epi64(transmute(a)))
}

/// Converts the two double-precision floating-point elements of a
/// 128-bit vector of [2 x double] into two signed 32-bit integer values,
/// returned in a 64-bit vector of [2 x i32].
#[inline]
#[target_feature(enable = "sse2,mmx")]
pub unsafe fn _mm_cvtpd_pi32(a: f64x2) -> i32x2 {
    transmute(::arch::_mm_cvtpd_pi32(transmute(a)))
}

/// Converts the two double-precision floating-point elements of a
/// 128-bit vector of [2 x double] into two signed 32-bit integer values,
/// returned in a 64-bit vector of [2 x i32].
/// If the result of either conversion is inexact, the result is truncated
/// (rounded towards zero) regardless of the current MXCSR setting.
#[inline]
#[target_feature(enable = "sse2,mmx")]
pub unsafe fn _mm_cvttpd_pi32(a: f64x2) -> i32x2 {
    transmute(::arch::_mm_cvttpd_pi32(transmute(a)))
}
