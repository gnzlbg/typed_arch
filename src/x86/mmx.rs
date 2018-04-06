//! Multi Media eXtensions (MMX)

use mem::transmute;
use simd::*;
pub use arch::_mm_setzero_si64; // FIXME: i64x1

/// Add
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_add_pi8(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_mm_add_pi8(transmute(a), transmute(b)))
}

/// Add
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddb(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_m_paddb(transmute(a), transmute(b)))
}

/// Add
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_add_pi16(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_mm_add_pi16(transmute(a), transmute(b)))
}

/// Add
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddw(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_m_paddw(transmute(a), transmute(b)))
}

/// Add
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_add_pi32(a: i32x2, b: i32x2) -> i32x2 {
    transmute(::arch::_mm_add_pi32(transmute(a), transmute(b)))
}

/// Add
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddd(a: i32x2, b: i32x2) -> i32x2 {
    transmute(::arch::_m_paddd(transmute(a), transmute(b)))
}

/// Add saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_adds_pi8(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_mm_adds_pi8(transmute(a), transmute(b)))
}

/// Add saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddsb(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_m_paddsb(transmute(a), transmute(b)))
}

/// Add saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_adds_pi16(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_mm_adds_pi16(
        transmute(a),
        transmute(b),
    ))
}

/// Add saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddsw(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_m_paddsw(transmute(a), transmute(b)))
}

/// Add saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_adds_pu8(a: u8x8, b: u8x8) -> u8x8 {
    transmute(::arch::_mm_adds_pu8(transmute(a), transmute(b)))
}

/// Add saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddusb(a: u8x8, b: u8x8) -> u8x8 {
    transmute(::arch::_m_paddusb(transmute(a), transmute(b)))
}

/// Add saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_adds_pu16(a: u16x4, b: u16x4) -> u16x4 {
    transmute(::arch::_mm_adds_pu16(
        transmute(a),
        transmute(b),
    ))
}

/// Add saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddusw(a: u16x4, b: u16x4) -> u16x4 {
    transmute(::arch::_m_paddusw(transmute(a), transmute(b)))
}

/// Sub
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_sub_pi8(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_mm_sub_pi8(transmute(a), transmute(b)))
}

/// Sub
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubb(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_m_psubb(transmute(a), transmute(b)))
}

/// Sub
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_sub_pi16(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_mm_sub_pi16(transmute(a), transmute(b)))
}

/// Sub
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubw(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_m_psubw(transmute(a), transmute(b)))
}

/// Sub
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_sub_pi32(a: i32x2, b: i32x2) -> i32x2 {
    transmute(::arch::_mm_sub_pi32(transmute(a), transmute(b)))
}

/// Sub
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubd(a: i32x2, b: i32x2) -> i32x2 {
    transmute(::arch::_m_psubd(transmute(a), transmute(b)))
}

/// Sub saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_subs_pi8(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_mm_subs_pi8(transmute(a), transmute(b)))
}

/// Sub saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubsb(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_m_psubsb(transmute(a), transmute(b)))
}

/// Sub saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_subs_pi16(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_mm_subs_pi16(
        transmute(a),
        transmute(b),
    ))
}

/// Sub saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubsw(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_m_psubsw(transmute(a), transmute(b)))
}

/// Sub saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_subs_pu8(a: u8x8, b: u8x8) -> u8x8 {
    transmute(::arch::_mm_subs_pu8(transmute(a), transmute(b)))
}

/// Sub saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubusb(a: u8x8, b: u8x8) -> u8x8 {
    transmute(::arch::_m_psubusb(transmute(a), transmute(b)))
}

/// Sub saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_subs_pu16(a: u16x4, b: u16x4) -> u16x4 {
    transmute(::arch::_mm_subs_pu16(
        transmute(a),
        transmute(b),
    ))
}

/// Sub saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubusw(a: u16x4, b: u16x4) -> u16x4 {
    transmute(::arch::_m_psubusw(transmute(a), transmute(b)))
}

/// Convert saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_packs_pi16(a: i16x4, b: i16x4) -> i8x8 {
    transmute(::arch::_mm_packs_pi16(
        transmute(a),
        transmute(b),
    ))
}

/// Convert saturated
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_packs_pi32(a: i32x2, b: i32x2) -> i16x4 {
    transmute(::arch::_mm_packs_pi32(
        transmute(a),
        transmute(b),
    ))
}

/// Greater-than
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_cmpgt_pi8(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_mm_cmpgt_pi8(
        transmute(a),
        transmute(b),
    ))
}

/// Greater-than
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_cmpgt_pi16(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_mm_cmpgt_pi16(
        transmute(a),
        transmute(b),
    ))
}

/// Greater-than
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_cmpgt_pi32(a: i32x2, b: i32x2) -> i32x2 {
    transmute(::arch::_mm_cmpgt_pi32(
        transmute(a),
        transmute(b),
    ))
}

/// Unpack high interleaved: `[a.2, b.2, a.3, b.3]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_unpackhi_pi16(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_mm_unpackhi_pi16(
        transmute(a),
        transmute(b),
    ))
}

/// Unpack higher elements interleaved: `[a.4, b.4, a.5, b.5, a.6, b.6, a.7, b.7]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_unpackhi_pi8(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_mm_unpackhi_pi8(
        transmute(a),
        transmute(b),
    ))
}

/// Unpack lower elements interleaved: `[a.0, b.0, a.1, b.1, a.2, b.2, a.3, b.3]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_unpacklo_pi8(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_mm_unpacklo_pi8(
        transmute(a),
        transmute(b),
    ))
}

/// Unpack lower elements interleaved: `[a.0 b.0 a.1 b.1]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_unpacklo_pi16(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_mm_unpacklo_pi16(
        transmute(a),
        transmute(b),
    ))
}

/// Unpack higher elements interleaved: `[a.1, b.1]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_unpackhi_pi32(a: i32x2, b: i32x2) -> i32x2 {
    transmute(::arch::_mm_unpackhi_pi32(
        transmute(a),
        transmute(b),
    ))
}

/// Unpack lower elements interleaved: [a.0, b.0]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_unpacklo_pi32(a: i32x2, b: i32x2) -> i32x2 {
    transmute(::arch::_mm_unpacklo_pi32(
        transmute(a),
        transmute(b),
    ))
}

/// Instantiate: `[e0, e1, e2, e3]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_set_pi16(e0: i16, e1: i16, e2: i16, e3: i16) -> i16x4 {
    transmute(::arch::_mm_set_pi16(
        transmute(e0),
        transmute(e1),
        transmute(e2),
        transmute(e3),
    ))
}

/// Instantiate: `[e0, e1]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_set_pi32(e0: i32, e1: i32) -> i32x2 {
    transmute(::arch::_mm_set_pi32(
        transmute(e0),
        transmute(e1),
    ))
}

/// Instantiate: `[e0, e1, e2, e3, e4, e5, e6, e7]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_set_pi8(
    e0: i8, e1: i8, e2: i8, e3: i8, e4: i8, e5: i8, e6: i8, e7: i8
) -> i8x8 {
    transmute(::arch::_mm_set_pi8(
        transmute(e0),
        transmute(e1),
        transmute(e2),
        transmute(e3),
        transmute(e4),
        transmute(e5),
        transmute(e6),
        transmute(e7),
    ))
}

/// Broadcast: `[a, a, a, a]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_set1_pi16(a: i16) -> i16x4 {
    transmute(::arch::_mm_set1_pi16(transmute(a)))
}

/// Broadcast: `[a, a]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_set1_pi32(a: i32) -> i32x2 {
    transmute(::arch::_mm_set1_pi32(transmute(a)))
}

/// Broadcast: `[a, a, a, a, a, a, a, a]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_set1_pi8(a: i8) -> i8x8 {
    transmute(::arch::_mm_set1_pi8(transmute(a)))
}

/// Instantiate reverse: `[e3, e2, e1, e0]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_setr_pi16(e0: i16, e1: i16, e2: i16, e3: i16) -> i16x4 {
    transmute(::arch::_mm_setr_pi16(
        transmute(e0),
        transmute(e1),
        transmute(e2),
        transmute(e3),
    ))
}

/// Instantiate reverse: `[e1, e0]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_setr_pi32(e0: i32, e1: i32) -> i32x2 {
    transmute(::arch::_mm_setr_pi32(
        transmute(e0),
        transmute(e1),
    ))
}

/// Instantiate reverse: `[e7, e6, e5, e4, e3, e2, e1, e0]`
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_setr_pi8(
    e0: i8, e1: i8, e2: i8, e3: i8, e4: i8, e5: i8, e6: i8, e7: i8
) -> i8x8 {
    transmute(::arch::_mm_setr_pi8(
        transmute(e0),
        transmute(e1),
        transmute(e2),
        transmute(e3),
        transmute(e4),
        transmute(e5),
        transmute(e6),
        transmute(e7),
    ))
}
