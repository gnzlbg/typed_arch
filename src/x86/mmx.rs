//! Multi Media eXtensions (MMX)

use mem::transmute;
use sealed::*;
use simd::*;

/// Instantiate zero-initialized vector
///
/// # Instruction
///
/// This intrinsic maps to different instructions depending on how the vector is
/// used:
///
/// * [`pxor mm, mm`](http://www.felixcloutier.com/x86/PXOR.html)
/// * [`xorps xmm, xmm`](https://www.felixcloutier.com/x86/XORPS.html)
///
/// amongst others.
///
/// # Examples
///
/// ```
/// let a: i8x8 = _mm_setzero_si64();
/// assert_eq!(transmute(a), 0_i64);
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_setzero_si64<T: Any64>() -> T {
    T::from_m64(::arch::_mm_setzero_si64())
}

/// Add packed integers
///
/// When an individual result is too large to be represented in 8 bits
/// (overflow), the result is wrapped around and the low 8 bits are written to
/// the destination operand (that is, the carry is ignored).
///
/// # Instruction
///
/// [`paddb mm, mm`](http://www.felixcloutier.com/x86/PADDB:PADDW:PADDD:PADDQ.html)
///
/// # Examples
///
/// ```
/// let min = i8::min_value();
/// let max = i8::max_value();
/// let a = i8x8::new(0, 1, 0, 42, 42, -42, max,  -1);
/// let b = i8x8::new(0, 0, 1, 3,  -3,   3,   1, min);
/// let e = i8x8::new(0, 1, 0, 45, 39, -39, min, max);
/// assert_eq!(e, _mm_add_pi8(a, b));
///
/// let max = i8::max_value();
/// let a = u8x8::new(0, 1, 0,   1,    1, max, 1, 42);
/// let b = u8x8::new(0, 0, 1, 254,  max,   1, 1,  3);
/// let e = u8x8::new(0, 1, 1, max,    0,   0, 2, 45);
/// assert_eq!(e, _mm_add_pi8(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_add_pi8<T: IU8x8>(a: T, b: T) -> T {
    T::from_m64(::arch::_mm_add_pi8(a.as_m64(), b.as_m64()))
}

/// Add packed integers
///
/// When an individual result is too large to be represented in 8 bits
/// (overflow), the result is wrapped around and the low 8 bits are written to
/// the destination operand (that is, the carry is ignored).
///
/// # Instruction
///
/// [`paddb mm, mm`](http://www.felixcloutier.com/x86/PADDB:PADDW:PADDD:PADDQ.html)
///
/// # Examples
///
/// ```
/// let min = i8::min_value();
/// let max = i8::max_value();
/// let a = i8x8::new(0, 1, 0, 42, 42, -42, max,  -1);
/// let b = i8x8::new(0, 0, 1, 3,  -3,   3,   1, min);
/// let e = i8x8::new(0, 1, 0, 45, 39, -39, min, max);
/// assert_eq!(e, _mm_add_pi8(a, b));
///
/// let max = i8::max_value();
/// let a = u8x8::new(0, 1, 0,   1,    1, max, 1, 42);
/// let b = u8x8::new(0, 0, 1, 254,  max,   1, 1,  3);
/// let e = u8x8::new(0, 1, 1, max,    0,   0, 2, 45);
/// assert_eq!(e, _mm_add_pi8(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddb<T: IU8x8>(a: T, b: T) -> T {
    _mm_add_pi8(a, b)
}

/// Add packed integers
///
/// When an individual result is too large to be represented in 16 bits
/// (overflow), the result is wrapped around and the low 16 bits are written to
/// the destination operand (that is, the carry is ignored).
///
/// # Instruction
///
/// [`paddw mm, mm`](http://www.felixcloutier.com/x86/PADDB:PADDW:PADDD:PADDQ.html)
///
/// # Examples
///
/// ```
/// let min = i16::min_value();
/// let max = i16::max_value();
/// let a = i16x4::new(max, min, 32, 32);
/// let b = i16x4::new(  1,  -1,  3, -3);
/// let e = i16x4::new(min, max, 35, 29);
/// assert_eq!(e, _mm_add_pi16(a, b));
///
/// let max = u16::max_value();
/// let a = u16x4::new(42, 42,   1, max);
/// let b = u16x4::new(10, -2, max,   1);
/// let e = u16x4::new(52, 40,   0,   0);
/// assert_eq!(e, _mm_add_pi16(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_add_pi16<T: IU16x4>(a: T, b: T) -> T {
    T::from_m64(::arch::_mm_add_pi16(a.as_m64(), b.as_m64()))
}

/// Add packed integers
///
/// When an individual result is too large to be represented in 16 bits
/// (overflow), the result is wrapped around and the low 16 bits are written to
/// the destination operand (that is, the carry is ignored).
///
/// # Instruction
///
/// [`paddw mm, mm`](http://www.felixcloutier.com/x86/PADDB:PADDW:PADDD:PADDQ.html)
///
/// # Examples
///
/// ```
/// let min = i16::min_value();
/// let max = i16::max_value();
/// let a = i16x4::new(max, min, 32, 32);
/// let b = i16x4::new(  1,  -1,  3, -3);
/// let e = i16x4::new(min, max, 35, 29);
/// assert_eq!(e, _mm_add_pi16(a, b));
///
/// let max = u16::max_value();
/// let a = u16x4::new(42, 42,   1, max);
/// let b = u16x4::new(10, -2, max,   1);
/// let e = u16x4::new(52, 40,   0,   0);
/// assert_eq!(e, _mm_add_pi16(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddw<T: IU16x4>(a: T, b: T) -> T {
    _mm_add_pi16(a, b)
}

/// Add packed integers
///
/// When an individual result is too large to be represented in 32 bits
/// (overflow), the result is wrapped around and the low 32 bits are written to
/// the destination operand (that is, the carry is ignored).
///
/// # Instruction
///
/// [`paddd mm, mm`](http://www.felixcloutier.com/x86/PADDB:PADDW:PADDD:PADDQ.html)
///
/// # Examples
///
/// ```
/// let min = i32::min_value();
/// let max = i32::max_value();
/// let a = i32x2::new(max, min);
/// let b = i32x2::new(  1,  -1);
/// let e = i32x2::new(min, max);
/// assert_eq!(e, _mm_add_pi32(a, b));
///
/// let max = u32::max_value();
/// let a = u32x2::new(42, max);
/// let b = u32x2::new(10,   1);
/// let e = u32x2::new(52,   0);
/// assert_eq!(e, _mm_add_pi32(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_add_pi32<T: IU32x2>(a: T, b: T) -> T {
    T::from_m64(::arch::_mm_add_pi32(a.as_m64(), b.as_m64()))
}

/// Add packed integers
///
/// When an individual result is too large to be represented in 32 bits
/// (overflow), the result is wrapped around and the low 32 bits are written to
/// the destination operand (that is, the carry is ignored).
///
/// # Instruction
///
/// [`paddd mm, mm`](http://www.felixcloutier.com/x86/PADDB:PADDW:PADDD:PADDQ.html)
///
/// # Examples
///
/// ```
/// let min = i32::min_value();
/// let max = i32::max_value();
/// let a = i32x2::new(max, min);
/// let b = i32x2::new(  1,  -1);
/// let e = i32x2::new(min, max);
/// assert_eq!(e, _mm_add_pi32(a, b));
///
/// let max = u32::max_value();
/// let a = u32x2::new(42, max);
/// let b = u32x2::new(10,   1);
/// let e = u32x2::new(52,   0);
/// assert_eq!(e, _mm_add_pi32(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddd<T: IU32x2>(a: T, b: T) -> T {
    _mm_add_pi32(a, b)
}

/// Add packed signed integers with signed saturation
///
/// When an individual byte result is beyond the range of a signed byte integer
/// (that is, greater than `7FH` or less than `80H`), the saturated value of
/// `7FH` or `80H`, respectively, is written to the destination operand.
///
/// # Instruction
///
/// [`paddsb mm, mm`](https://www.felixcloutier.com/x86/PADDSB:PADDSW.html)
///
/// # Examples
///
/// ```
/// let min = i8::min_value();
/// let max = i8::max_value();
/// let a = i8x8::new(-100, -1,  1, 100, -1,  0, 1, 0);
/// let b = i8x8::new(-100,  1, -1, 100,  0, -1, 0, 1);
/// let e = i8x8::new( min,  0,  0, max, -1, -1, 1, 1);
/// assert_eq!(e, _mm_adds_pi8(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_adds_pi8(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_mm_adds_pi8(transmute(a), transmute(b)))
}

/// Add packed signed integers with signed saturation
///
/// When an individual byte result is beyond the range of a signed byte integer
/// (that is, greater than `7FH` or less than `80H`), the saturated value of
/// `7FH` or `80H`, respectively, is written to the destination operand.
///
/// # Instruction
///
/// [`paddsb mm, mm`](https://www.felixcloutier.com/x86/PADDSB:PADDSW.html)
///
/// # Examples
///
/// ```
/// let min = i8::min_value();
/// let max = i8::max_value();
/// let a = i8x8::new(-100, -1,  1, 100, -1,  0, 1, 0);
/// let b = i8x8::new(-100,  1, -1, 100,  0, -1, 0, 1);
/// let e = i8x8::new( min,  0,  0, max, -1, -1, 1, 1);
/// assert_eq!(e, _m_paddsb(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddsb(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_m_paddsb(transmute(a), transmute(b)))
}

/// Add packed signed integers with signed saturation
///
/// When an individual word result is beyond the range of a signed word integer
/// (that is, greater than `7FFFH` or less than `8000H`), the saturated value of
/// `7FFFH` or `8000H`, respectively, is written to the destination operand.
///
/// # Instruction
///
/// [`paddsw mm, mm`](https://www.felixcloutier.com/x86/PADDSB:PADDSW.html)
///
/// # Examples
///
/// ```
/// let min = i16::min_value();
/// let max = i16::max_value();
/// let a = i16x4::new(-32_000, 32_000,  4, 0);
/// let b = i16x4::new(-32_000, 32_000, -5, 1);
/// let e = i16x4::new(min, max, -1, 1);
/// assert_eq!(e, _mm_adds_pi16(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_adds_pi16(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_mm_adds_pi16(
        transmute(a),
        transmute(b),
    ))
}

/// Add packed signed integers with signed saturation
///
/// When an individual word result is beyond the range of a signed word integer
/// (that is, greater than `7FFFH` or less than `8000H`), the saturated value of
/// `7FFFH` or `8000H`, respectively, is written to the destination operand.
///
/// # Instruction
///
/// [`paddsw mm, mm`](https://www.felixcloutier.com/x86/PADDSB:PADDSW.html)
///
/// # Examples
///
/// ```
/// let min = i16::min_value();
/// let max = i16::max_value();
/// let a = i16x4::new(-32_000, 32_000,  4, 0);
/// let b = i16x4::new(-32_000, 32_000, -5, 1);
/// let e = i16x4::new(min, max, -1, 1);
/// assert_eq!(e, _m_paddsw(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddsw(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_m_paddsw(transmute(a), transmute(b)))
}

/// Add packed unsigned integers with unsigned saturation
///
/// When an individual byte result is beyond the range of a signed byte integer
/// (that is, greater than `7FH` or less than `80H`), the saturated value of
/// `7FH` or `80H`, respectively, is written to the destination operand.
///
/// # Instruction
///
/// [`paddusb mm, mm`](http://www.felixcloutier.com/x86/PADDUSB:PADDUSW.html)
///
/// # Examples
///
/// ```
/// let max = u8::max_value();
/// let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 200);
/// let b = u8x8::new(0, 10, 20, 30, 40, 50, 60, 200);
/// let e = u8x8::new(0, 11, 22, 33, 44, 55, 66, max);
/// assert_eq!(e, _mm_adds_pu8(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_adds_pu8(a: u8x8, b: u8x8) -> u8x8 {
    transmute(::arch::_mm_adds_pu8(transmute(a), transmute(b)))
}

/// Add packed unsigned integers with unsigned saturation
///
/// When an individual byte result is beyond the range of a signed byte integer
/// (that is, greater than `7FH` or less than `80H`), the saturated value of
/// `7FH` or `80H`, respectively, is written to the destination operand.
///
/// # Instruction
///
/// [`paddusb mm, mm`](http://www.felixcloutier.com/x86/PADDUSB:PADDUSW.html)
///
/// # Examples
///
/// ```
/// let max = u8::max_value();
/// let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 200);
/// let b = u8x8::new(0, 10, 20, 30, 40, 50, 60, 200);
/// let e = u8x8::new(0, 11, 22, 33, 44, 55, 66, max);
/// assert_eq!(e, _m_paddusb(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddusb(a: u8x8, b: u8x8) -> u8x8 {
    transmute(::arch::_m_paddusb(transmute(a), transmute(b)))
}

/// Add packed unsigned integers with unsigned saturation
///
/// When an individual word result is beyond the range of a signed word integer
/// (that is, greater than `7FFFH` or less than `8000H`), the saturated value of
/// `7FFFH` or `8000H`, respectively, is written to the destination operand.
///
/// # Instruction
///
/// [`paddusw mm, mm`](http://www.felixcloutier.com/x86/PADDUSB:PADDUSW.html)
///
/// # Examples
///
/// ```
/// let max = u16::max_value();
/// let a = u16x4::new(0, 1, 2, 60_000);
/// let b = u16x4::new(0, 10, 20, 60_000);
/// let e = u16x4::new(0, 11, 22, max);
/// assert_eq!(e, _m_paddusb(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_adds_pu16(a: u16x4, b: u16x4) -> u16x4 {
    transmute(::arch::_mm_adds_pu16(
        transmute(a),
        transmute(b),
    ))
}

/// Add packed unsigned integers with unsigned saturation
///
/// When an individual word result is beyond the range of a signed word integer
/// (that is, greater than `7FFFH` or less than `8000H`), the saturated value of
/// `7FFFH` or `8000H`, respectively, is written to the destination operand.
///
/// # Instruction
///
/// [`paddusw mm, mm`](http://www.felixcloutier.com/x86/PADDUSB:PADDUSW.html)
///
/// # Examples
///
/// ```
/// let max = u16::max_value();
/// let a = u16x4::new(0, 1, 2, 60_000);
/// let b = u16x4::new(0, 10, 20, 60_000);
/// let e = u16x4::new(0, 11, 22, max);
/// assert_eq!(e, _m_paddusb(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_paddusw(a: u16x4, b: u16x4) -> u16x4 {
    transmute(::arch::_m_paddusw(transmute(a), transmute(b)))
}

/// Subtract packed integers
///
/// When an individual result is too large or too small to be represented in a
/// byte, the result is wrapped around and the low 8 bits are written to the
/// destination element.
///
/// # Instruction
///
/// [`psubb mm, mm`](http://www.felixcloutier.com/x86/PSUBB:PSUBW:PSUBD.html)
///
/// # Examples
///
/// ```
/// let max = i8::max_value();
/// let min = i8::min_value();
/// let a = i8x8::new( 0,  0,  1,  1,   -1,   -1, 0,     0);
/// let b = i8x8::new(-1,  1, -2,  2,  100, -100, min, max);
/// let e = i8x8::new( 1, -1,  3, -1, -101,   99, max, min);
/// assert_eq!(e, _mm_sub_pi8(a, b));
///
/// let max = u8::max_value();
/// let a = u8x8::new(  0, 1,   1,  2,       1, 100,   0, max);
/// let b = u8x8::new(  1, 0,   2,  1,     100,   1, max,   0);
/// let e = u8x8::new(max, 1, max,  1, max-100,  99, max, max);
/// assert_eq!(e, _mm_sub_pi8(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_sub_pi8<T: IU8x8>(a: T, b: T) -> T {
    T::from_m64(::arch::_mm_sub_pi8(a.as_m64(), b.as_m64()))
}

/// Subtract packed integers
///
/// When an individual result is too large or too small to be represented in a
/// byte, the result is wrapped around and the low 8 bits are written to the
/// destination element.
///
/// # Instruction
///
/// [`psubb mm, mm`](http://www.felixcloutier.com/x86/PSUBB:PSUBW:PSUBD.html)
///
/// # Examples
///
/// ```
/// let max = i8::max_value();
/// let min = i8::min_value();
/// let a = i8x8::new( 0,  0,  1,  1,   -1,   -1, 0,     0);
/// let b = i8x8::new(-1,  1, -2,  2,  100, -100, min, max);
/// let e = i8x8::new( 1, -1,  3, -1, -101,   99, max, min);
/// assert_eq!(e, _m_psubb(a, b));
///
/// let max = u8::max_value();
/// let a = u8x8::new(  0, 1,   1,  2,       1, 100,   0, max);
/// let b = u8x8::new(  1, 0,   2,  1,     100,   1, max,   0);
/// let e = u8x8::new(max, 1, max,  1, max-100,  99, max, max);
/// assert_eq!(e, _m_psubb(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubb<T: IU8x8>(a: T, b: T) -> T {
    _mm_sub_pi8(a, b)
}

/// Subtract packed integers
///
/// When an individual result is too large or too small to be represented in a
/// byte, the result is wrapped around and the low 16 bits are written to the
/// destination element.
///
/// # Instruction
///
/// [`psubw mm, mm`](http://www.felixcloutier.com/x86/PSUBB:PSUBW:PSUBD.html)
///
/// # Examples
///
/// ```
/// let max = i16::max_value();
/// let min = i16::min_value();
/// let a = i16x4::new( 0,  0, 0,     0);
/// let b = i16x4::new(-1,  1, min, max);
/// let e = i16x4::new( 1, -1, max, min);
/// assert_eq!(e, _mm_sub_pi16(a, b));
///
/// let max = u16::max_value();
/// let a = u16x4::new(  1, 1,   0,   0    );
/// let b = u16x4::new(  2, 0,   1, max    );
/// let e = u16x4::new(max, 1, max, max - 1);
/// assert_eq!(e, _mm_sub_pi16(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_sub_pi16<T: IU16x4>(a: T, b: T) -> T {
    T::from_m64(::arch::_mm_sub_pi16(a.as_m64(), b.as_m64()))
}

/// Subtract packed integers
///
/// When an individual result is too large or too small to be represented in a
/// byte, the result is wrapped around and the low 16 bits are written to the
/// destination element.
///
/// # Instruction
///
/// [`psubw mm, mm`](http://www.felixcloutier.com/x86/PSUBB:PSUBW:PSUBD.html)
///
/// # Examples
///
/// ```
/// let max = i16::max_value();
/// let min = i16::min_value();
/// let a = i16x4::new( 0,  0, 0,     0);
/// let b = i16x4::new(-1,  1, min, max);
/// let e = i16x4::new( 1, -1, max, min);
/// assert_eq!(e, _m_psubw(a, b));
///
/// let max = u16::max_value();
/// let a = u16x4::new(  1, 1,   0,   0    );
/// let b = u16x4::new(  2, 0,   1, max    );
/// let e = u16x4::new(max, 1, max, max - 1);
/// assert_eq!(e, _m_psubw(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubw(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_m_psubw(transmute(a), transmute(b)))
}

/// Subtract packed integers
///
/// When an individual result is too large or too small to be represented in a
/// byte, the result is wrapped around and the low 32 bits are written to the
/// destination element.
///
/// # Instruction
///
/// [`psubd mm, mm`](http://www.felixcloutier.com/x86/PSUBB:PSUBW:PSUBD.html)
///
/// # Examples
///
/// ```
/// let max = i32::max_value();
/// let min = i32::min_value();
/// let a = i32x2::new(0,     0);
/// let b = i32x2::new(min, max);
/// let e = i32x2::new(max, min);
/// assert_eq!(e, _mm_sub_pi32(a, b));
///
/// let max = u32::max_value();
/// let a = u32x2::new(  1,       0);
/// let b = u32x2::new(  2,     max);
/// let e = u32x2::new(max, max - 1);
/// assert_eq!(e, _mm_sub_pi32(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_sub_pi32(a: i32x2, b: i32x2) -> i32x2 {
    transmute(::arch::_mm_sub_pi32(transmute(a), transmute(b)))
}

/// Subtract packed integers
///
/// When an individual result is too large or too small to be represented in a
/// byte, the result is wrapped around and the low 32 bits are written to the
/// destination element.
///
/// # Instruction
///
/// [`psubd mm, mm`](http://www.felixcloutier.com/x86/PSUBB:PSUBW:PSUBD.html)
///
/// # Examples
///
/// ```
/// let max = i32::max_value();
/// let min = i32::min_value();
/// let a = i32x2::new(0,     0);
/// let b = i32x2::new(min, max);
/// let e = i32x2::new(max, min);
/// assert_eq!(e, _m_psubd(a, b));
///
/// let max = u32::max_value();
/// let a = u32x2::new(  1,       0);
/// let b = u32x2::new(  2,     max);
/// let e = u32x2::new(max, max - 1);
/// assert_eq!(e, _m_psubd(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubd(a: i32x2, b: i32x2) -> i32x2 {
    transmute(::arch::_m_psubd(transmute(a), transmute(b)))
}

/// Subtract packed signed integers with signed saturation
///
/// When an individual byte result is beyond the range of a signed byte integer
/// (that is, greater than `7FH` or less than `80H`), the saturated value of `7FH` or
/// `80H`, respectively, is written to the destination operand.
///
/// # Instruction
///
/// [`psubsb mm, mm`](https://www.felixcloutier.com/x86/PSUBSB:PSUBSW.html)
///
/// # Examples
///
/// ```
/// let max = i8::max_value();
/// let min = i8::min_value();
/// let a = i8x8::new(-100,  100,   0,    0,  0,  0, -5,  5);
/// let b = i8x8::new( 100, -100, min,  127, -1,  1,  3, -3);
/// let e = i8x8::new( min,  max, max, -127,  1, -1, -8,  8);
/// assert_eq!(e, _mm_subs_pi8(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_subs_pi8(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_mm_subs_pi8(transmute(a), transmute(b)))
}

/// Subtract packed signed integers with signed saturation
///
/// When an individual byte result is beyond the range of a signed byte integer
/// (that is, greater than `7FH` or less than `80H`), the saturated value of `7FH` or
/// `80H`, respectively, is written to the destination operand.
///
/// # Instruction
///
/// [`psubsb mm, mm`](https://www.felixcloutier.com/x86/PSUBSB:PSUBSW.html)
///
/// # Examples
///
/// ```
/// let max = i8::max_value();
/// let min = i8::min_value();
/// let a = i8x8::new(-100,  100,   0,    0,  0,  0, -5,  5);
/// let b = i8x8::new( 100, -100, min,  127, -1,  1,  3, -3);
/// let e = i8x8::new( min,  max, max, -127,  1, -1, -8,  8);
/// assert_eq!(e, _m_psubsb(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubsb(a: i8x8, b: i8x8) -> i8x8 {
    transmute(::arch::_m_psubsb(transmute(a), transmute(b)))
}

/// Subtract packed signed integers with signed saturation
///
/// When an individual word result is beyond the range of a signed word integer
/// (that is, greater than `7FFFH` or less than `8000H`), the saturated value of
/// `7FFFH` or `8000H`, respectively, is written to the destination operand.
///
/// # Instruction
///
/// [`psubsw mm, mm`](https://www.felixcloutier.com/x86/PSUBSB:PSUBSW.html)
///
/// # Examples
///
/// ```
/// let max = i16::max_value();
/// let min = i16::min_value();
/// let a = i16x4::new(-20_000,  20_000,   0,    1);
/// let b = i16x4::new( 20_000, -20_000, min,  127);
/// let e = i16x4::new(    min,     max, max, -126);
/// assert_eq!(e, _mm_subs_pi16(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_subs_pi16(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_mm_subs_pi16(
        transmute(a),
        transmute(b),
    ))
}

/// Subtract packed signed integers with signed saturation
///
/// When an individual word result is beyond the range of a signed word integer
/// (that is, greater than `7FFFH` or less than `8000H`), the saturated value of
/// `7FFFH` or `8000H`, respectively, is written to the destination operand.
///
/// # Instruction
///
/// [`psubsw mm, mm`](https://www.felixcloutier.com/x86/PSUBSB:PSUBSW.html)
///
/// # Examples
///
/// ```
/// let max = i16::max_value();
/// let min = i16::min_value();
/// let a = i16x4::new(-20_000,  20_000,   0,    1);
/// let b = i16x4::new( 20_000, -20_000, min,  127);
/// let e = i16x4::new(    min,     max, max, -126);
/// assert_eq!(e, _m_psubsw(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubsw(a: i16x4, b: i16x4) -> i16x4 {
    transmute(::arch::_m_psubsw(transmute(a), transmute(b)))
}

/// Subtract packed unsigned integers with unsigned saturation
///
/// When an individual byte result is less than zero, the saturated value of
/// `00H` is written to the destination operand.
///
/// # Instruction
///
/// [`psubusb mm, mm`](https://www.felixcloutier.com/x86/PSUBUSB:PSUBUSW.html)
///
/// # Examples
///
/// ```
/// let max = u8::max_value();
/// let a = u8x8::new(50, 10, 20, 30, 40, 60, 70, 80);
/// let b = u8x8::new(60, 20, 30, 40, 30, 20, 10,  0);
/// let e = u8x8::new( 0,  0,  0,  0, 10, 40, 60, 80);
/// assert_eq!(e, _mm_subs_pu8(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_subs_pu8(a: u8x8, b: u8x8) -> u8x8 {
    transmute(::arch::_mm_subs_pu8(transmute(a), transmute(b)))
}

/// Subtract packed unsigned integers with unsigned saturation
///
/// When an individual byte result is less than zero, the saturated value of
/// `00H` is written to the destination operand.
///
/// # Instruction
///
/// [`psubusb mm, mm`](https://www.felixcloutier.com/x86/PSUBUSB:PSUBUSW.html)
///
/// # Examples
///
/// ```
/// let max = u8::max_value();
/// let a = u8x8::new(50, 10, 20, 30, 40, 60, 70, 80);
/// let b = u8x8::new(60, 20, 30, 40, 30, 20, 10,  0);
/// let e = u8x8::new( 0,  0,  0,  0, 10, 40, 60, 80);
/// assert_eq!(e, _m_psubusb(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubusb(a: u8x8, b: u8x8) -> u8x8 {
    transmute(::arch::_m_psubusb(transmute(a), transmute(b)))
}

/// Subtract packed unsigned integers with unsigned saturation
///
/// When an individual word result is less than zero, the saturated value of
/// `0000H` is written to the destination operand.
///
/// # Instruction
///
/// [`psubusw mm, mm`](https://www.felixcloutier.com/x86/PSUBUSB:PSUBUSW.html)
///
/// # Examples
///
/// ```
/// let max = u16::max_value();
/// let a = u16x4::new(10000, 200, 0, 44444);
/// let b = u16x4::new(20000, 300, 1, 11111);
/// let e = u16x4::new(    0,   0, 0, 33333);
/// assert_eq!(e, _mm_subs_pu16(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_subs_pu16(a: u16x4, b: u16x4) -> u16x4 {
    transmute(::arch::_mm_subs_pu16(
        transmute(a),
        transmute(b),
    ))
}

/// Subtract packed unsigned integers with unsigned saturation
///
/// When an individual word result is less than zero, the saturated value of
/// `0000H` is written to the destination operand.
///
/// # Instruction
///
/// [`psubusw mm, mm`](https://www.felixcloutier.com/x86/PSUBUSB:PSUBUSW.html)
///
/// # Examples
///
/// ```
/// let max = u16::max_value();
/// let a = u16x4::new(10000, 200, 0, 44444);
/// let b = u16x4::new(20000, 300, 1, 11111);
/// let e = u16x4::new(    0,   0, 0, 33333);
/// assert_eq!(e, _m_psubusw(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _m_psubusw(a: u16x4, b: u16x4) -> u16x4 {
    transmute(::arch::_m_psubusw(transmute(a), transmute(b)))
}

/// Pack with signed saturation
///
/// Converts packed signed word integers in `a` and `b` into packed signed byte
/// integers using signed saturation to handle overflow conditions beyond the
/// range of signed byte integers. If the signed doubleword value is beyond the
/// range of an unsigned word (i.e. greater than `7FH` or less than `80H`), the
/// saturated signed byte integer value of `7FH` or `80H`, respectively, is
/// stored in the destination.
///
/// # Instruction
///
/// [`packsswb mm, mm`](https://www.felixcloutier.com/x86/PACKSSWB:PACKSSDW.html)
///
/// # Examples
///
/// ```
/// let min = i8::min_value();
/// let max = i8::max_value();
/// let a = i16x4::new(-1, 2, -200, 4);
/// let b = i16x4::new(-5, 200, -7, 8);
/// let e = i8x8::new(-1, 2, min, 4, -5, max, -7, 8);
/// assert_eq!(e, _mm_packs_pi16(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_packs_pi16(a: i16x4, b: i16x4) -> i8x8 {
    transmute(::arch::_mm_packs_pi16(
        transmute(a),
        transmute(b),
    ))
}

/// Pack with signed saturation
///
/// Converts packed signed word integers in `a` and `b` into packed signed byte
/// integers using signed saturation to handle overflow conditions beyond the
/// range of signed byte integers. If the signed doubleword value is beyond the
/// range of an unsigned word (i.e. greater than `7FFFH` or less than `8000H`),
/// the saturated signed byte integer value of `7FFFH` or `8000H`, respectively,
/// is stored in the destination.
///
/// # Instruction
///
/// [`packssdw mm, mm`](https://www.felixcloutier.com/x86/PACKSSWB:PACKSSDW.html)
///
/// # Examples
///
/// ```
/// let min = i16::min_value();
/// let max = i16::max_value();
/// let a = i32x2::new(-1,  40_000);
/// let b = i32x2::new(-5, -40_000);
/// let e = i16x4::new(-1, max, -5, min);
/// assert_eq!(e, _mm_packs_pi32(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_packs_pi32(a: i32x2, b: i32x2) -> i16x4 {
    transmute(::arch::_mm_packs_pi32(
        transmute(a),
        transmute(b),
    ))
}

/// Compare packed signed integers for greater than
///
/// If an element in `a` is greater than the corresponding element in `b`, the
/// corresponding element of the result is set to all `1`s; otherwise, it is set
/// to all `0`s.
///
/// # Instruction
///
/// [`pcmpgtb mm, mm`](http://www.felixcloutier.com/x86/PCMPGTB:PCMPGTW:PCMPGTD.html)
///
/// # Examples
///
/// ```
/// let f = 0_i8;
/// let t = -1_i8;
/// let a = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
/// let b = i8x8::new(8, 7, 6, 5, 4, 3, 2, 1);
/// let e = i8x8::new(f, f, f, f, f, t, t, t);
/// assert_eq!(e, _mm_cmpgt_pi8(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_cmpgt_pi8(a: i8x8, b: i8x8) -> i8x8 {
    // FIXME: return m8x8
    transmute(::arch::_mm_cmpgt_pi8(
        transmute(a),
        transmute(b),
    ))
}

/// Compare packed signed integers for greater than
///
/// If an element in `a` is greater than the corresponding element in `b`, the
/// corresponding element of the result is set to all `1`s; otherwise, it is set
/// to all `0`s.
///
/// # Instruction
///
/// [`pcmpgtw mm, mm`](http://www.felixcloutier.com/x86/PCMPGTB:PCMPGTW:PCMPGTD.html)
///
/// # Examples
///
/// ```
/// let f = 0_i16;
/// let t = -1_i16;
/// let a = i16x4::new(0, 1, 5, 6);
/// let b = i16x4::new(8, 7, 3, 2);
/// let e = i16x4::new(f, f, t, t);
/// assert_eq!(e, _mm_cmpgt_pi16(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_cmpgt_pi16(a: i16x4, b: i16x4) -> i16x4 {
    // FIXME: return m16x4
    transmute(::arch::_mm_cmpgt_pi16(
        transmute(a),
        transmute(b),
    ))
}

/// Compare packed signed integers for greater than
///
/// If an element in `a` is greater than the corresponding element in `b`, the
/// corresponding element of the result is set to all `1`s; otherwise, it is set
/// to all `0`s.
///
/// # Instruction
///
/// [`pcmpgtd mm, mm`](http://www.felixcloutier.com/x86/PCMPGTB:PCMPGTW:PCMPGTD.html)
///
/// # Examples
///
/// ```
/// let f = 0_i32;
/// let t = -1_i32;
/// let a = i32x2::new(0, 5);
/// let b = i32x2::new(8, 3);
/// let e = i32x2::new(f, t);
/// assert_eq!(e, _mm_cmpgt_pi32(a, b));
/// ```
#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn _mm_cmpgt_pi32(a: i32x2, b: i32x2) -> i32x2 {
    // FIXME: return m32x2
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
