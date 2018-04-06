//! Sealed traits

use arch::__m64;
use mem::transmute;
use simd::*;

macro_rules! impl_all {
    ($macro:ident: $($id:ident),*) => {
        $(
            $macro!($id);
        )*
    }
}

/// Any 64-bit wide vector type
pub trait Any64 {
    fn as_m64(self) -> __m64;
    fn from_m64(x: __m64) -> Self;
}

macro_rules! impl_any64 {
    ($id:ident) => {
        impl Any64 for $id {
            #[inline]
            fn as_m64(self) -> __m64 {
                unsafe { transmute(self) }
            }
            #[inline]
            fn from_m64(x: __m64) -> Self {
                unsafe { transmute(x) }
            }
        }
    };
}
impl_all!(
    impl_any64: i8x8,
    u8x8,
    i16x4,
    u16x4,
    i32x2,
    u32x2,
    __m64
);

/// Signed or Unsigned 64-bit wide vector type with 8 lanes
pub trait IU8x8 {
    fn as_m64(self) -> __m64;
    fn from_m64(x: __m64) -> Self;
}

macro_rules! impl_iu8x8 {
    ($id:ident) => {
        impl IU8x8 for $id {
            #[inline]
            fn as_m64(self) -> __m64 {
                unsafe { transmute(self) }
            }
            #[inline]
            fn from_m64(x: __m64) -> Self {
                unsafe { transmute(x) }
            }
        }
    };
}
impl_all!(impl_iu8x8: i8x8, u8x8, __m64);

/// Signed or Unsigned 64-bit wide vector type with 4 lanes
pub trait IU16x4 {
    fn as_m64(self) -> __m64;
    fn from_m64(x: __m64) -> Self;
}

macro_rules! impl_iu16x4 {
    ($id:ident) => {
        impl IU16x4 for $id {
            #[inline]
            fn as_m64(self) -> __m64 {
                unsafe { transmute(self) }
            }
            #[inline]
            fn from_m64(x: __m64) -> Self {
                unsafe { transmute(x) }
            }
        }
    };
}
impl_all!(impl_iu16x4: i16x4, u16x4, __m64);

/// Signed or Unsigned 64-bit wide vector type with 2 lanes
pub trait IU32x2 {
    fn as_m64(self) -> __m64;
    fn from_m64(x: __m64) -> Self;
}

macro_rules! impl_iu32x2 {
    ($id:ident) => {
        impl IU32x2 for $id {
            #[inline]
            fn as_m64(self) -> __m64 {
                unsafe { transmute(self) }
            }
            #[inline]
            fn from_m64(x: __m64) -> Self {
                unsafe { transmute(x) }
            }
        }
    };
}
impl_all!(impl_iu32x2: i32x2, u32x2, __m64);
