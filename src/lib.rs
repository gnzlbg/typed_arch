//! Typed `std::arch` intrinsics
#![feature(stdsimd, target_feature)]

#![no_std]

use core::{simd, mem};

mod arch {
    #[cfg(target_arch = "x86")]
    pub use core::arch::x86::*;

    #[cfg(target_arch = "x86_64")]
    pub use core::arch::x86_64::*;
}


#[cfg(target_arch = "x86")]
pub mod x86;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;
