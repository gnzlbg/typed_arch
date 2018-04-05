//! `x86_64` intrinsics

#[path = "../x86/mod.rs"]
mod x86;

pub use self::x86::*;

mod abm;
pub use self::abm::*;

mod avx;
pub use self::avx::*;

// mod avx2;
// pub use self::avx2::*;

// mod bmi1;
// pub use self::bmi1::*;

// mod bmi2;
// pub use self::bmi2::*;

mod bswap;
pub use self::bswap::*;

// mod fxsr;
// pub use self::fxsr::*;

mod rdrand;
pub use self::rdrand::*;

// mod sse;
// pub use self::sse::*;

mod sse2;
pub use self::sse2::*;

// mod sse41;
// pub use self::sse41::*;

// mod sse42;
// pub use self::sse42::*;

// mod xsave;
// pub use self::xsave::*;
