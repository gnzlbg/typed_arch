//! `x86` intrinsics

#[macro_use]
mod macros;

mod abm;
// pub use self::abm::*;

// FIXME: requires i128x1
// mod aes;
// // pub use self::aes::*;

mod avx;
pub use self::avx::*;

// mod avx2;
// // pub use self::avx2::*;

// mod bmi1;
// pub use self::bmi1::*;

// mod bmi2;
// pub use self::bmi2::*;

mod bswap;
pub use self::bswap::*;

mod cpuid;
pub use self::cpuid::*;

mod eflags;
pub use self::eflags::*;

// mod fxsr;
// pub use self::fxsr::*;

// mod mmx;
// pub use self::mmx::*;

// mod pclmulqdq;
// pub use self::pclmulqdq::*;

mod rdrand;
pub use self::rdrand::*;

// mod rdtsc;
// pub use self::rdtsc::*;

// mod sha;
// pub use self::sha::*;

// mod sse;
// pub use self::sse::*;

mod sse2;
pub use self::sse2::*;

// mod sse3;
// pub use self::sse3::*;

// mod sse41;
// pub use self::sse41::*;

// mod sse42;
// pub use self::sse42::*;

// mod ssse3;
// pub use self::ssse3::*;

// mod tbm;
// pub use self::tbm::*;

// mod xsave;
// pub use self::xsave::*;
