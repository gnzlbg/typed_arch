//! `x86` intrinsics

#[macro_use]
mod macros;

mod abm;
// pub use self::abm::*;

// mod aes;
// // pub use self::aes::*;

mod avx; // FIXME: unfinished
pub use self::avx::*; // FIXME: unfinished

mod avx2; // FIXME: unfinished
pub use self::avx2::*;

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

// mod fxsr; // FIXME: requires updating std::arch with stdsimd
// pub use self::fxsr::*;

mod mmx;
pub use self::mmx::*;

// mod pclmulqdq;
// pub use self::pclmulqdq::*;

mod rdrand;
pub use self::rdrand::*;

// mod rdtsc;
// pub use self::rdtsc::*;

// mod sha;
// pub use self::sha::*;

mod sse; // FIXME: unfinished
pub use self::sse::*;

mod sse2;
pub use self::sse2::*;

mod sse3; // FIXME: unfinished
pub use self::sse3::*;

mod sse41; // FIXME: unfinished
pub use self::sse41::*;

// mod sse42;
// pub use self::sse42::*;

mod ssse3; // FIXME: unfinished
pub use self::ssse3::*;

// mod tbm;
// pub use self::tbm::*;

// mod xsave;
// pub use self::xsave::*;
