# Type-safe `std::arch` SIMD intrinsics

> Work-in-progress

[![Travis-CI Status]][travis] [![Appveyor Status]][appveyor] [![Latest Version]][crates.io] [![docs]][docs.rs]

This library provides a zero-overhead abstraction over `std::arch` to enforce
many intrinsic pre-conditions at compile-time.

Consider the following `std::arch` intrinsics:

* `_mm_castps_si128(a: __m128) -> __m128i`. Its name is informative, but its
  signature is not. This intrinsic casts a `[4 x f32]` 128-bit wide floating
  point vector into a `[4 x i32]` integer vector, and passing it any other kind
  of input like a `[2 x f64]` vector is going to produce garbage. This library
  uses the `std::simd` portable vector types to prevent these errors from
  happening while simulatenously making the intrinsics more convenient to use:
  
  ```rust
  // With std::arch this is an error:
  let y: i32x4 = std::arch::_mm_castps_si128(f32x4::splat(3.14));
  // Two transmutes are required:
  let y: i32x4 = transmute(std::arch::_mm_castps_si128(transmute(f32x4::splat(3.14))));
  // The transmute easily allow mistakes:
  let y: i64x2 = transmute(std::arch::_mm_castps_si128(transmute(f64x2::splat(3.14))));
  
  // With typed_arch this works correctly:
  let y: i32x4 = _mm_castps_si128(f32x4::splat(3.14));
  // And this does not compile
  let y: i64x2 = _mm_castps_si128(f64x2::splat(3.14)); // ERROR: expected f32x4
  ```

* `_mm_store_pd(mem_addr: *mut f64, a: __m128d)`. This intrinsic requires
  `mem_addr` to be aligned to a 16-byte boundary. Otherwise, a
  general-protection exception will be generated. When this happens, chances are
  that your program will crash. With `typed_arch` passing this intrinsic an
  unaligned pointer is a compilation error.

*  `_mm_round_ps(a: __m128, rounding: i32)`. This intrinsic requires a
   `rounding` parameter which is actually is a bit-set for which only certain
   bit patterns make sense. With `typed_arch` passing this intrinsic an invalid
   rounding mode is a compilation error.

Many _many_ other pitfalls like these are all prevented by `typed_arch` at compile-time.

# Work in progress

The following table displays which target features are implemented and documented:

|   feature   | Impl | Docs |
|-------------|------|------|
| `mmx`       |  ✓   | wip  |
| `sse`       |  ✗   | ✗    |
| `sse2`      |  ✓   | ✗    |
| `sse3`      |  ✗   | ✗    |
| `ssse3`     |  ✗   | ✗    |
| `sse41`     |  ✗   | ✗    |
| `sse42`     |  ✗   | ✗    |
| `sse4a`     |  ✗   | ✗    |
| `avx`       |  ✗   | ✗    |
| `avx2`      |  ✗   | ✗    |
| `aes`       |  ✗   | ✗    |
| `abm`       |  ✗   | ✗    |
| `tbm`       |  ✗   | ✗    |
| `fxsr`      |  ✗   | ✗    |
| `bswap`     |  ✗   | ✗    |
| `eflags`    |  ✗   | ✗    |
| `cpuid`     |  ✗   | ✗    |
| `pclmulqdq` |  ✗   | ✗    |
| `rdrand`    |  ✗   | ✗    |
| `rdtsc`     |  ✗   | ✗    |
| `sha`       |  ✗   | ✗    |
| `xsave`     |  ✗   | ✗    |



[travis]: https://travis-ci.org/gnzlbg/typed_arch
[Travis-CI Status]: https://travis-ci.org/gnzlbg/typed_arch.svg?branch=master
[appveyor]: https://ci.appveyor.com/project/gnzlbg/typed_arch/branch/master
[Appveyor Status]: https://ci.appveyor.com/api/projects/status/lh0895i13e83d2q9?svg=true
[Latest Version]: https://img.shields.io/crates/v/typed_arch.svg
[crates.io]: https://crates.io/crates/typed_arch
[docs]: https://docs.rs/typed_arch/badge.svg
[docs.rs]: https://docs.rs/typed_arch/
