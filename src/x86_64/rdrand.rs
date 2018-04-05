//! RDRAND and RDSEED instructions.

/// Returns a hardware generated 64-bit random value
#[inline]
#[target_feature(enable = "rdrand")]
pub unsafe fn _rdrand64_step() -> Option<u64> {
    let mut v: u64 = 0;
    if ::arch::_rdrand64_step(&mut v) == 1 {
        Some(v)
    } else {
        None
    }
}

/// Returns a 64-bit NIST SP800-90B and SP800-90C compliant random value.
#[inline]
#[target_feature(enable = "rdseed")]
pub unsafe fn _rdseed64_step() -> Option<u64> {
    let mut v: u64 = 0;
    if ::arch::_rdseed64_step(&mut v) == 1 {
        Some(v)
    } else {
        None
    }
}
