//! RDRAND and RDSEED instructions.

/// Returns a hardware generated 16-bit random value.
#[inline]
#[target_feature(enable = "rdrand")]
pub unsafe fn _rdrand16_step() -> Option<u16> {
    let mut v: u16 = 0;
    if ::arch::_rdrand16_step(&mut v) == 1 {
        Some(v)
    } else {
        None
    }
}

/// Read a hardware generated 32-bit random value.
#[inline]
#[target_feature(enable = "rdrand")]
pub unsafe fn _rdrand32_step() -> Option<u32> {
    let mut v: u32 = 0;
    if ::arch::_rdrand32_step(&mut v) == 1 {
        Some(v)
    } else {
        None
    }
}

/// Returns a 16-bit NIST SP800-90B and SP800-90C compliant random value.
#[inline]
#[target_feature(enable = "rdseed")]
pub unsafe fn _rdseed16_step() -> Option<u16> {
    let mut v: u16 = 0;
    if ::arch::_rdseed16_step(&mut v) == 1 {
        Some(v)
    } else {
        None
    }
}

/// Returns a 32-bit NIST SP800-90B and SP800-90C compliant random value.
#[inline]
#[target_feature(enable = "rdseed")]
pub unsafe fn _rdseed32_step() -> Option<u32> {
    let mut v: u32 = 0;
    if ::arch::_rdrand32_step(&mut v) == 1 {
        Some(v)
    } else {
        None
    }
}
