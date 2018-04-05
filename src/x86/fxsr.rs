//! FXSR floating-point context fast save and restor.

/// 512-byte wide 16-byte aligned floating-point context.
#[derive(Copy,Clone,PartialEq)]
#[repr(align(16))]
pub struct fxsr([u8; 512]);

impl ::fmt::Debug for fxsr {
    fn fmt(&self, f: &mut ::fmt::Formatter) -> ::fmt::Result {
        write!(f, "[")?;
        for i in 0..self.0.len() {
            write!(f, "{}", self.0[i])?;
            if i != self.0.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

impl fxsr {
    /// Saves the `x87` FPU, `MMX` technology, `XMM`, and `MXCSR` registers.
    pub unsafe fn save() -> Self {
        let mut x: fxsr = ::mem::uninitialized();
        ::arch::_fxsave(&mut x.0 as *mut u8);
        x
    }
    /// Restores the `XMM`, `MMX`, `MXCSR`, and `x87` FPU registers.
    pub unsafe fn restore(&self) {
        ::arch::_fxrstor(&mut self.0 as *const u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fxsr() {
        #[cfg(target_arch = "x86")] {
            if !is_x86_feature_detected!("fxsr") { return; }
        }
        #[cfg(target_arch = "x86_64")] {
            if !is_x86_64_feature_detected!("fxsr") { return; }
        }

        let a = fxsr::save();
        a.store();
        let b = fxsr::save();
        assert_eq!(a, b);
    }
}
