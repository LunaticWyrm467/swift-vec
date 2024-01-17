use approx::RelativeEq;
use num_traits::{ Num, Signed, Float, FloatConst, PrimInt, FromPrimitive, ToPrimitive };


pub trait Scalar: Clone + Copy + Num + Default + PartialOrd + std::fmt::Display + std::fmt::Debug + FromPrimitive + ToPrimitive {}
pub trait IntScalar<T: IntScalar<T>>: Scalar + PrimInt + IntExtra<T> {}
pub trait SignedScalar: Scalar + Signed {}
pub trait FloatScalar: SignedScalar + Float + FloatConst + RelativeEq {}

pub trait IntExtra<T: IntScalar<T>> {
    fn ilog(self, base: T) -> T;
}


impl Scalar for u8 {}
impl IntScalar<u8> for u8 {}
impl IntExtra<u8> for u8 {
    fn ilog(self, base: u8) -> u8 {
        self.ilog(base) as u8
    }
}

impl Scalar for u16 {}
impl IntScalar<u16> for u16 {}
impl IntExtra<u16> for u16 {
    fn ilog(self, base: u16) -> u16 {
        self.ilog(base) as u16
    }
}

impl Scalar for u32 {}
impl IntScalar<u32> for u32 {}
impl IntExtra<u32> for u32 {
    fn ilog(self, base: u32) -> u32 {
        self.ilog(base) as u32
    }
}

impl Scalar for u64 {}
impl IntScalar<u64> for u64 {}
impl IntExtra<u64> for u64 {
    fn ilog(self, base: u64) -> u64 {
        self.ilog(base) as u64
    }
}

impl Scalar for u128 {}
impl IntScalar<u128> for u128 {}
impl IntExtra<u128> for u128 {
    fn ilog(self, base: u128) -> u128 {
        self.ilog(base) as u128
    }
}

impl Scalar for usize {}
impl IntScalar<usize> for usize {}
impl IntExtra<usize> for usize {
    fn ilog(self, base: usize) -> usize {
        self.ilog(base) as usize
    }
}

impl Scalar for isize {}
impl SignedScalar for isize {}
impl IntScalar<isize> for isize {}
impl IntExtra<isize> for isize {
    fn ilog(self, base: isize) -> isize {
        self.ilog(base) as isize
    }
}

impl Scalar for i8 {}
impl SignedScalar for i8 {}
impl IntScalar<i8> for i8 {}
impl IntExtra<i8> for i8 {
    fn ilog(self, base: i8) -> i8 {
        self.ilog(base) as i8
    }
}

impl Scalar for i16 {}
impl SignedScalar for i16 {}
impl IntScalar<i16> for i16 {}
impl IntExtra<i16> for i16 {
    fn ilog(self, base: i16) -> i16 {
        self.ilog(base) as i16
    }
}

impl Scalar for i32 {}
impl SignedScalar for i32 {}
impl IntScalar<i32> for i32 {}
impl IntExtra<i32> for i32 {
    fn ilog(self, base: i32) -> i32 {
        self.ilog(base) as i32
    }
}

impl Scalar for i64 {}
impl SignedScalar for i64 {}
impl IntScalar<i64> for i64 {}
impl IntExtra<i64> for i64 {
    fn ilog(self, base: i64) -> i64 {
        self.ilog(base) as i64
    }
}

impl Scalar for i128 {}
impl SignedScalar for i128 {}
impl IntScalar<i128> for i128 {}
impl IntExtra<i128> for i128 {
    fn ilog(self, base: i128) -> i128 {
        self.ilog(base) as i128
    }
}

impl Scalar for f32 {}
impl SignedScalar for f32 {}
impl FloatScalar for f32 {}

impl Scalar for f64 {}
impl SignedScalar for f64 {}
impl FloatScalar for f64 {}