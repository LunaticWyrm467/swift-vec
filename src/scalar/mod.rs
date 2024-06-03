use core::fmt::{ Display, Debug };

use num_traits::{ Num, Signed, Float, FloatConst, PrimInt, NumCast };


/*
    Trait
        Definitions
*/


/// Implements common behaviours and additional operations for all primitives.
pub trait Scalar: Clone + Copy + Num + Default + PartialOrd + Display + Debug + NumCast {

    /// Returns the minimum value of this value and another.
    /// This is implemented manually to not rely on the Ord trait.
    fn min(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }

    /// Returns the maximum value of this value and another.
    /// This is implemented manually to not rely on the Ord trait.
    fn max(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }

    /// Clamps this value between a provided minimum and maximum.
    /// This is implemented manually to not rely on the Ord trait.
    fn clamp(self, min: Self, max: Self) -> Self {
        self.min(max).max(min)
    }

    /// A simple linear interpolation between two values.
    /// Samples at the point `t` between `self` and `other`.
    fn lerp(self, other: Self, t: Self) -> Self {
        self + (other - self) * t
    }
}

/// Implements unique integer operations for all integer primitives.
pub trait IntScalar<T: IntScalar<T>>: Scalar + Ord + PrimInt + IntUnique<T> {}

/// Implements unique operations for all signed primitives.
pub trait SignedScalar: Scalar + Signed {

    /// Calculates the derivative of the Bézier curve set by this scalar and the given control and terminal points
    /// at position `t`.
    fn bezier_derivative(self, control_1: Self, control_2: Self, terminal: Self, t: Self) -> Self {

        // Define some commonly used constants.
        let t_3: Self = Self::from(3).unwrap();
        let t_6: Self = Self::from(6).unwrap();

        // Formula from https://en.wikipedia.org/wiki/Bézier_curve
		let omt:  Self = Self::one() - t;
		let omt2: Self = omt * omt;
		let t2:   Self = t * t;

		(control_1 - self) * t_3 * omt2 + (control_2 - control_1) * t_6 * omt * t + (terminal - control_2) * t_3 * t2
    }

    /// Calculates the point on the Bézier curve set by this scalar and the given control and terminal points
    /// at position `t`.
    fn bezier_sample(self, control_1: Self, control_2: Self, terminal: Self, t: Self) -> Self {

        // Define some commonly used constants.
        let t_3: Self = Self::from(3).unwrap();

        // Formula from https://en.wikipedia.org/wiki/Bézier_curve
        let omt:  Self = Self::one() - t;
		let omt2: Self = omt * omt;
		let omt3: Self = omt2 * omt;
		let t2:   Self = t * t;
		let t3:   Self = t2 * t;

		self * omt3 + control_1 * omt2 * t * t_3 + control_2 * omt * t2 * t_3 + terminal * t3
    }

    /// Calculates and samples the cubic interpolation between this scalar and another
    /// given `pre_start` and `post_terminal` scalars as handles, and a given `t` value.
    fn cubic_interpolate(self, b: Self, pre_a: Self, post_b: Self, weight: Self) -> Self {

        // Define some commonly used constants.
        let t_05: Self = Self::from(0.5).unwrap();
        let t_2:  Self = Self::from(2.0).unwrap();
        let t_3:  Self = Self::from(3.0).unwrap();
        let t_4:  Self = Self::from(4.0).unwrap();
        let t_5:  Self = Self::from(5.0).unwrap();
        
        // Derived from https://github.com/godotengine/godot/blob/1952f64b07b2a0d63d5ba66902fd88190b0dcf08/core/math/math_funcs.h#L275
        t_05 * (
            (self * t_2) +
            (-pre_a + b) * weight +
            (t_2 * pre_a - t_5 * self + t_4 * b - post_b) * (weight * weight) +
            (-pre_a + t_3 * self - t_3 * b + post_b) * (weight * weight * weight)
        )
    }

    /// Similar to `cubic_interpolate`, but it has additional time parameters `terminal_t`, `pre_start_t`, and `post_terminal_t`.
    /// This can be smoother than `cubic_interpolate` in certain instances.
    fn cubic_interpolate_in_time(self, b: Self, pre_a: Self, post_b: Self, weight: Self, b_t: Self, pre_a_t: Self, post_b_t: Self) -> Self {

        // Define some commonly used constants.
        let t_0:  Self = Self::zero();
        let t_05: Self = Self::from(0.5).unwrap();
        let t_1:  Self = Self::one();
        
        // Formula of the Barry-Goldman method.
        let t:  Self = t_0.lerp(b_t, weight);
        let a1: Self = pre_a.lerp(self, if pre_a_t == t_0 { t_0 } else { (t - pre_a_t) / -pre_a_t });
        let a2: Self = self.lerp(b, if b_t == t_0 { t_05 } else { t / b_t });
        let a3: Self = b.lerp(post_b, if post_b_t - b_t == t_0 { t_1 } else { (t - b_t) / (post_b_t - b_t) });
        let b1: Self = a1.lerp(a2, if b_t - pre_a_t == t_0 { t_0 } else { (t - pre_a_t) / (b_t - pre_a_t) });
        let b2: Self = a2.lerp(a3, if post_b_t == t_0 { t_1 } else { t / post_b_t });
        b1.lerp(b2, if b_t == t_0 { t_05 } else { t / b_t })
    }
}

/// Implements unique operations for all floating point primitives.
pub trait FloatScalar: SignedScalar + Float + FloatConst {
    
    /// Modulates a value so that it stays between 0-1.
    fn fract(self) -> Self {
        self - self.floor()
    }
    
    /// Computes the inverse square root of a scalar.
    fn inv_sqrt(self) -> Self {
        Self::one() / self.sqrt()
    }

    /// A `smoothstep` implementation similar to that of OpenGL's, which uses smooth Hermite
    /// interpolation between the values `a` and `b` for `self.
    fn smoothstep(self, a: Self, b: Self) -> Self {
        let t_2: Self = Self::from(2).unwrap();
        let t_3: Self = Self::from(3).unwrap();

        let y: Self = (self - a) / (b - a).clamp(Self::zero(), Self::one());
	    y * y * (t_3 - (t_2 * y))
    }

    /// Checks if two floating point values are approximately equal.
    fn approx_eq(self, other: Self) -> bool {
        
        /*
         * Uses Godot's method:
         * https://github.com/godotengine/godot/blob/f4b0c7a1ea8d86c1dfd96478ca12ad1360903d9d/core/math/math_funcs.h#L342-L362
         */

        const EPSILON: f64 = 0.00001;
        
        if self == other {
			return true;
		}
    
        let     epsilon:   Self = Self::from(EPSILON).unwrap();
        let mut tolerance: Self = epsilon * self.abs();
		if tolerance < epsilon {
			tolerance = epsilon;
		}
		(self - other).abs() < tolerance
    }
}

/// Adds some additional operations featured in rust that are not available in the standard PrimInt trait for some odd reason.
pub trait IntUnique<T: IntScalar<T>> {
    fn ilog(self, base: T) -> T;
}


/*
    Trait
        Implementations
*/


impl <T: Clone + Copy + Num + Default + PartialOrd + Display + Debug + NumCast> Scalar for T {}
impl <T: Scalar + Ord + PrimInt + IntUnique<T>> IntScalar<T> for T {}
impl <T: Scalar + Signed> SignedScalar for T {}
impl <T: SignedScalar + Float + FloatConst> FloatScalar for T {}

impl IntUnique<u8> for u8 {
    fn ilog(self, base: u8) -> u8 {
        self.ilog(base) as u8
    }
}

impl IntUnique<u16> for u16 {
    fn ilog(self, base: u16) -> u16 {
        self.ilog(base) as u16
    }
}

impl IntUnique<u32> for u32 {
    fn ilog(self, base: u32) -> u32 {
        self.ilog(base) as u32
    }
}

impl IntUnique<u64> for u64 {
    fn ilog(self, base: u64) -> u64 {
        self.ilog(base) as u64
    }
}

impl IntUnique<u128> for u128 {
    fn ilog(self, base: u128) -> u128 {
        self.ilog(base) as u128
    }
}

impl IntUnique<usize> for usize {
    fn ilog(self, base: usize) -> usize {
        self.ilog(base) as usize
    }
}

impl IntUnique<isize> for isize {
    fn ilog(self, base: isize) -> isize {
        self.ilog(base) as isize
    }
}

impl IntUnique<i8> for i8 {
    fn ilog(self, base: i8) -> i8 {
        self.ilog(base) as i8
    }
}

impl IntUnique<i16> for i16 {
    fn ilog(self, base: i16) -> i16 {
        self.ilog(base) as i16
    }
}

impl IntUnique<i32> for i32 {
    fn ilog(self, base: i32) -> i32 {
        self.ilog(base) as i32
    }
}

impl IntUnique<i64> for i64 {
    fn ilog(self, base: i64) -> i64 {
        self.ilog(base) as i64
    }
}

impl IntUnique<i128> for i128 {
    fn ilog(self, base: i128) -> i128 {
        self.ilog(base) as i128
    }
}
