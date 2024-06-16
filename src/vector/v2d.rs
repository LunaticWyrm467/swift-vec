//===================================================================================================================================================================================//
//
//  /$$    /$$                      /$$                                /$$$$$$  /$$$$$$$
// | $$   | $$                     | $$                               /$$__  $$| $$__  $$
// | $$   | $$ /$$$$$$   /$$$$$$$ /$$$$$$    /$$$$$$   /$$$$$$       |__/  \ $$| $$  \ $$
// |  $$ / $$//$$__  $$ /$$_____/|_  $$_/   /$$__  $$ /$$__  $$        /$$$$$$/| $$  | $$
//  \  $$ $$/| $$$$$$$$| $$        | $$    | $$  \ $$| $$  \__/       /$$____/ | $$  | $$
//   \  $$$/ | $$_____/| $$        | $$ /$$| $$  | $$| $$            | $$      | $$  | $$
//    \  $/  |  $$$$$$$|  $$$$$$$  |  $$$$/|  $$$$$$/| $$            | $$$$$$$$| $$$$$$$/
//     \_/    \_______/ \_______/   \___/   \______/ |__/            |________/|_______/
//
//===================================================================================================================================================================================//

//?
//? Created by LunaticWyrm467 and others.
//? 
//? All code is licensed under the MIT license.
//? Feel free to reproduce, modify, and do whatever.
//?

//!
//! A private submodule for the vector module that contains all of the implementations
//! for any of the non-shared behaviours of the 2D vector.
//!

#[cfg(feature = "glam")]
use glam::{ Vec2 as GVec2, vec2, DVec2, dvec2, UVec2, uvec2, IVec2, ivec2 };

use super::*;


/*
 * 2D Vector
 *      Indexing
 */


/// Simply represents an axis of a 2-dimensional graph or plane,
/// which can be used to index a scalar from a `Vec2` type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis2 {
    NoAxis,
    X,
    Y
}
use Axis2::{ X, Y, NoAxis };

impl Axis2 {

    /// Gets a normalized `Vec2` representing this axis.
    pub fn to_vec2<T: Scalar>(&self) -> Vec2<T> {
        match self {
            X      => Vec2(T::one(),  T::zero()),
            Y      => Vec2(T::zero(), T::one() ),
            NoAxis => Vec2(T::zero(), T::zero())
        }
    }
}

impl <T: Scalar> Index<Axis2> for Vec2<T> {
    type Output = T;
    fn index(&self, index: Axis2) -> &Self::Output {
        match index {
            X      => &self.0,
            Y      => &self.1,
            NoAxis => panic!("`NoAxis` is not a valid index!")
        }
    }
}

impl <T: Scalar> IndexMut<Axis2> for Vec2<T> {
    fn index_mut(&mut self, index: Axis2) -> &mut Self::Output {
        match index {
            X      => &mut self.0,
            Y      => &mut self.1,
            NoAxis => panic!("`NoAxis` is not a valid index!")
        }
    }
}


/*
    2D Vector
        Implementation
*/


/// A 2D Vector with an X and Y component.
/// Contains behaviours and methods for allowing for algebraic, geometric, and trigonometric operations,
/// as well as interpolation and other common vector operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
#[repr(C)]
pub struct Vec2<T: Scalar>(pub T, pub T);

impl <T: Scalar> VectorAbstract<T, Vec2<T>> for Vec2<T> {}

impl <T: Scalar> Vector<T, Vec2<T>, Axis2> for Vec2<T>  {
    const ZERO: Vec2<T> = Vec2(T::ZERO, T::ZERO);
    const ONE:  Vec2<T> = Vec2(T::ONE,  T::ONE );

    fn identity(&self) -> Vec2<T> {
        *self
    }

    fn rank() -> usize {
        2
    }

    #[cfg(feature = "alloc")]
    fn to_vec(&self) -> Vec<T> {
        vec![self[X], self[Y]]
    }

    fn sum(&self) -> T {
        self[X] + self[Y]
    }

    fn product(&self) -> T {
        self[X] * self[Y]
    }
    
    fn dot(&self, other: Vec2<T>) -> T {
        self[X] * other[X] + self[Y] * other[Y]
    }

    fn argmax(&self) -> Axis2 {
        if self[X] == self[Y] {
            NoAxis
        } else if self[X] > self[Y] {
            X
        } else {
            Y
        }
    }

    fn argmin(&self) -> Axis2 {
        if self[X] == self[Y] {
            NoAxis
        } else if self[X] < self[Y] {
            X
        } else {
            Y
        }
    }

    fn min<I: Vectorized<T, Vec2<T>>>(&self, other: I) -> Vec2<T> {
        let other: Vec2<T> = other.dvec();
        Vec2(self[X].min(other[X]), self[Y].min(other[Y]))
    }

    fn max<I: Vectorized<T, Vec2<T>>>(&self, other: I) -> Vec2<T> {
        let other: Vec2<T> = other.dvec();
        Vec2(self[X].max(other[X]), self[Y].max(other[Y]))
    }

    fn clamp<I: Vectorized<T, Vec2<T>>>(&self, min: I, max: I) -> Vec2<T> {
        let min: Vec2<T> = min.dvec();
        let max: Vec2<T> = max.dvec();
        Vec2(self[X].clamp(min[X], max[X]), self[Y].clamp(min[Y], max[Y]))
    }
}

impl <T: SignedScalar> SignedVector<T, Vec2<T>, Axis2, T> for Vec2<T> {
    fn signum(&self) -> Vec2<T> {
        Vec2(self[X].signum(), self[Y].signum())
    }

    fn abs(&self) -> Vec2<T> {
        Vec2(self[X].abs(), self[Y].abs())
    }

    fn cross(&self, other: Vec2<T>) -> T {
        self[X] * other[Y] - self[Y] * other[X]
    }
}

impl <T: IntScalar<T>> IntVector<T, Vec2<T>, Axis2> for Vec2<T> {
    fn pow<I: Vectorized<T, Vec2<T>>>(&self, pow: I) -> Vec2<T> {
        let pow: Vec2<T> = pow.dvec();
        Vec2(self[X].pow(pow[X].to_u32().unwrap()), self[Y].pow(pow[Y].to_u32().unwrap()))
    }

    fn log<I: Vectorized<T, Vec2<T>>>(&self, base: I) -> Vec2<T> {
        let base: Vec2<T> = base.dvec();
        Vec2(self[X].ilog(base[X]), self[Y].ilog(base[Y]))
    }
}

impl <T: FloatScalar> FloatVector<T, Vec2<T>, Axis2, T> for Vec2<T> {    
    fn to_radians(&self) -> Vec2<T> {
        Vec2(self[X].to_radians(), self[Y].to_radians())
    }

    fn to_degrees(&self) -> Vec2<T> {
        Vec2(self[X].to_degrees(), self[Y].to_degrees())
    }

    fn sin(&self) -> Vec2<T> {
        Vec2(self[X].sin(), self[Y].sin())
    }

    fn cos(&self) -> Vec2<T> {
        Vec2(self[X].cos(), self[Y].cos())
    }

    fn tan(&self) -> Vec2<T> {
        Vec2(self[X].tan(), self[Y].tan())
    }

    fn asin(&self) -> Vec2<T> {
        Vec2(self[X].asin(), self[Y].asin())
    }

    fn acos(&self) -> Vec2<T> {
        Vec2(self[X].acos(), self[Y].acos())
    }

    fn atan(&self) -> Vec2<T> {
        Vec2(self[X].atan(), self[Y].atan())
    }

    fn distance_squared_to(&self, other: Vec2<T>) -> T {
        (self[X] - other[X]).powi(2) + (self[Y] - other[Y]).powi(2)
    }
    
    fn smoothstep<I: Vectorized<T, Vec2<T>>>(&self, a: I, b: I) -> Vec2<T> {
        let a: Vec2<T> = a.dvec();
        let b: Vec2<T> = b.dvec();

        Vec2(
            self[X].smoothstep(a[X], b[X]),
            self[Y].smoothstep(a[Y], b[Y])
        )
    }

    fn bezier_derivative(&self, control_1: Vec2<T>, control_2: Vec2<T>, terminal: Vec2<T>, t: T) -> Vec2<T> {
        Vec2(
            self[X].bezier_derivative(control_1[X], control_2[X], terminal[X], t),
            self[Y].bezier_derivative(control_1[Y], control_2[Y], terminal[Y], t)
        )
    }

    fn bezier_sample(&self, control_1: Vec2<T>, control_2: Vec2<T>, terminal: Vec2<T>, t: T) -> Vec2<T> {
        Vec2(
            self[X].bezier_sample(control_1[X], control_2[X], terminal[X], t),
            self[Y].bezier_sample(control_1[Y], control_2[Y], terminal[Y], t)
        )
    }
    
    fn cubic_interpolate(&self, b: Vec2<T>, pre_a: Vec2<T>, post_b: Vec2<T>, weight: T) -> Vec2<T> {
        Vec2(
            self[X].cubic_interpolate(b[X], pre_a[X], post_b[X], weight),
            self[Y].cubic_interpolate(b[Y], pre_a[Y], post_b[Y], weight)
        )
    }
    
    fn cubic_interpolate_in_time(&self, b: Vec2<T>, pre_a: Vec2<T>, post_b: Vec2<T>, weight: T, b_t: T, pre_a_t: T, post_b_t: T) -> Vec2<T> {
        Vec2(
            self[X].cubic_interpolate_in_time(b[X], pre_a[X], post_b[X], weight, b_t, pre_a_t, post_b_t),
            self[Y].cubic_interpolate_in_time(b[Y], pre_a[Y], post_b[Y], weight, b_t, pre_a_t, post_b_t)
        )
    }

    fn sqrt(&self) -> Vec2<T> {
        Vec2(self[X].sqrt(), self[Y].sqrt())
    }

    fn pow2(&self) -> Vec2<T> {
        Vec2(self[X] * self[X], self[Y] * self[Y])
    }

    fn pow<I: Vectorized<T, Vec2<T>>>(&self, pow: I) -> Vec2<T> {
        let pow: Vec2<T> = pow.dvec();
        Vec2(self[X].powf(pow[X]), self[Y].powf(pow[Y]))
    }

    fn log<I: Vectorized<T, Vec2<T>>>(&self, base: I) -> Vec2<T> {
        let base: Vec2<T> = base.dvec();
        Vec2(self[X].log(base[X]), self[Y].log(base[Y]))
    }

    fn floor(&self) -> Vec2<T> {
        Vec2(self[X].floor(), self[Y].floor())
    }

    fn ceil(&self) -> Vec2<T> {
        Vec2(self[Y].ceil(), self[Y].ceil())
    }

    fn round(&self) -> Vec2<T> {
        Vec2(self[X].round(), self[Y].round())
    }

    fn approx_eq(&self, other: Vec2<T>) -> bool {
        self[X].approx_eq(other[X]) && self[Y].approx_eq(other[Y])
    }
}

impl <T: Scalar> Vec2<T> {
    
    /// A `Vec2` that describes an upwards direction:
    /// `Vec2(0, 1)`.
    pub const UP: Vec2<T> = Vec2(T::ZERO, T::ONE);
    
    /// A `Vec2` that describes a rightwards direction:
    /// `Vec2(1, 0)`.
    pub const RIGHT: Vec2<T> = Vec2(T::ONE, T::ZERO);

    /// Initializes a `Vec2` that describes a plotted point along the x axis.
    pub fn on_x(x: T) -> Vec2<T> {
        Vec2(x, T::zero())
    }

    /// Initializes a `Vec2` that describes a plotted point along the y axis.
    pub fn on_y(y: T) -> Vec2<T> {
        Vec2(T::zero(), y)
    }

    /// Converts a `Vec2` to a `Vec2` of a different type.
    /// Returns `None` if the cast was unsuccessful.
    pub fn cast<U: Scalar>(&self) -> Option<Vec2<U>> {
        match (U::from(self[X]), U::from(self[Y])) {
            (Some(x), Some(y)) => Some(Vec2(x, y)),
            _                  => None
        }
    }

    /// Returns a `Vec2` that represents this `Vec2`'s X component.
    pub fn of_x(&self) -> Vec2<T> {
        Vec2(self[X], T::zero())
    }

    /// Returns a `Vec2` that represents this `Vec2`'s Y component.
    pub fn of_y(&self) -> Vec2<T> {
        Vec2(T::zero(), self[Y])
    }

    /// Gets the x component.
    pub fn x(&self) -> T {
        self.0
    }
    
    /// Gets the y component.
    pub fn y(&self) -> T {
        self.1
    }

    /// Gets the x and y components of this `Vec2` as another identity function.
    pub fn xy(&self) -> Vec2<T> {
        self.identity()
    }

    /// Gets the x and y components of this `Vec2` in inverse order as a `Vec2`.
    pub fn yx(&self) -> Vec2<T> {
        Vec2(self[Y], self[X])
    }

    /// Calculates the aspect ratio of this `Vec2`.
    pub fn aspect_ratio(&self) -> T {
        self[X] / self[Y]
    }    
}

impl <T: SignedScalar> Vec2<T> {

    /// A `Vec2` that describes a downwards direction:
    /// `Vec2(0, -1)`.
    pub const DOWN: Vec2<T> = Vec2(T::ZERO, T::NEG_ONE);

    /// A `Vec2` that describes a leftwards direction.
    /// `Vec2(-1, 0)`.
    pub const LEFT: Vec2<T> = Vec2(T::NEG_ONE, T::ZERO);
}

impl <T: FloatScalar> Vec2<T> {

    /// Initializes a `Vec2` from an angle in radians.
    pub fn from_angle(angle: T) -> Vec2<T> {
        Vec2(angle.cos(), angle.sin())
    }

    /// Calculates the angle of a `Vec2` in respect to the positive x-axis.
    /// # Returns
    /// The angle of the `Vec2` in radians.
    pub fn angle(&self) -> T {
        self[Y].atan2(self[X])
    }

    /// Calculates the angle between the line connecting the two positions `self` and `other` and the x-axis.
    /// # Returns
    /// The angle in radians.
    pub fn angle_between(&self, other: &Vec2<T>) -> T {
        (other - self).angle()
    }

    /// Rotates this vector by a given angle in radians.
    pub fn rotated(&self, angle: T) -> Vec2<T> {
        
        // Compute the sine and cosine of the angle.
        let (sine, cosine): (T, T) = angle.sin_cos();

        // Rotate the vector using the rotation matrix formula.
        Vec2(
            self[X] * cosine - self[Y] * sine,
            self[X] * sine   + self[Y] * cosine
        )
    }

    /// Rotates this vector by 90 degrees counter clockwise.
    pub fn orthogonal(&self) -> Vec2<T> {
        self.rotated(T::from(FRAC_PI_2).unwrap())
    }

    /// Spherically interpolates between two vectors.
    /// This interpolation is focused on the length or magnitude of the vectors. If the magnitudes are equal,
    /// the interpolation is linear and behaves the same way as `lerp()`.
    pub fn slerp(&self, other: Vec2<T>, t: T) -> Vec2<T> {
        let theta: T = self.angle_to(other);
        self.rotated(theta * t)
    }
}


/*
    Glam
        Support
*/


impl Vec2<f32> {

    /// Converts this vector into a glam vector.
    #[cfg(feature = "glam")]
    pub fn to_glam(&self) -> GVec2 {
        vec2(self.x(), self.y())
    }
}

impl Vec2<f64> {

    /// Converts this vector into a glam vector.
    #[cfg(feature = "glam")]
    pub fn to_glam(&self) -> DVec2 {
        dvec2(self.x(), self.y())
    }
}

impl Vec2<u32> {

    /// Converts this vector into a glam vector.
    #[cfg(feature = "glam")]
    pub fn to_glam(&self) -> UVec2 {
        uvec2(self.x(), self.y())
    }
}

impl Vec2<i32> {

    /// Converts this vector into a glam vector.
    #[cfg(feature = "glam")]
    pub fn to_glam(&self) -> IVec2 {
        ivec2(self.x(), self.y())
    }
}


/*
    Global Operations
        Base Arithmetic
*/


impl <T: Scalar> Add for Vec2<T> {
    type Output = Vec2<T>;
    fn add(self, other: Self) -> Self::Output {
        Vec2(self.0 + other.0, self.1 + other.1)
    }
}

impl <T: Scalar> Sub for Vec2<T> {
    type Output = Vec2<T>;
    fn sub(self, other: Self) -> Self::Output {
        Vec2(self.0 - other.0, self.1 - other.1)
    }
}

impl <T: Scalar> Mul for Vec2<T> {
    type Output = Vec2<T>;
    fn mul(self, other: Self) -> Self::Output {
        Vec2(self.0 * other.0, self.1 * other.1)
    }
}

impl <T: Scalar> Div for Vec2<T> {
    type Output = Vec2<T>;
    fn div(self, other: Self) -> Self::Output {
        Vec2(self.0 / other.0, self.1 / other.1)
    }
}

impl <T: Scalar> Rem for Vec2<T> {
    type Output = Vec2<T>;
    fn rem(self, other: Self) -> Self::Output {
        Vec2(self.0 % other.0, self.1 % other.1)
    }
}

impl <T: SignedScalar> Neg for Vec2<T> {
    type Output = Vec2<T>;
    fn neg(self) -> Self::Output {
        Vec2(-self.0, -self.1)
    }
}


/*
    Global Operations
        Reference Arithmetic
*/


impl <'a, T: Scalar> Add<&'a Vec2<T>> for &'a Vec2<T> {
    type Output = Vec2<T>;
    fn add(self, other: Self) -> Self::Output {
        Vec2(self.0 + other.0, self.1 + other.1)
    }
}

impl <'a, T: Scalar> Sub<&'a Vec2<T>> for &'a Vec2<T> {
    type Output = Vec2<T>;
    fn sub(self, other: Self) -> Self::Output {
        Vec2(self.0 - other.0, self.1 - other.1)
    }
}

impl <'a, T: Scalar> Mul<&'a Vec2<T>> for &'a Vec2<T> {
    type Output = Vec2<T>;
    fn mul(self, other: Self) -> Self::Output {
        Vec2(self.0 * other.0, self.1 * other.1)
    }
}

impl <'a, T: Scalar> Div<&'a Vec2<T>> for &'a Vec2<T> {
    type Output = Vec2<T>;
    fn div(self, other: Self) -> Self::Output {
        Vec2(self.0 / other.0, self.1 / other.1)
    }
}

impl <'a, T: Scalar> Rem<&'a Vec2<T>> for &'a Vec2<T> {
    type Output = Vec2<T>;
    fn rem(self, other: Self) -> Self::Output {
        Vec2(self.0 % other.0, self.1 % other.1)
    }
}

impl <T: SignedScalar> Neg for &Vec2<T> {
    type Output = Vec2<T>;
    fn neg(self) -> Self::Output {
        Vec2(-self.0, -self.1)
    }
}


/*
    Global Operations
        Reference vs Base Arithmetic
*/


impl <T: Scalar> Add<&Vec2<T>> for Vec2<T> {
    type Output = Vec2<T>;
    fn add(self, other: &Self) -> Self::Output {
        Vec2(self.0 + other.0, self.1 + other.1)
    }
}

impl <T: Scalar> Sub<&Vec2<T>> for Vec2<T> {
    type Output = Vec2<T>;
    fn sub(self, other: &Self) -> Self::Output {
        Vec2(self.0 - other.0, self.1 - other.1)
    }
}

impl <T: Scalar> Mul<&Vec2<T>> for Vec2<T> {
    type Output = Vec2<T>;
    fn mul(self, other: &Self) -> Self::Output {
        Vec2(self.0 * other.0, self.1 * other.1)
    }
}

impl <T: Scalar> Div<&Vec2<T>> for Vec2<T> {
    type Output = Vec2<T>;
    fn div(self, other: &Self) -> Self::Output {
        Vec2(self.0 / other.0, self.1 / other.1)
    }
}

impl <T: Scalar> Rem<&Vec2<T>> for Vec2<T> {
    type Output = Vec2<T>;
    fn rem(self, other: &Self) -> Self::Output {
        Vec2(self.0 % other.0, self.1 % other.1)
    }
}


/*
    Global Operations
        Base vs Scalar Arithmetic
*/


impl <T: Scalar> Add<T> for Vec2<T> {
    type Output = Vec2<T>;
    fn add(self, other: T) -> Self::Output {
        self + Vec2(other, other)
    }
}

impl <T: Scalar> Sub<T> for Vec2<T> {
    type Output = Vec2<T>;
    fn sub(self, other: T) -> Self::Output {
        self - Vec2(other, other)
    }
}

impl <T: Scalar> Mul<T> for Vec2<T> {
    type Output = Vec2<T>;
    fn mul(self, other: T) -> Self::Output {
        self * Vec2(other, other)
    }
}

impl <T: Scalar> Div<T> for Vec2<T> {
    type Output = Vec2<T>;
    fn div(self, other: T) -> Self::Output {
        self / Vec2(other, other)
    }
}

impl <T: Scalar> Rem<T> for Vec2<T> {
    type Output = Vec2<T>;
    fn rem(self, other: T) -> Self::Output {
        self % Vec2(other, other)
    }
}


/*
    Global Operations
        Reference vs Scalar Arithmetic
*/


impl <T: Scalar> Add<T> for &Vec2<T> {
    type Output = Vec2<T>;
    fn add(self, other: T) -> Self::Output {
        self + &Vec2(other, other)
    }
}

impl <T: Scalar> Sub<T> for &Vec2<T> {
    type Output = Vec2<T>;
    fn sub(self, other: T) -> Self::Output {
        self - &Vec2(other, other)
    }
}

impl <T: Scalar> Mul<T> for &Vec2<T> {
    type Output = Vec2<T>;
    fn mul(self, other: T) -> Self::Output {
        self * &Vec2(other, other)
    }
}

impl <T: Scalar> Div<T> for &Vec2<T> {
    type Output = Vec2<T>;
    fn div(self, other: T) -> Self::Output {
        self / &Vec2(other, other)
    }
}

impl <T: Scalar> Rem<T> for &Vec2<T> {
    type Output = Vec2<T>;
    fn rem(self, other: T) -> Self::Output {
        self % &Vec2(other, other)
    }
}


/*
    Global Operations
        Assignment & Arithmetic
*/


impl <T: Scalar> AddAssign for Vec2<T> {
    fn add_assign(&mut self, other: Self) -> () {
        *self = *self + other;
    }
}

impl <T: Scalar> SubAssign for Vec2<T> {
    fn sub_assign(&mut self, other: Self) -> () {
        *self = *self - other;
    }
}

impl <T: Scalar> MulAssign for Vec2<T> {
    fn mul_assign(&mut self, other: Self) -> () {
        *self = *self * other;
    }
}

impl <T: Scalar> DivAssign for Vec2<T> {
    fn div_assign(&mut self, other: Self) -> () {
        *self = *self / other;
    }
}

impl <T: Scalar> RemAssign for Vec2<T> {
    fn rem_assign(&mut self, other: Self) -> () {
        *self = *self % other;
    }
}


/*
    Global
        Behaviours
*/


impl <T: Scalar> Default for Vec2<T> {
    fn default() -> Self {
        Vec2(T::default(), T::default())
    }
}

impl <T: Scalar> Display for Vec2<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vec2({}, {})", self.0, self.1)
    }
}


/*
    Vectorized
        Trait
*/


impl <T: Scalar> Vectorized<T, Vec2<T>> for T {
    fn attempt_get_scalar(self) -> Option<T> {
        Some(self)
    }

    fn dvec(self) -> Vec2<T> {
        Vec2(self, self)
    }
}

impl <T: Scalar> Vectorized<T, Vec2<T>> for (T, T) {
    fn attempt_get_scalar(self) -> Option<T> {
        None
    }

    fn dvec(self) -> Vec2<T> {
        Vec2(self.0, self.1)
    }
}

impl <T: Scalar> Vectorized<T, Vec2<T>> for Vec2<T> {
    fn attempt_get_scalar(self) -> Option<T> {
        None
    }

    fn dvec(self) -> Vec2<T> {
        self
    }
}

pub trait Vectorized2D<T: Scalar> {
    
    /// Converts a type that can be represented as a Vector of 2 as a `Vec2`.
    /// This is the public interface for the `Vectorized` trait.
    fn vec2(self) -> Vec2<T>;
}

impl <T: Scalar + Vectorized<T, Vec2<T>>> Vectorized2D<T> for T {
    fn vec2(self) -> Vec2<T> {
        self.dvec()
    }
}

impl <T: Scalar + Vectorized<T, Vec2<T>>> Vectorized2D<T> for (T, T) {
    fn vec2(self) -> Vec2<T> {
        self.dvec()
    }
}
