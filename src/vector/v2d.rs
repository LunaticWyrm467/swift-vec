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

use super::*;
use crate::scalar::{ Scalar, SignedScalar };


/*
    2D Vector
        Implementation
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis2 {
    None,
    X,
    Y
}

/// A 2D Vector with an X and Y component.
/// Contains behaviours and methods for allowing for algebraic, geometric, and trigonometric operations,
/// as well as interpolation and other common vector operations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
pub struct Vec2<T: Scalar>(pub T, pub T);

impl <T: Scalar> VectorAbstract<T, Vec2<T>> for Vec2<T> {}

impl <T: Scalar> Vector<T, Vec2<T>, Axis2> for Vec2<T>  {
    fn rank() -> usize {
        2
    }

    fn get(&self, axis: Axis2) -> T {
        match axis {
            Axis2::X    => self.x(),
            Axis2::Y    => self.y(),
            Axis2::None => panic!("Axis2::None is not a valid axis!")
        }
    }

    fn to_vec(&self) -> Vec<T> {
        vec![self.x(), self.y()]
    }

    fn identity(&self) -> &Vec2<T> {
        self
    }

    fn ones_like() -> Vec2<T> {
        Vec2(T::one(), T::one())
    }

    fn zeros_like() -> Vec2<T> {
        Vec2(T::zero(), T::zero())
    }

    fn sum(&self) -> T {
        self.x() + self.y()
    }

    fn product(&self) -> T {
        self.x() * self.y()
    }

    fn argmax(&self) -> Axis2 {
        if self.x() == self.y() {
            Axis2::None
        } else if self.x() > self.y() {
            Axis2::X
        } else {
            Axis2::Y
        }
    }

    fn argmin(&self) -> Axis2 {
        if self.x() == self.y() {
            Axis2::None
        } else if self.x() < self.y() {
            Axis2::X
        } else {
            Axis2::Y
        }
    }

    fn v_min(&self, other: &Vec2<T>) -> Vec2<T> {
        Vec2(self.x().min(other.x()), self.y().min(other.y()))
    }

    fn v_max(&self, other: &Vec2<T>) -> Vec2<T> {
        Vec2(self.x().max(other.x()), self.y().max(other.y()))
    }
}

impl <T: SignedScalar> SignedVector<T, Vec2<T>, Axis2> for Vec2<T> {
    fn signum(&self) -> Vec2<T> {
        Vec2(self.x().signum(), self.y().signum())
    }

    fn abs(&self) -> Vec2<T> {
        Vec2(self.x().abs(), self.y().abs())
    }
}

impl <T: IntScalar<T>> IntVector<T, Vec2<T>, Axis2> for Vec2<T> {
    fn v_pow(&self, pow: &Vec2<T>) -> Vec2<T> {
        Vec2(self.x().pow(pow.x().to_u32().unwrap()), self.y().pow(pow.y().to_u32().unwrap()))
    }

    fn v_log(&self, base: &Vec2<T>) -> Vec2<T> {
        Vec2(self.x().ilog(base.x()), self.y().ilog(base.y()))
    }
}

impl <T: FloatScalar> FloatVector<T, Vec2<T>, Axis2> for Vec2<T> {
    

    fn rotated(&self, angle: T) -> Vec2<T> {
        
        // Compute the sine and cosine of the angle.
        let (sine, cosine): (T, T) = angle.sin_cos();

        // Rotate the vector using the rotation matrix formula.
        Vec2(
            self.x() * cosine - self.y() * sine,
            self.x() * sine   + self.y() * cosine
        )
    }

    fn to_radians(&self) -> Vec2<T> {
        Vec2(self.x().to_radians(), self.y().to_radians())
    }

    fn to_degrees(&self) -> Vec2<T> {
        Vec2(self.x().to_degrees(), self.y().to_degrees())
    }

    fn sin(&self) -> Vec2<T> {
        Vec2(self.x().sin(), self.y().sin())
    }

    fn cos(&self) -> Vec2<T> {
        Vec2(self.x().cos(), self.y().cos())
    }

    fn tan(&self) -> Vec2<T> {
        Vec2(self.x().tan(), self.y().tan())
    }

    fn asin(&self) -> Vec2<T> {
        Vec2(self.x().asin(), self.y().asin())
    }

    fn acos(&self) -> Vec2<T> {
        Vec2(self.x().acos(), self.y().acos())
    }

    fn atan(&self) -> Vec2<T> {
        Vec2(self.x().atan(), self.y().atan())
    }

    fn distance_squared_to(&self, other: &Vec2<T>) -> T {
        (self.x() - other.x()).powi(2) + (self.y() - other.y()).powi(2)
    }

    fn bezier_derivative(&self, control_1: &Vec2<T>, control_2: &Vec2<T>, terminal: &Vec2<T>, t: T) -> Vec2<T> {
        Vec2(
            self.x().bezier_derivative(control_1.x(), control_2.x(), terminal.x(), t),
            self.y().bezier_derivative(control_1.y(), control_2.y(), terminal.y(), t)
        )
    }

    fn bezier_sample(&self, control_1: &Vec2<T>, control_2: &Vec2<T>, terminal: &Vec2<T>, t: T) -> Vec2<T> {
        Vec2(
            self.x().bezier_sample(control_1.x(), control_2.x(), terminal.x(), t),
            self.y().bezier_sample(control_1.y(), control_2.y(), terminal.y(), t)
        )
    }

    fn cubic_interpolate(&self, terminal: &Vec2<T>, pre_start: &Vec2<T>, post_terminal: &Vec2<T>, t: T) -> Vec2<T> {
        Vec2(
            self.x().cubic_interpolate(terminal.x(), pre_start.x(), post_terminal.x(), t),
            self.y().cubic_interpolate(terminal.y(), pre_start.y(), post_terminal.y(), t)
        )
    }

    fn cubic_interpolate_in_time(&self, terminal: &Vec2<T>, pre_start: &Vec2<T>, post_terminal: &Vec2<T>, t0: T, terminal_t: T, pre_start_t: T, post_terminal_t: T) -> Vec2<T> {
        Vec2(
            self.x().cubic_interpolate_in_time(terminal.x(), pre_start.x(), post_terminal.x(), t0, terminal_t, pre_start_t, post_terminal_t),
            self.y().cubic_interpolate_in_time(terminal.y(), pre_start.y(), post_terminal.y(), t0, terminal_t, pre_start_t, post_terminal_t)
        )
    }

    fn dot(&self, other: &Vec2<T>) -> T {
        self.x() * other.x() + self.y() * other.y()
    }

    fn cross(&self, other: &Vec2<T>) -> T {
        self.x() * other.y() - self.y() * other.x()
    }

    fn sqrt(&self) -> Vec2<T> {
        Vec2(self.x().sqrt(), self.y().sqrt())
    }

    fn sqr(&self) -> Vec2<T> {
        Vec2(self.x() * self.x(), self.y() * self.y())
    }

    fn v_pow(&self, pow: &Vec2<T>) -> Vec2<T> {
        Vec2(self.x().powf(pow.x()), self.y().powf(pow.y()))
    }

    fn v_log(&self, base: &Vec2<T>) -> Vec2<T> {
        Vec2(self.x().log(base.x()), self.y().log(base.y()))
    }

    fn floor(&self) -> Vec2<T> {
        Vec2(self.x().floor(), self.y().floor())
    }

    fn ceil(&self) -> Vec2<T> {
        Vec2(self.y().ceil(), self.y().ceil())
    }

    fn round(&self) -> Vec2<T> {
        Vec2(self.x().round(), self.y().round())
    }

    fn approx_eq(&self, other: &Vec2<T>) -> bool {
        let eps: T = T::epsilon() * T::from(4).unwrap();
        relative_eq!(self.x(), other.x(), epsilon = eps) && approx::relative_eq!(self.y(), other.y(), epsilon = eps)
    }
}

impl <T: Scalar> Vec2<T> {
    
    /// Initializes a vector that describes an upwards direction.
    pub fn up() -> Vec2<T> {
        Vec2(T::zero(), T::one())
    }
    
    /// Initializes a vector that describes a rightwards direction.
    pub fn right() -> Vec2<T> {
        Vec2(T::one(), T::zero())
    }

    /// Initializes a vector from a scalar.
    pub fn of(scalar: T) -> Vec2<T> {
        Vec2(scalar, scalar)
    }

    /// Converts a vector to a vector of a different type.
    pub fn cast<U: Scalar>(&self) -> Vec2<U> {
        Vec2(U::from(self.x()).unwrap(), U::from(self.y()).unwrap())
    }
    
    /// Gets the x component of the vector.
    pub fn x(&self) -> T {
        self.0
    }

    /// Gets the y component of the vector.
    pub fn y(&self) -> T {
        self.1
    }

    /// Gets the x and y components of this vector in order as a tuple.
    pub fn raw(&self) -> (T, T) {
        (self.x(), self.y())
    }

    /// An alias for the identity function.
    pub fn xy(&self) -> Vec2<T> {
        self.identity().to_owned()
    }

    /// Gets the x and y components of this vector in inverse order as a vector of 2.
    pub fn yx(&self) -> Vec2<T> {
        Vec2(self.y(), self.x())
    }

    /// Calculates the aspect ratio of this vector.
    pub fn aspect_ratio(&self) -> T {
        self.x() / self.y()
    }    
}

impl <T: SignedScalar> Vec2<T> {

    /// Initializes a vector that describes a downwards direction.
    pub fn down() -> Vec2<T> {
        Vec2(T::zero(), -T::one())
    }

    /// Initializes a vector that describes a leftwards direction.
    pub fn left() -> Vec2<T> {
        Vec2(-T::one(), T::zero())
    }
}

impl <T: FloatScalar> Vec2<T> {

    /// Initializes a vector from an angle in radians.
    pub fn from_angle(angle: T) -> Vec2<T> {
        Vec2(angle.cos(), angle.sin())
    }

    /// Calculates the angle of a vector in respect to the positive x-axis.
    /// # Returns
    /// The angle of the vector in radians.
    pub fn angle(&self) -> T {
        self.y().atan2(self.x())
    }

    /// Calculates the angle between the line connecting the two positions `self` and `other` and the x-axis.
    /// # Returns
    /// The angle in radians.
    pub fn angle_between(&self, other: &Vec2<T>) -> T {
        (other - self).angle()
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
        *self = self.to_owned() + other;
    }
}

impl <T: Scalar> SubAssign for Vec2<T> {
    fn sub_assign(&mut self, other: Self) -> () {
        *self = self.to_owned() - other;
    }
}

impl <T: Scalar> MulAssign for Vec2<T> {
    fn mul_assign(&mut self, other: Self) -> () {
        *self = self.to_owned() * other;
    }
}

impl <T: Scalar> DivAssign for Vec2<T> {
    fn div_assign(&mut self, other: Self) -> () {
        *self = self.to_owned() / other;
    }
}

impl <T: Scalar> RemAssign for Vec2<T> {
    fn rem_assign(&mut self, other: Self) -> () {
        *self = self.to_owned() % other;
    }
}

impl <T: Scalar> Default for Vec2<T> {
    fn default() -> Self {
        Vec2(T::default(), T::default())
    }
}

impl <T: Scalar> std::fmt::Display for Vec2<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vec2({}, {})", self.0, self.1)
    }
}