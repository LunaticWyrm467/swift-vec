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

//!
//! Created by LunaticWyrm467
//!

//?
//? Contains the implementations for all of the Vector2D types.
//?

use super::*;
use crate::scalar::{ Scalar, SignedScalar };


/*
    2D Vector
        Implementation
*/

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Axis2 {
    None,
    X,
    Y
}


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
        Vec2(
            if self.x() < other.x() { self.x() } else { other.x() },
            if self.y() < other.y() { self.y() } else { other.y() }
        )
    }

    fn v_max(&self, other: &Vec2<T>) -> Vec2<T> {
        Vec2(
            if self.x() > other.x() { self.x() } else { other.x() },
            if self.y() > other.y() { self.y() } else { other.y() }
        )
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
    fn from_angle(angle: T) -> Vec2<T> {
        Vec2(angle.cos(), angle.sin())
    }

    fn angle(&self) -> T {
        self.x().atan2(self.y())
    }

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

    fn approx_eq(&self, other: &Vec2<T>) -> bool {   // TODO: Figure out epsilon trait requirements.
        relative_eq!(self.x(), other.x()) && approx::relative_eq!(self.y(), other.y())
    }

    fn is_zero_approx(&self) -> bool {   // TODO: Figure out epsilon trait requirements.
        relative_eq!(self.x(), T::zero()) && approx::relative_eq!(self.y(), T::zero())
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
    
    /// Gets the x component of the vector.
    pub fn x(&self) -> T {
        self.0
    }

    /// Gets the y component of the vector.
    pub fn y(&self) -> T {
        self.1
    }

    /// Gets the x and y components of this vector in order as a tuple.
    pub fn xy(&self) -> (T, T) {
        (self.x(), self.y())
    }

    /// Gets the x and y components of this vector in inverse order as a tuple.
    pub fn yx(&self) -> (T, T) {
        (self.y(), self.x())
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