//===================================================================================================================================================================================//
//
//  /$$    /$$                      /$$                                /$$$$$$  /$$$$$$$ 
// | $$   | $$                     | $$                               /$$__  $$| $$__  $$
// | $$   | $$ /$$$$$$   /$$$$$$$ /$$$$$$    /$$$$$$   /$$$$$$       |__/  \ $$| $$  \ $$
// |  $$ / $$//$$__  $$ /$$_____/|_  $$_/   /$$__  $$ /$$__  $$         /$$$$$/| $$  | $$
//  \  $$ $$/| $$$$$$$$| $$        | $$    | $$  \ $$| $$  \__/        |___  $$| $$  | $$
//   \  $$$/ | $$_____/| $$        | $$ /$$| $$  | $$| $$             /$$  \ $$| $$  | $$
//    \  $/  |  $$$$$$$|  $$$$$$$  |  $$$$/|  $$$$$$/| $$            |  $$$$$$/| $$$$$$$/
//     \_/    \_______/ \_______/   \___/   \______/ |__/             \______/ |_______/ 
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
//! for any of the non-shared behaviours of the 3D vector.
//!

use super::Axis2::{ X as X2, Y as Y2 };
use super::*;


/*
 * 3D Vector
 *      Indexing
 */


/// Simply represents an axis of a 3-dimensional graph or plane.
/// which can be used to index a scalar from a `Vec3` type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis3 {
    NoAxis,
    X,
    Y,
    Z
}
use Axis3::{ X, Y, Z, NoAxis };

impl Axis3 {

    /// Gets a normalized `Vec3` representing this axis.
    pub fn to_vec3<T: Scalar>(&self) -> Vec3<T> {
        match self {
            X    => Vec3(T::one(),  T::zero(), T::zero()),
            Y    => Vec3(T::zero(), T::one(),  T::zero()),
            Z    => Vec3(T::zero(), T::zero(), T::one()),
            NoAxis => Vec3(T::zero(), T::zero(), T::zero())
        }
    }
}

impl <T: Scalar> Index<Axis3> for Vec3<T> {
    type Output = T;
    fn index(&self, index: Axis3) -> &Self::Output {
        match index {
            X      => &self.0,
            Y      => &self.1,
            Z      => &self.2,
            NoAxis => panic!("`NoAxis` is not a valid index!")
        }
    }
}

impl <T: Scalar> IndexMut<Axis3> for Vec3<T> {
    fn index_mut(&mut self, index: Axis3) -> &mut Self::Output {
        match index {
            X      => &mut self.0,
            Y      => &mut self.1,
            Z      => &mut self.2,
            NoAxis => panic!("`NoAxis` is not a valid index!")
        }
    }
}


/*
    3D Vector
        Implementation
*/


/// A 3D Vector with an X, Y and Z component.
/// Contains behaviours and methods for allowing for algebraic, geometric, and trigonometric operations,
/// as well as interpolation and other common vector operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub struct Vec3<T: Scalar>(pub T, pub T, pub T);

impl <T: Scalar> VectorAbstract<T, Vec3<T>> for Vec3<T> {}

impl <T: Scalar> Vector<T, Vec3<T>, Axis3> for Vec3<T>  {
    fn identity(&self) -> Vec3<T> {
        *self
    }

    fn rank() -> usize {
        2
    }

    #[cfg(feature = "alloc")]
    fn to_vec(&self) -> Vec<T> {
        vec![self[X], self[Y], self[Z]]
    }

    fn sum(&self) -> T {
        self[X] + self[Y] + self[Z]
    }

    fn product(&self) -> T {
        self[X] * self[Y] * self[Z]
    }

    fn argmax(&self) -> Axis3 {
        if self[X] == self[Y] && self[Y] == self[Z] {
            NoAxis
        } else if self[X] >= self[Y] && self[X] >= self[Z] {
            X
        } else if self[Y] >= self[X] && self[Y] >= self[Z] {
            Y
        } else {
            Z
        }
    }

    fn argmin(&self) -> Axis3 {
        if self[X] == self[Y] && self[Y] == self[Z] {
            NoAxis
        } else if self[X] <= self[Y] && self[X] <= self[Z] {
            X
        } else if self[Y] <= self[X] && self[Y] <= self[Z] {
            Y
        } else {
            Z
        }
    }

    fn min<I: Vectorized<T, Vec3<T>>>(&self, other: I) -> Vec3<T> {
        let other: Vec3<T> = other.dvec();
        Vec3(self[X].min(other[X]), self[Y].min(other[Y]), self[Z].min(other[Z]))
    }

    fn max<I: Vectorized<T, Vec3<T>>>(&self, other: I) -> Vec3<T> {
        let other: Vec3<T> = other.dvec();
        Vec3(self[X].max(other[X]), self[Y].max(other[Y]), self[Z].max(other[Z]))
    }

    fn clamp<I: Vectorized<T, Vec3<T>>>(&self, min: I, max: I) -> Vec3<T> {
        let min: Vec3<T> = min.dvec();
        let max: Vec3<T> = max.dvec();
        Vec3(self[X].clamp(min[X], max[X]), self[Y].clamp(min[Y], max[Y]), self[Z].clamp(min[Z], max[Z]))
    }
}

impl <T: SignedScalar> SignedVector<T, Vec3<T>, Axis3> for Vec3<T> {
    fn signum(&self) -> Vec3<T> {
        Vec3(self[X].signum(), self[Y].signum(), self[Z].signum())
    }

    fn abs(&self) -> Vec3<T> {
        Vec3(self[X].abs(), self[Y].abs(), self[Z].signum())
    }
}

impl <T: IntScalar<T>> IntVector<T, Vec3<T>, Axis3> for Vec3<T> {
    fn pow<I: Vectorized<T, Vec3<T>>>(&self, pow: I) -> Vec3<T> {
        let pow: Vec3<T> = pow.dvec();
        Vec3(self[X].pow(pow[X].to_u32().unwrap()), self[Y].pow(pow[Y].to_u32().unwrap()), self[Z].pow(pow[Z].to_u32().unwrap()))
    }

    fn log<I: Vectorized<T, Vec3<T>>>(&self, base: I) -> Vec3<T> {
        let base: Vec3<T> = base.dvec();
        Vec3(self[X].ilog(base[X]), self[Y].ilog(base[Y]), self[Z].ilog(base[Z]))
    }
}

impl <T: FloatScalar> FloatVector<T, Vec3<T>, Axis3, Vec3<T>> for Vec3<T> {
    fn to_radians(&self) -> Vec3<T> {
        Vec3(self[X].to_radians(), self[Y].to_radians(), self[Z].to_radians())
    }

    fn to_degrees(&self) -> Vec3<T> {
        Vec3(self[X].to_degrees(), self[Y].to_degrees(), self[Z].to_degrees())
    }

    fn sin(&self) -> Vec3<T> {
        Vec3(self[X].sin(), self[Y].sin(), self[Z].sin())
    }

    fn cos(&self) -> Vec3<T> {
        Vec3(self[X].cos(), self[Y].cos(), self[Z].cos())
    }

    fn tan(&self) -> Vec3<T> {
        Vec3(self[X].tan(), self[Y].tan(), self[Z].tan())
    }

    fn asin(&self) -> Vec3<T> {
        Vec3(self[X].asin(), self[Y].asin(), self[Z].asin())
    }

    fn acos(&self) -> Vec3<T> {
        Vec3(self[X].acos(), self[Y].acos(), self[Z].acos())
    }

    fn atan(&self) -> Vec3<T> {
        Vec3(self[X].atan(), self[Y].atan(), self[Z].atan())
    }

    fn distance_squared_to(&self, other: Vec3<T>) -> T {
        (self[X] - other[X]).powi(2) + (self[Y] - other[Y]).powi(2) + (self[Z] - other[Z]).powi(2)
    }
    
    fn smoothstep<I: Vectorized<T, Vec3<T>>>(&self, a: I, b: I) -> Vec3<T> {
        let a: Vec3<T> = a.dvec();
        let b: Vec3<T> = b.dvec();

        Vec3(
            self[X].smoothstep(a[X], b[X]),
            self[Y].smoothstep(a[Y], b[Y]),
            self[Z].smoothstep(a[Z], b[Z])
        )
    }

    fn bezier_derivative(&self, control_1: Vec3<T>, control_2: Vec3<T>, terminal: Vec3<T>, t: T) -> Vec3<T> {
        Vec3(
            self[X].bezier_derivative(control_1[X], control_2[X], terminal[X], t),
            self[Y].bezier_derivative(control_1[Y], control_2[Y], terminal[Y], t),
            self[Z].bezier_derivative(control_1[Z], control_2[Z], terminal[Z], t)
        )
    }

    fn bezier_sample(&self, control_1: Vec3<T>, control_2: Vec3<T>, terminal: Vec3<T>, t: T) -> Vec3<T> {
        Vec3(
            self[X].bezier_sample(control_1[X], control_2[X], terminal[X], t),
            self[Y].bezier_sample(control_1[Y], control_2[Y], terminal[Y], t),
            self[Z].bezier_sample(control_1[Z], control_2[Z], terminal[Z], t)
        )
    }
    
    fn cubic_interpolate(&self, b: Vec3<T>, pre_a: Vec3<T>, post_b: Vec3<T>, weight: T) -> Vec3<T> {
        Vec3(
            self[X].cubic_interpolate(b[X], pre_a[X], post_b[X], weight),
            self[Y].cubic_interpolate(b[Y], pre_a[Y], post_b[Y], weight),
            self[Z].cubic_interpolate(b[Z], pre_a[Z], post_b[Z], weight)
        )
    }
    
    fn cubic_interpolate_in_time(&self, b: Vec3<T>, pre_a: Vec3<T>, post_b: Vec3<T>, weight: T, b_t: T, pre_a_t: T, post_b_t: T) -> Vec3<T> {
        Vec3(
            self[X].cubic_interpolate_in_time(b[X], pre_a[X], post_b[X], weight, b_t, pre_a_t, post_b_t),
            self[Y].cubic_interpolate_in_time(b[Y], pre_a[Y], post_b[Y], weight, b_t, pre_a_t, post_b_t),
            self[Z].cubic_interpolate_in_time(b[Z], pre_a[Z], post_b[Z], weight, b_t, pre_a_t, post_b_t)
        )
    }

    fn dot(&self, other: Vec3<T>) -> T {
        self[X] * other[X] + self[Y] * other[Y] + self[Z] * other[Z]
    }

    fn cross(&self, other: Vec3<T>) -> Vec3<T> {
        Vec3(
            self[Y] * other[Z] - self[Z] * other[Y],
            self[Z] * other[X] - self[X] * other[Z],
            self[X] * other[Y] - self[Y] * other[X]
        )
    }

    fn sqrt(&self) -> Vec3<T> {
        Vec3(self[X].sqrt(), self[Y].sqrt(), self[Z].sqrt())
    }

    fn sqr(&self) -> Vec3<T> {
        Vec3(self[X] * self[X], self[Y] * self[Y], self[Z] * self[Z])
    }

    fn pow<I: Vectorized<T, Vec3<T>>>(&self, pow: I) -> Vec3<T> {
        let pow: Vec3<T> = pow.dvec();
        Vec3(self[X].powf(pow[X]), self[Y].powf(pow[Y]), self[Z].powf(pow[Z]))
    }

    fn log<I: Vectorized<T, Vec3<T>>>(&self, base: I) -> Vec3<T> {
        let base: Vec3<T> = base.dvec();
        Vec3(self[X].log(base[X]), self[Y].log(base[Y]), self[Z].log(base[Z]))
    }

    fn floor(&self) -> Vec3<T> {
        Vec3(self[X].floor(), self[Y].floor(), self[Z].floor())
    }

    fn ceil(&self) -> Vec3<T> {
        Vec3(self[Y].ceil(), self[Y].ceil(), self[Z].ceil())
    }

    fn round(&self) -> Vec3<T> {
        Vec3(self[X].round(), self[Y].round(), self[Z].round())
    }

    fn approx_eq(&self, other: Vec3<T>) -> bool {
        self[X].approx_eq(other[X]) && self[Y].approx_eq(other[Y]) && self[Z].approx_eq(other[Z])
    }
}

impl <T: Scalar> Vec3<T> {
    
    /// Initializes a `Vec3` that describes a forwards direction.
    pub fn forward() -> Vec3<T> {
        Vec3(T::zero(), T::zero(), T::one())
    }
    
    /// Initializes a `Vec3` that describes an upwards direction.
    pub fn up() -> Vec3<T> {
        Vec3(T::zero(), T::one(), T::zero())
    }
    
    /// Initializes a `Vec3` that describes a rightwards direction.
    pub fn right() -> Vec3<T> {
        Vec3(T::one(), T::zero(), T::zero())
    }

    /// Initializes a `Vec3` that describes a plotted point along the x axis.
    pub fn on_x(x: T) -> Vec3<T> {
        Vec3(x, T::zero(), T::zero())
    }

    /// Initializes a `Vec3` that describes a plotted point along the y axis.
    pub fn on_y(y: T) -> Vec3<T> {
        Vec3(T::zero(), y, T::zero())
    }
    
    /// Initializes a `Vec3` that describes a plotted point along the z axis.
    pub fn on_z(z: T) -> Vec3<T> {
        Vec3(T::zero(), T::zero(), z)
    }

    /// Converts a `Vec3` to a `Vec3` of a different type.
    /// Returns `None` if the cast was unsuccessful.
    pub fn cast<U: Scalar>(&self) -> Option<Vec3<U>> {
        match (U::from(self[X]), U::from(self[Y]), U::from(self[Z])) {
            (Some(x), Some(y), Some(z)) => Some(Vec3(x, y, z)),
            _                           => None
        }
    }

    /// Returns a `Vec3` that represents this `Vec3`'s X component.
    pub fn of_x(&self) -> Vec3<T> {
        Vec3(self[X], T::zero(), T::zero())
    }

    /// Returns a `Vec3` that represents this `Vec3`'s Y component.
    pub fn of_y(&self) -> Vec3<T> {
        Vec3(T::zero(), self[Y], T::zero())
    }
    
    /// Returns a `Vec3` that represents this `Vec3`'s Z component.
    pub fn of_z(&self) -> Vec3<T> {
        Vec3(T::zero(), T::zero(), self[Y])
    }
    
    /// Gets the x and y compontnets of this `Vec3` as a `Vec2`.
    pub fn xy(&self) -> Vec2<T> {
        Vec2(self[X], self[Y])
    }
    
    /// Gets the x and y components of this `Vec3` in inverse order as a `Vec2`.
    pub fn yx(&self) -> Vec2<T> {
        Vec2(self[Y], self[X])
    }

    /// Gets the x, y, and z components of this `Vec3` as another identity function.
    pub fn xyz(&self) -> Vec3<T> {
        self.identity()
    }

    /// Gets the x, y, and z components of this `Vec3` in inverse order as a `Vec3`.
    pub fn zyx(&self) -> Vec3<T> {
        Vec3(self[Z], self[Y], self[X])
    }

    /// Creates a `Vec3` from this `Vec3`'s y, x, and z components in that order.
    pub fn yxz(&self) -> Vec3<T> {
        Vec3(self[Y], self[X], self[Z])
    }
    
    /// Creates a `Vec3` from this `Vec3`'s z, x, and y components in that order.
    pub fn zxy(&self) -> Vec3<T> {
        Vec3(self[Z], self[X], self[Y])
    }
    
    /// Creates a `Vec3` from this `Vec3`'s x, z, and y components in that order.
    pub fn xzy(&self) -> Vec3<T> {
        Vec3(self[X], self[Z], self[Y])
    }
    
    /// Creates a `Vec3` from this `Vec3`'s y, z, and x components in that order.
    pub fn yzx(&self) -> Vec3<T> {
        Vec3(self[Y], self[Z], self[X])
    }
}

impl <T: SignedScalar> Vec3<T> {
    
    /// Initializes a `Vec3` that describes a backwards direction.
    pub fn back() -> Vec3<T> {
        Vec3(T::zero(), T::zero(), -T::one())
    }

    /// Initializes a `Vec3` that describes a downwards direction.
    pub fn down() -> Vec3<T> {
        Vec3(T::zero(), -T::one(), T::zero())
    }

    /// Initializes a `Vec3` that describes a leftwards direction.
    pub fn left() -> Vec3<T> {
        Vec3(-T::one(), T::zero(), T::zero())
    }
}

impl <T: FloatScalar> Vec3<T> {
    
    /// Returns the signed angle to this vector in radians.
    /// The sign of the angle returned is positive in the angle is in a counter-clockwise
    /// direction or negative if not.
    /// The `axis` is the viewing angle relative to the angle's direction.
    pub fn signed_angle_to(&self, _axis: Axis3) -> Vec3<T> {
        todo!()
    }

    /// Rotates this vector by a given angle in radians across a given Axis.
    pub fn rotated(&self, angle: T, axis: Axis3) -> Vec3<T> {
        self.rotated_free(angle, axis.to_vec3())
    }
    
    /// Rotates this vector by a given angle in radians across a given Axis.
    /// Compared to `rotated`, this function gives you the freedom to specify a custom Axis vector.
    pub fn rotated_free(&self, _angle: T, _axis: Vec3<T>) -> Vec3<T> {
        //Mat3::from_angle(axis, angle).xform(self)
        todo!()
    }

    /// Encodes this `Vec3` into a `Vec2` via Octahedral Encoding.
    /// Can be used to compress the Vector to two-thirds of its original size.
    /// # Note
    /// This can only be used for normalized vectors.
    pub fn octahedron_encode(&self) -> Vec2<T> {
        
        /*
         * Based on Godot's code
         */

        let mut n: Vec3<T> = *self;
                n         /= n[X].abs() + n[Y].abs() + n[Z].abs();
        
        let mut o: Vec2<T> = T::zero().vec2();
        if n[Z] >= T::zero() {
            o[X2] = n[X];
            o[Y2] = n[Y];
        } else {
            o[X2] = (T::one() - n[Y].abs()) * (if n[X] >= T::zero() { T::one() } else { -T::one() });
            o[Y2] = (T::one() - n[X].abs()) * (if n[Y] >= T::zero() { T::one() } else { -T::one() });
        }
        
        let t_05: T = T::from(0.5).unwrap();
        
        // Normalize the values.
        o[X2] = o[X2] * t_05 + t_05;
        o[Y2] = o[Y2] * t_05 + t_05;
        o
    }
    
    /// Decodes an Octahedral-Encoded `Vec2` back into a `Vec3`.
    /// # Note
    /// If the original vector was not normalized, this will not return the original value.
    pub fn octahedron_decode(encoded: Vec2<T>) -> Vec3<T> {
        
        /*
         * Based on Godot's code
         */

        let t_2: T = T::from(2).unwrap();

        let     f: Vec2<T> = Vec2(encoded[X2] * t_2 - T::one(), encoded[Y2] * t_2 - T::one());
        let mut n: Vec3<T> = Vec3(f[X2], f[Y2], T::one() - f[X2].abs() - f[Y2].abs());
        
        let t: T = -n[Z].clamp(T::zero(), T::one());
        n[X] = n[X] + (if n[X] >= T::zero() { -t } else { t });
        n[Y] = n[Y] + (if n[Y] >= T::zero() { -t } else { t });
        n.normalized()
    }

    /*/// Returns the outer product of this vector.
    pub fn outer(&self) -> Mat3<T> {
        todo!()
    }*/
}


/*
    Global Operations
        Base Arithmetic
*/


impl <T: Scalar> Add for Vec3<T> {
    type Output = Vec3<T>;
    fn add(self, other: Self) -> Self::Output {
        Vec3(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl <T: Scalar> Sub for Vec3<T> {
    type Output = Vec3<T>;
    fn sub(self, other: Self) -> Self::Output {
        Vec3(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl <T: Scalar> Mul for Vec3<T> {
    type Output = Vec3<T>;
    fn mul(self, other: Self) -> Self::Output {
        Vec3(self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }
}

impl <T: Scalar> Div for Vec3<T> {
    type Output = Vec3<T>;
    fn div(self, other: Self) -> Self::Output {
        Vec3(self.0 / other.0, self.1 / other.1, self.2 / other.2)
    }
}

impl <T: Scalar> Rem for Vec3<T> {
    type Output = Vec3<T>;
    fn rem(self, other: Self) -> Self::Output {
        Vec3(self.0 % other.0, self.1 % other.1, self.2 % other.2)
    }
}

impl <T: SignedScalar> Neg for Vec3<T> {
    type Output = Vec3<T>;
    fn neg(self) -> Self::Output {
        Vec3(-self.0, -self.1, -self.2)
    }
}


/*
    Global Operations
        Reference Arithmetic
*/


impl <'a, T: Scalar> Add<&'a Vec3<T>> for &'a Vec3<T> {
    type Output = Vec3<T>;
    fn add(self, other: Self) -> Self::Output {
        Vec3(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl <'a, T: Scalar> Sub<&'a Vec3<T>> for &'a Vec3<T> {
    type Output = Vec3<T>;
    fn sub(self, other: Self) -> Self::Output {
        Vec3(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl <'a, T: Scalar> Mul<&'a Vec3<T>> for &'a Vec3<T> {
    type Output = Vec3<T>;
    fn mul(self, other: Self) -> Self::Output {
        Vec3(self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }
}

impl <'a, T: Scalar> Div<&'a Vec3<T>> for &'a Vec3<T> {
    type Output = Vec3<T>;
    fn div(self, other: Self) -> Self::Output {
        Vec3(self.0 / other.0, self.1 / other.1, self.2 / other.2)
    }
}

impl <'a, T: Scalar> Rem<&'a Vec3<T>> for &'a Vec3<T> {
    type Output = Vec3<T>;
    fn rem(self, other: Self) -> Self::Output {
        Vec3(self.0 % other.0, self.1 % other.1, self.2 % other.2)
    }
}

impl <T: SignedScalar> Neg for &Vec3<T> {
    type Output = Vec3<T>;
    fn neg(self) -> Self::Output {
        Vec3(-self.0, -self.1, -self.2)
    }
}


/*
    Global Operations
        Reference vs Base Arithmetic
*/


impl <T: Scalar> Add<&Vec3<T>> for Vec3<T> {
    type Output = Vec3<T>;
    fn add(self, other: &Self) -> Self::Output {
        Vec3(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl <T: Scalar> Sub<&Vec3<T>> for Vec3<T> {
    type Output = Vec3<T>;
    fn sub(self, other: &Self) -> Self::Output {
        Vec3(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl <T: Scalar> Mul<&Vec3<T>> for Vec3<T> {
    type Output = Vec3<T>;
    fn mul(self, other: &Self) -> Self::Output {
        Vec3(self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }
}

impl <T: Scalar> Div<&Vec3<T>> for Vec3<T> {
    type Output = Vec3<T>;
    fn div(self, other: &Self) -> Self::Output {
        Vec3(self.0 / other.0, self.1 / other.1, self.2 / other.2)
    }
}

impl <T: Scalar> Rem<&Vec3<T>> for Vec3<T> {
    type Output = Vec3<T>;
    fn rem(self, other: &Self) -> Self::Output {
        Vec3(self.0 % other.0, self.1 % other.1, self.2 % other.2)
    }
}


/*
    Global Operations
        Base vs Scalar Arithmetic
*/


impl <T: Scalar> Add<T> for Vec3<T> {
    type Output = Vec3<T>;
    fn add(self, other: T) -> Self::Output {
        self + Vec3(other, other, other)
    }
}

impl <T: Scalar> Sub<T> for Vec3<T> {
    type Output = Vec3<T>;
    fn sub(self, other: T) -> Self::Output {
        self - Vec3(other, other, other)
    }
}

impl <T: Scalar> Mul<T> for Vec3<T> {
    type Output = Vec3<T>;
    fn mul(self, other: T) -> Self::Output {
        self * Vec3(other, other, other)
    }
}

impl <T: Scalar> Div<T> for Vec3<T> {
    type Output = Vec3<T>;
    fn div(self, other: T) -> Self::Output {
        self / Vec3(other, other, other)
    }
}

impl <T: Scalar> Rem<T> for Vec3<T> {
    type Output = Vec3<T>;
    fn rem(self, other: T) -> Self::Output {
        self % Vec3(other, other, other)
    }
}


/*
    Global Operations
        Reference vs Scalar Arithmetic
*/


impl <T: Scalar> Add<T> for &Vec3<T> {
    type Output = Vec3<T>;
    fn add(self, other: T) -> Self::Output {
        self + &Vec3(other, other, other)
    }
}

impl <T: Scalar> Sub<T> for &Vec3<T> {
    type Output = Vec3<T>;
    fn sub(self, other: T) -> Self::Output {
        self - &Vec3(other, other, other)
    }
}

impl <T: Scalar> Mul<T> for &Vec3<T> {
    type Output = Vec3<T>;
    fn mul(self, other: T) -> Self::Output {
        self * &Vec3(other, other, other)
    }
}

impl <T: Scalar> Div<T> for &Vec3<T> {
    type Output = Vec3<T>;
    fn div(self, other: T) -> Self::Output {
        self / &Vec3(other, other, other)
    }
}

impl <T: Scalar> Rem<T> for &Vec3<T> {
    type Output = Vec3<T>;
    fn rem(self, other: T) -> Self::Output {
        self % &Vec3(other, other, other)
    }
}


/*
    Global Operations
        Assignment & Arithmetic
*/


impl <T: Scalar, V: Vectorized<T, Vec3<T>>> AddAssign<V> for Vec3<T> {
    fn add_assign(&mut self, other: V) -> () {
        *self = *self + other.dvec();
    }
}

impl <T: Scalar, V: Vectorized<T, Vec3<T>>> SubAssign<V> for Vec3<T> {
    fn sub_assign(&mut self, other: V) -> () {
        *self = *self - other.dvec();
    }
}

impl <T: Scalar, V: Vectorized<T, Vec3<T>>> MulAssign<V> for Vec3<T> {
    fn mul_assign(&mut self, other: V) -> () {
        *self = *self * other.dvec();
    }
}

impl <T: Scalar, V: Vectorized<T, Vec3<T>>> DivAssign<V> for Vec3<T> {
    fn div_assign(&mut self, other: V) -> () {
        *self = *self / other.dvec();
    }
}

impl <T: Scalar, V: Vectorized<T, Vec3<T>>> RemAssign<V> for Vec3<T> {
    fn rem_assign(&mut self, other: V) -> () {
        *self = *self % other.dvec();
    }
}


/*
    Global
        Behaviours
*/


impl <T: Scalar> Default for Vec3<T> {
    fn default() -> Self {
        Vec3(T::default(), T::default(), T::default())
    }
}

impl <T: Scalar> Display for Vec3<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vec2({}, {}, {})", self.0, self.1, self.2)
    }
}


/*
    Vectorized
        Trait
*/


impl <T: Scalar> Vectorized<T, Vec3<T>> for T {
    fn attempt_get_scalar(self) -> Option<T> {
        Some(self)
    }

    fn dvec(self) -> Vec3<T> {
        Vec3(self, self, self)
    }
}

impl <T: Scalar> Vectorized<T, Vec3<T>> for (T, T, T) {
    fn attempt_get_scalar(self) -> Option<T> {
        None
    }

    fn dvec(self) -> Vec3<T> {
        Vec3(self.0, self.1, self.2)
    }
}

impl <T: Scalar> Vectorized<T, Vec3<T>> for (Vec2<T>, T) {
    fn attempt_get_scalar(self) -> Option<T> {
        None
    }

    fn dvec(self) -> Vec3<T> {
        Vec3(self.0.0, self.0.1, self.1)
    }
}

impl <T: Scalar> Vectorized<T, Vec3<T>> for (T, Vec2<T>) {
    fn attempt_get_scalar(self) -> Option<T> {
        None
    }

    fn dvec(self) -> Vec3<T> {
        Vec3(self.0, self.1.0, self.1.1)
    }
}

impl <T: Scalar> Vectorized<T, Vec3<T>> for Vec3<T> {
    fn attempt_get_scalar(self) -> Option<T> {
        None
    }

    fn dvec(self) -> Vec3<T> {
        self
    }
}

pub trait Vectorized3D<T: Scalar> {
    
    /// Converts a type that can be represented as a Vector of 3 as a `Vec3`.
    /// This is the public interface for the `Vectorized` trait.
    fn vec3(self) -> Vec3<T>;
}

impl <T: Scalar + Vectorized<T, Vec3<T>>> Vectorized3D<T> for T {
    fn vec3(self) -> Vec3<T> {
        self.dvec()
    }
}

impl <T: Scalar + Vectorized<T, Vec3<T>>> Vectorized3D<T> for (T, T, T) {
    fn vec3(self) -> Vec3<T> {
        self.dvec()
    }
}

impl <T: Scalar + Vectorized<T, Vec3<T>>> Vectorized3D<T> for (T, Vec2<T>) {
    fn vec3(self) -> Vec3<T> {
        self.dvec()
    }
}

impl <T: Scalar + Vectorized<T, Vec3<T>>> Vectorized3D<T> for (Vec2<T>, T) {
    fn vec3(self) -> Vec3<T> {
        self.dvec()
    }
}
