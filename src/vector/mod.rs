//===================================================================================================================================================================================//
//
//  /$$    /$$                      /$$                        
// | $$   | $$                     | $$                        
// | $$   | $$ /$$$$$$   /$$$$$$$ /$$$$$$    /$$$$$$   /$$$$$$ 
// |  $$ / $$//$$__  $$ /$$_____/|_  $$_/   /$$__  $$ /$$__  $$
//  \  $$ $$/| $$$$$$$$| $$        | $$    | $$  \ $$| $$  \__/
//   \  $$$/ | $$_____/| $$        | $$ /$$| $$  | $$| $$      
//    \  $/  |  $$$$$$$|  $$$$$$$  |  $$$$/|  $$$$$$/| $$      
//     \_/    \_______/ \_______/   \___/   \______/ |__/
//
//===================================================================================================================================================================================//

//?
//? Created by LunaticWyrm467 and others.
//? 
//? All code is licensed under the MIT license.
//? Feel free to reproduce, modify, and do whatever.
//?

//!
//! The vector module contains the definitions of the various vector types, such as `Vec2`, `Vec3`, and `Vec4`.
//! Also contains global traits and functions that are shared between all vector types.
//!

mod v2d;
mod v3d;

use core::{
    ops::{ Add, Sub, Mul, Div, Rem, Neg, AddAssign, SubAssign, MulAssign, DivAssign, RemAssign, Index, IndexMut },
    f32::consts::FRAC_PI_2,
    fmt::{ self, Debug, Display }
};

#[cfg(feature = "alloc")]
use alloc::{ vec::Vec, vec };

    use crate::vectorized::Vectorized;
    use crate::scalar::{ Scalar, FloatScalar, SignedScalar, IntScalar };
pub use v2d::{ Axis2, Vec2, Vectorized2D };
pub use v3d::{ Axis3, SignedAxis3, Vec3, Vectorized3D };


/*
    Vector
        Trait
*/


pub trait VectorAbstract<T: Scalar + Vectorized<T, V>, V: VectorAbstract<T, V>>
where
    Self:
        Vectorized<T, V> +
        Clone + Copy + PartialEq + PartialOrd + Default + Display + Debug +
        Add + Sub + Mul + Div + Rem + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign +
        Add<Output = V> + Sub<Output = V> + Mul<Output = V> + Div<Output = V> + Rem<Output = V> +
        Add<T, Output = V> + Sub<T, Output = V> + Mul<T, Output = V> + Div<T, Output = V> + Rem<T, Output = V>
{}

pub trait Vector<T: Scalar + Vectorized<T, V>, V: Vector<T, V, A>, A>: VectorAbstract<T, V> {
    
    /// A vector with all of its fields set to zero.
    const ZERO: V;
    
    /// A vector with all of its fields set to one.
    const ONE: V;

    //=====// Getters //=====//
    /// A simple identity function. Useful for trait implementations where trait bounds need to be kept.
    fn identity(&self) -> V;

    /// Returns the rank or dimension of the vector.
    fn rank() -> usize;

    /// Converts the vector into a `Vec<T>`.
    #[cfg(feature = "alloc")]
    fn to_vec(&self) -> Vec<T>;


    //=====// Operations //=====//
    /// Calculates the sum of the vector.
    fn sum(&self) -> T;

    /// Calculates the product of the vector.
    fn product(&self) -> T;
    
    /// Calculates the dot product of two vectors.
    fn dot(&self, other: V) -> T;

    /// Calculates the average of the vector.
    fn average(&self) -> T {
        self.sum() / T::from(Self::rank()).unwrap()
    }

    /// Returns the axis of the highest value.
    fn argmax(&self) -> A;

    /// Returns the axis of the lowest value.
    fn argmin(&self) -> A;

    /// Returns the maximum between this vector and another value.
    fn max<I: Vectorized<T, V>>(&self, other: I) -> V;

    /// Returns the minimum between this vector and another value.
    fn min<I: Vectorized<T, V>>(&self, other: I) -> V;

    /// Clamps this vector between a minimum and maximum value and returns the result.
    fn clamp<I: Vectorized<T, V>>(&self, min: I, max: I) -> V;
}

pub trait SignedVector<T: SignedScalar + Vectorized<T, V>, V: SignedVector<T, V, A, C>, A, C: Vectorized<T, V>>: Vector<T, V, A>
where
    Self: Neg<Output = V> {

    /// Returns the sign of the vector.
    /// For example:
    /// ```rust
    /// use swift_vec::vector::Vec2;
    /// use swift_vec::prelude::SignedVector;
    /// 
    /// let sign: Vec2<f32> = Vec2(-1.0, 1.0).signum();
    /// assert_eq!(sign, Vec2(-1f32, 1f32));
    /// ```
    fn signum(&self) -> V;

    /// Gets the absolute value of a vector.
    fn abs(&self) -> V;
    
    /// Calculates the cross product of two vectors.
    /// Note: in 2D vectors, the cross product is the z-component of the 3D cross product of the vectors broadcasted to 3D.
    fn cross(&self, other: V) -> C;
}

pub trait IntVector<T: IntScalar<T> + Vectorized<T, V>, V: IntVector<T, V, A>, A>: Vector<T, V, A> {

    /// Raises the vector by a certain power.
    fn pow<I: Vectorized<T, V>>(&self, pow: I) -> V;

    /// Derives the log of the vector using a base.
    fn log<I: Vectorized<T, V>>(&self, base: I) -> V;
}

pub trait FloatVector<T: FloatScalar + Vectorized<T, V>, V: FloatVector<T, V, A, C>, A, C: Vectorized<T, V>>: SignedVector<T, V, A, C> {

    //=====// Trigonometry //=====//
    /// Calculates the angle to another vector.
    /// # Returns
    /// The angle in radians.
    fn angle_to(&self, other: V) -> T {
        let cross: C = self.cross(other);
        match cross.attempt_get_scalar() {
            Some(cross_scalar) => -cross_scalar.atan2(self.dot(other)),
            None               => -cross.dvec().length().atan2(self.dot(other))
        }
    }

    /// Converts this vector to radians.
    fn to_radians(&self) -> V;

    /// Converts this vector to degrees.
    fn to_degrees(&self) -> V;

    /// Calculates the sin of this vector.
    fn sin(&self) -> V;

    /// Calculates the cosine of this vector.
    fn cos(&self) -> V;

    /// Calculates the tangent of this vector.
    fn tan(&self) -> V;

    /// Calculates the arc sin of this vector.
    fn asin(&self) -> V;

    /// Calculates the arc cosine of this vector.
    fn acos(&self) -> V;

    /// Calculates the arc tangent of this vector.
    fn atan(&self) -> V;

    /// Calculates the cosecant of this vector.
    fn csc(&self) -> V {
        T::one().dvec() / self.sin()
    }

    /// Calculates the secant of this vector.
    fn sec(&self) -> V {
        T::one().dvec() / self.cos()
    }

    /// Calculates the cotangent of this vector.
    fn cot(&self) -> V {
        T::one().dvec() / self.tan()
    }

    /// Calculates the arc cosecant of this vector.
    fn acsc(&self) -> V {
        (T::one().dvec() / self.identity()).asin()
    }

    /// Calculates the arc secant of this vector.
    fn asec(&self) -> V {
        (T::one().dvec() / self.identity()).acos()
    }

    /// Calculates the arc cotangent of this vector.
    fn acot(&self) -> V {
        (T::one().dvec() / self.identity()).atan()
    }


    //=====// Algebra //=====//
    /// Partially calculates the magnitude of the vector via the pythagorean theorem,
    /// but without applying the final step of squaring the value.
    fn magnitude_squared(&self) -> T {
        self.pow2().sum()
    }

    /// Calculates the magnitude of a vector via the pythagorean theorem.
    fn magnitude(&self) -> T {
        self.magnitude_squared().sqrt()
    }

    /// Calculates the length of the vector.
    /// This is just an alias to `magnitude()`.
    fn length(&self) -> T {
        self.magnitude()
    }

    /// Calculates the inverse magnitude of a vector.
    fn inv_magnitude(&self) -> T {
        self.identity().dot(self.identity()).inv_sqrt()
    }

    /// Calculates the inverse length of a vector.
    /// This is just an alias to `inv_magnitude`.
    fn inv_length(&self) -> T {
        self.inv_magnitude()
    }

    /// Limits the magnitude or length of the vector to a certain value.
    fn limit(&self, limit: T) -> V {
        
        // If the magnitude is less than or equal to the limit, return the vector.
        if self.magnitude() <= limit {
            return self.identity();
        }

        // Otherwise, calculate the scale factor and multiply the vector by it.
        let scale: T = limit / self.magnitude();
        self.identity() * scale
    }

    /// Normalizes the vector to a length of 1.
    fn normalized(&self) -> V {
        self.identity() * self.inv_magnitude()
    }

    /// Derives the unit vector from this vector.
    /// This is just an alias to `normalized()`.
    fn unit(&self) -> V {
        self.normalized()
    }

    /// Computes the reflection of this vector about the given normal vector.
    fn reflect(&self, normal: V) -> V {
        
        // Normalize the normals and calculate the dot product between this vector and the normals
        // to determine the alignment.
        let normals_normalized: V = normal.normalized();
        let alignment:          T = self.dot(normals_normalized);

        // Compute and return the reflection.
        self.identity() - (normals_normalized * alignment * T::from(2).unwrap())
    }

    /// An alias for the `reflect()` function.
    fn bounce(&self, normal: V) -> V {
        self.reflect(normal)
    }

    /// Computes the refraction of this vector about the given normal vector and a refraction index.
    fn refract(&self, normal: V, refraction_index: T) -> V {
        
        // Normalize the normals and calculate the dot product between this vector and the normals
        // to determine the alignment.
        let normals_normalized: V = normal.normalized();
        let alignment:          T = self.dot(normals_normalized);

        // Compute the discriminant - the 'k' in Snell's law.
        let discriminant: T = T::one() - refraction_index * refraction_index * (T::one() - alignment * alignment);

        // If the discriminant is negative, then an internal reflection occurred.
        if discriminant < T::zero() {
            T::zero().dvec()
        } else {   // Otherwise, compute and return the refraction using Snell's law.
            (self.identity() * refraction_index) - (normals_normalized * (refraction_index * alignment + discriminant.sqrt()))
        }
    }

    /// Calculates the direction from this vector to another.
    fn direction_to(&self, other: V) -> V {
        (other - self.identity()).normalized()
    }

    /// Computes the squared distance between two vectors.
    /// This is faster than `distance_to()` because it avoids a square root.
    fn distance_squared_to(&self, other: V) -> T;

    /// Computes the distance between two vectors using the pythagorean theorem.
    fn distance_to(&self, other: V) -> T {
        self.distance_squared_to(other).sqrt()
    }

    /// Moves towards a target vector by a delta value.
    /// Does not exceed or pass the target vector.
    fn move_towards(&self, target: V, delta: T) -> V {
        let direction: V = self.direction_to(target);
        let position:  V = self.identity() + (direction * delta);
        position.min(target)
    }


    //=====// Interpolation //=====//
    /// A simple linear interpolation between two vectors sampled at point `t`.
    fn lerp(&self, other: V, t: T) -> V {
        let identity: V = self.identity();
        identity + (other - identity) * t
    }

    /// A `smoothstep` implementation similar to that of OpenGL's, which uses smooth Hermite
    /// interpolation between the values `a` and `b` for `self.
    fn smoothstep<I: Vectorized<T, V>>(&self, a: I, b: I) -> V;

    /// Calculates the derivative of the Bézier curve set by this vector and the given control and terminal points
    /// at position `t`.
    fn bezier_derivative(&self, control_1: V, control_2: V, terminal: V, t: T) -> V;

    /// Calculates the point on the Bézier curve set by this vector and the given control and terminal points
    /// at position `t`.
    fn bezier_sample(&self, control_1: V, control_2: V, terminal: V, t: T) -> V;

    /// Calculates and samples the cubic interpolation between this vector and another
    /// given `pre_start` and `post_terminal` vectors as handles, and a given `t` value.
    fn cubic_interpolate(&self, b: V, pre_a: V, post_b: V, weight: T) -> V;

    /// Similar to `cubic_interpolate`, but it has additional time parameters `terminal_t`, `pre_start_t`, and `post_terminal_t`.
    /// This can be smoother than `cubic_interpolate` in certain instances.
    fn cubic_interpolate_in_time(&self, b: V, pre_a: V, post_b: V, weight: T, b_t: T, pre_a_t: T, post_b_t: T) -> V;

    /// Returns the result of sliding a vector along a plane as defined by a normal vector.
    fn slide(&self, normal: V) -> V {
        self.identity() - (normal * self.dot(normal))
    }

    //=====// Operations //=====//
    /// Projects this vector onto another vector.
    fn project(&self, other: V) -> V {
        
        // Calculate the scalar projection via the dot product and the magnitude of the other vector.
        let scalar_projection: T = self.dot(other) / other.magnitude();

        // Multiply the scalar by the unit vector (a normalized vector) of the other vector.
        let unit: V = other.unit();
        unit * scalar_projection
    }

    /// Calculates the square root of the vector.
    fn sqrt(&self) -> V;

    /// Calculates the inverse square root of the vector.
    fn inv_sqrt(&self) -> V {
        T::one().dvec() / self.sqrt()
    }

    /// Calculates the square of the vector through multiplication.
    fn pow2(&self) -> V;

    /// Raises the vector by a certain power.
    fn pow<I: Vectorized<T, V>>(&self, pow: I) -> V;

    /// Uses this vector as a power over Euler's constant and returns the result.
    fn exp(&self) -> V {
        T::E().dvec().pow(self.identity())
    }

    /// Derives the log of the vector using a base.
    fn log<I: Vectorized<T, V>>(&self, base: I) -> V;

    /// Derives the natural log of a vector.
    fn ln(&self) -> V {
        self.log(T::E())
    }

    /// Returns a new vector rounded down from this one.
    fn floor(&self) -> V;

    /// Returns a new vector rounded up from this one.
    fn ceil(&self) -> V;

    /// Returns a new vector rounded by normal conventions from this one.
    fn round(&self) -> V;

    /// Snaps the vector to the nearest multiple of a given vector.
    fn snap_to(&self, multiple: V) -> V {
        let diff: V = self.identity() % multiple;
        self.identity() - diff
    }

    /// Modulates a value so that it stays between 0-1.
    fn fract(&self) -> V {
        self.identity() - self.floor()
    }

    /// Determines if the vector is approximately equal to another vector.
    fn approx_eq(&self, other: V) -> bool;


    //=====// Checks & Flags //=====//
    /// Returns whether this vector is finite.
    fn is_finite(&self) -> bool {
        self.sum().is_finite()   // If either of the components is not finite, then the whole sum will be NaN or infinite.
    }

    /// Returns whether this vector contains a NaN value.
    fn is_nan(&self) -> bool {
        self.sum().is_nan()
    }

    /// Returns whether this vector is normalized.
    fn is_normalized(&self) -> bool {
        self.magnitude().approx_eq(T::one())
    }

    /// Returns if this vector is approximately zero.
    fn is_zero_approx(&self) -> bool {
        self.approx_eq(T::zero().dvec())
    }
}
