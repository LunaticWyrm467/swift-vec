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

//!
//! Created by LunaticWyrm467
//!

//?
//? Contains the Vector trait which is used for all vectors (tuples).
//?

pub mod v2d;

use std::{ ops::{ Add, Sub, Mul, Div, Rem, Neg, AddAssign, SubAssign, MulAssign, DivAssign, RemAssign }, f32::consts::FRAC_PI_2 };

use approx::relative_eq;

use crate::scalar::{ Scalar, FloatScalar, SignedScalar, IntScalar };


/*
    Vector
        Trait
*/


pub trait VectorAbstract<T: Scalar, V: VectorAbstract<T, V>>
where
    Self:
        Clone + PartialEq + PartialOrd + Default + Add + Sub + Mul + Div + Rem + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + std::fmt::Display + std::fmt::Debug +
        Add<Output = V> + Sub<Output = V> + Mul<Output = V> + Div<Output = V> + Rem<Output = V> +
        Add<T, Output = V> + Sub<T, Output = V> + Mul<T, Output = V> + Div<T, Output = V> + Rem<T, Output = V>
{}

pub trait Vector<T: Scalar, V: Vector<T, V, A>, A>: VectorAbstract<T, V> {

    /// Returns the rank or dimension of the vector.
    fn rank() -> usize;

    /// Gets the value at the specified axis.
    fn get(&self, axis: A) -> T;

    /// A simple identity function. Useful for trait implementations where trait bounds need to be kept.
    fn identity(&self) -> &V;

    /// Returns a vector containing zeros.
    fn ones_like() -> V;
    
    /// Returns a vector containing ones.
    fn zeros_like() -> V;

    /// Returns a vector encompassing a scalar value.
    fn scalar_like(x: T) -> V {
        Self::ones_like() * x
    }

    /// Calculates the sum of the vector.
    fn sum(&self) -> T;

    /// Calculates the product of the vector.
    fn product(&self) -> T;

    /// Calculates the average of the vector.
    fn average(&self) -> T {
        self.sum() / T::from_usize(Self::rank()).unwrap()
    }

    /// Returns the axis of the highest value.
    fn argmax(&self) -> A;

    /// Returns the axis of the lowest value.
    fn argmin(&self) -> A;

    /// Returns the maximum between this vector and another.
    fn v_max(&self, other: &V) -> V;

    /// Returns the maximum between this vector and a scalar threshold.
    fn max(&self, threshold: T) -> V {
        self.v_max(&Self::scalar_like(threshold))
    }

    /// Returns the minimum between this vector and another.
    fn v_min(&self, other: &V) -> V;

    /// Returns the minimum between this vector and a scalar threshold.
    fn min(&self, threshold: T) -> V {
        self.v_min(&Self::scalar_like(threshold))
    }

    /// Clamps this vector between a minimum and maximum vector and returns the result.
    fn v_clamp(&self, min: &V, max: &V) -> V {
        self.v_min(max).v_max(min)
    }

    /// Clamps this vector between a minimum and maximum scalar threshold and returns the result.
    fn clamp(&self, min: T, max: T) -> V {
        self.min(max).max(min)
    }
}

pub trait SignedVector<T: SignedScalar, V: SignedVector<T, V, A>, A>: Vector<T, V, A>
where
    Self: Neg<Output = V> {

    /// Returns the sign of the vector.
    /// For example:
    /// ```rust
    /// use swift_vec::vector::v2d::Vec2;
    /// use swift_vec::vector::SignedVector;
    /// 
    /// let sign: Vec2<f32> = Vec2(-1.0, 1.0).signum();
    /// assert_eq!(sign, Vec2(-1f32, 1f32));
    /// ```
    fn signum(&self) -> V;

    /// Gets the absolute value of a vector.
    fn abs(&self) -> V;
}

pub trait IntVector<T: IntScalar<T>, V: IntVector<T, V, A>, A>: Vector<T, V, A> {

    /// Raises the vector by a certain power vector.
    fn v_pow(&self, pow: &V) -> V;

    /// Raises the vector by a certain power.
    fn pow(&self, pow: T) -> V {
        self.v_pow(&Self::scalar_like(pow))
    }

    /// Derives the log of a vector using a base vector.
    fn v_log(&self, base: &V) -> V;

    /// Derives the log of the vector using a base.
    fn log(&self, base: T) -> V {
        self.v_log(&Self::scalar_like(base))
    }
}

pub trait FloatVector<T: FloatScalar, V: FloatVector<T, V, A>, A>: SignedVector<T, V, A> {

    //=====// Trigonometry //=====//
    /// Initializes a vector from an angle.
    fn from_angle(angle: T) -> V;

    /// Calculates the angle of a vector in respect to the positive x-axis.
    /// # Returns
    /// The angle of the vector in radians.
    fn angle(&self) -> T;

    /// Calculates the angle to another vector.
    /// # Returns
    /// The angle in radians.
    fn angle_to(&self, other: &V) -> T {
        -(self.cross(&other)).atan2(self.dot(&other))
    }

    /// Calculates the angle between the line connecting the two positions and the x-axis.
    /// # Returns
    /// The angle in radians.
    fn angle_between(&self, other: &V) -> T {
        (other.to_owned() - self.identity().to_owned()).angle()
    }

    /// Rotates this vector by a given angle in radians.
    fn rotated(&self, angle: T) -> V;

    /// Rotates this vector by 90 degrees counter clockwise.
    fn orthogonal(&self) -> V {
        self.rotated(T::from(FRAC_PI_2).unwrap())
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
        Self::ones_like() / self.sin()
    }

    /// Calculates the secant of this vector.
    fn sec(&self) -> V {
        Self::ones_like() / self.cos()
    }

    /// Calculates the cotangent of this vector.
    fn cot(&self) -> V {
        Self::ones_like() / self.tan()
    }

    /// Calculates the arc cosecant of this vector.
    fn acsc(&self) -> V {
        (Self::ones_like() / self.identity().to_owned()).asin()
    }

    /// Calculates the arc secant of this vector.
    fn asec(&self) -> V {
        (Self::ones_like() / self.identity().to_owned()).acos()
    }

    /// Calculates the arc cotangent of this vector.
    fn acot(&self) -> V {
        (Self::ones_like() / self.identity().to_owned()).atan()
    }


    //=====// Algebra //=====//
    /// Partially calculates the magnitude of the vector via the pythagorean theorem,
    /// but without applying the final step of squaring the value.
    fn magnitude_squared(&self) -> T {
        self.sqr().sum()
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

    /// Limits the magnitude or length of the vector to a certain value.
    fn limit(&self, limit: T) -> V {
        
        // If the magnitude is less than or equal to the limit, return the vector.
        if self.magnitude() <= limit {
            return self.identity().to_owned();
        }

        // Otherwise, calculate the scale factor and multiply the vector by it.
        let scale: T = limit / self.magnitude();
        self.identity().to_owned() * scale
    }

    /// Normalizes the vector to a length of 1.
    fn normalized(&self) -> V {
        self.identity().to_owned() / self.magnitude()
    }

    /// Derives the unit vector from this vector.
    /// This is just an alias to `normalized()`.
    fn unit(&self) -> V {
        self.normalized()
    }

    /// Computes the reflection of this vector about the given normal vector.
    fn reflect(&self, normal: &V) -> V {
        
        // Normalize the normals and calculate the dot product between this vector and the normals
        // to determine the alignment.
        let normals_normalized: V = normal.normalized();
        let alignment:          T = self.dot(&normals_normalized);

        // Compute and return the reflection.
        self.identity().to_owned() - (normals_normalized * alignment * T::from(2).unwrap())
    }

    /// Computes the refraction of this vector about the given normal vector and a refraction index.
    fn refract(&self, normal: &V, refraction_index: T) -> V {
        
        // Normalize the normals and calculate the dot product between this vector and the normals
        // to determine the alignment.
        let normals_normalized: V = normal.normalized();
        let alignment:          T = self.dot(&normals_normalized);

        // Compute the discriminant - the 'k' in Snell's law.
        let discriminant: T = T::one() - refraction_index * refraction_index * (T::one() - alignment * alignment);

        // If the discriminant is negative, then an internal reflection occurred.
        if discriminant < T::zero() {
            Self::zeros_like()
        } else {

            // Otherwise, compute and return the refraction using Snell's law.
            (self.identity().to_owned() * refraction_index) - (normals_normalized * (refraction_index * alignment + discriminant.sqrt()))
        }
    }

    /// Calculates the direction from this vector to another.
    fn direction_to(&self, other: &V) -> V {
        (other.to_owned() - self.identity().to_owned()).normalized()
    }

    /// Computes the squared distance between two vectors.
    /// This is faster than `distance_to()` because it avoids a square root.
    fn distance_squared_to(&self, other: &V) -> T;

    /// Computes the distance between two vectors using the pythagorean theorem.
    fn distance_to(&self, other: &V) -> T {
        self.distance_squared_to(other).sqrt()
    }

    /// Moves towards a target vector by a delta value.
    /// Does not exceed or pass the target vector.
    fn move_towards(&self, target: &V, delta: T) -> V {
        let direction: V = self.direction_to(target);
        let position:  V = self.identity().to_owned() + (direction * delta);
        position.v_min(target)
    }


    //=====// Interpolation //=====//
    /// A simple linear interpolation between two vectors sampled at point `t`.
    fn lerp(&self, other: &V, t: T) -> V {
        let identity: &V = self.identity();
        identity.to_owned() + (other.to_owned() - identity.to_owned()) * t
    }

    /// Calculates the derivative of the Bézier curve set by this vector and the given control and terminal points
    /// at position `t`.
    fn bezier_derivative(&self, control_1: &V, control_2: &V, terminal: &V, t: T) -> V;

    /// Calculates the point on the Bézier curve set by this vector and the given control and terminal points
    /// at position `t`.
    fn bezier_sample(&self, control_1: &V, control_2: &V, terminal: &V, t: T) -> V;

    /// Calculates and samples the cubic interpolation between this vector and another
    /// given `pre_start` and `post_terminal` vectors as handles, and a given `t` value.
    fn cubic_interpolate(&self, terminal: &V, pre_start: &V, post_terminal: &V, t: T) -> V;

    /// Similar to `cubic_interpolate`, but it has additional time parameters `terminal_t`, `pre_start_t`, and `post_terminal_t`.
    /// This can be smoother than `cubic_interpolate` in certain instances.
    fn cubic_interpolate_in_time(&self, terminal: &V, pre_start: &V, post_terminal: &V, t0: T, terminal_t: &V, pre_start_t: &V, post_terminal_t: &V) -> V;

    /// Spherically interpolates between two vectors.
    /// This interpolation is focused on the length or magnitude of the vectors. If the magnitudes are equal,
    /// the interpolation is linear and behaves the same way as `lerp()`.
    fn slerp(&self, other: &V, t: T) -> V {
        let theta: T = self.angle_to(other);
        self.rotated(theta * t)
    }

    /// Returns the result of sliding a vector along a plane as defined by a normal vector.
    fn slide(&self, normal: &V) -> V {
        self.identity().to_owned() - (normal.to_owned() * self.dot(normal))
    }


    //=====// Linear Algebra //=====//
    /// Calculates the dot product of two vectors.
    fn dot(&self, other: &V) -> T;

    /// Calculates the cross product of two vectors.
    /// Note: in 2D vectors, the cross product is the z-component of the 3D cross product of the vectors broadcasted to 3D.
    fn cross(&self, other: &V) -> T;

    /// Projects this vector onto another vector.
    fn project(&self, other: &V) -> V {
        
        // Calculate the scalar projection via the dot product and the magnitude of the other vector.
        let scalar_projection: T = self.dot(other) / other.magnitude();

        // Multiply the scalar by the unit vector (a normalized vector) of the other vector.
        let unit: V = other.unit();
        unit * scalar_projection
    }


    //=====// Operations //=====//
    /// Calculates the square root of the vector.
    fn sqrt(&self) -> V;

    /// Calculates the square of the vector through multiplication.
    fn sqr(&self) -> V;

    /// Raises the vector by a certain power vector.
    fn v_pow(&self, pow: &V) -> V;

    /// Raises the vector by a certain power.
    fn pow(&self, pow: T) -> V {
        self.v_pow(&Self::scalar_like(pow))
    }

    /// Uses this vector as a power over Euler's constant and returns the result.
    fn exp(&self) -> V {
        Self::scalar_like(T::E()).v_pow(self.identity())
    }

    /// Derives the log of a vector using a base vector.
    fn v_log(&self, base: &V) -> V;

    /// Derives the log of the vector using a base.
    fn log(&self, base: T) -> V {
        self.v_log(&Self::scalar_like(base))
    }

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
    fn snap_to(&self, multiple: &V) -> V {
        let diff: V = self.identity().to_owned() % multiple.to_owned();
        self.identity().to_owned() - diff
    }

    /// Determines if the vector is approximately equal to another vector.
    fn approx_eq(&self, other: &V) -> bool;


    //=====// Checks & Flags //=====//
    /// Returns whether this vector is finite.
    fn is_finite(&self) -> bool {
        self.sum().is_finite()   // If either of the components is not finite, then the whole sum will be NaN or infinite.
    }

    /// Returns whether this vector is normalized.
    fn is_normalized(&self) -> bool {
        let magnitude: T = self.magnitude();
        relative_eq!(magnitude, T::one())   // TODO: Figure out type constraints on epsilon.
    }

    /// Returns if this vector is approximately zero.
    fn is_zero_approx(&self) -> bool;


}