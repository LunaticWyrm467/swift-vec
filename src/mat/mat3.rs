//===================================================================================================================================================================================//
//
//  /$$      /$$             /$$               /$$                  /$$$$$$             /$$$$$$ 
// | $$$    /$$$            | $$              |__/                 /$$__  $$           /$$__  $$
// | $$$$  /$$$$  /$$$$$$  /$$$$$$    /$$$$$$  /$$ /$$   /$$      |__/  \ $$ /$$   /$$|__/  \ $$
// | $$ $$/$$ $$ |____  $$|_  $$_/   /$$__  $$| $$|  $$ /$$/         /$$$$$/|  $$ /$$/   /$$$$$/
// | $$  $$$| $$  /$$$$$$$  | $$    | $$  \__/| $$ \  $$$$/         |___  $$ \  $$$$/   |___  $$
// | $$\  $ | $$ /$$__  $$  | $$ /$$| $$      | $$  >$$  $$        /$$  \ $$  >$$  $$  /$$  \ $$
// | $$ \/  | $$|  $$$$$$$  |  $$$$/| $$      | $$ /$$/\  $$      |  $$$$$$/ /$$/\  $$|  $$$$$$/
// |__/     |__/ \_______/   \___/  |__/      |__/|__/  \__/       \______/ |__/  \__/ \______/ 
//
//===================================================================================================================================================================================//

//?
//? Created by LunaticWyrm467 and others.
//? 
//? All code is licensed under the MIT license.
//? Feel free to reproduce, modify, and do whatever.
//?

//!
//! A private submodule for the matrix module that contains all of the implementations
//! for any of the non-shared behaviours of the 3x3 Matrix.
//!

use core::fmt::{ self, Display };
use core::ops::{ Mul, Neg, MulAssign, Index, IndexMut };

#[cfg(feature = "alloc")]
use alloc::format;

use crate::{ prelude::*, scalar };
use crate::vector::SignedAxis3;
use crate::vector::{ Vec3, Axis3::{ self, X, Y, Z, NoAxis } };


/*
 * Euler Order
 *      Enums
 */


/// Describes a specific Euler order.
/// Angles are composed in the order specified, whilst being decomposed in the reverse order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EulerOrder {
    OrderXYZ,
    OrderXZY,
    OrderYXZ,
    OrderYZX,
    OrderZXY,
    OrderZYX
}


/*
 * Looking At
 *      Enums
 */


/// Describes a specific coordinate system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookingAtMode {
    
    /// The default coordinate system is used, with -Z considered the front axis and +X being
    /// considered the right axis.
    Default,
    
    /// The coordinate system where +Z is considered the front axis and -X is considered the right
    /// axis.
    ModelFront
}


/*
 * Matrix 3x3
 *      Indexing
 */


impl <T: Scalar> Index<Axis3> for Mat3<T> {
    type Output = Vec3<T>;
    fn index(&self, index: Axis3) -> &Self::Output {
        match index {
            X      => &self.x,
            Y      => &self.y,
            Z      => &self.z,
            NoAxis => panic!("`NoAxis` is not a valid index!")
        }
    }
}

impl <T: Scalar> IndexMut<Axis3> for Mat3<T> {
    fn index_mut(&mut self, index: Axis3) -> &mut Self::Output {
        match index {
            X      => &mut self.x,
            Y      => &mut self.y,
            Z      => &mut self.z,
            NoAxis => panic!("`NoAxis` is not a valid index!")
        }
    }
}


/*
 * Matrix 3x3
 *      Implementation
 */


/// A 3x3 row-majory matrix which can be used to describe 3D transformations in computer graphics.
/// More specifically, it can be used to descrive the shear (local position), rotation, and scale of an object in
/// 3D space.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
#[repr(C)]
pub struct Mat3<T: Scalar> {
    pub x: Vec3<T>,
    pub y: Vec3<T>,
    pub z: Vec3<T>
}

impl <T: Scalar> Mat3<T> {
    
    /// An Identity Matrix is the default matrix which has no rotation, shear or any modifications
    /// to its scale.
    pub const IDENTITY: Mat3<T> = Mat3 {
        x: Vec3::RIGHT,
        y: Vec3::UP,
        z: Vec3::BACK
    };

    /// A Matrix with all of its fields set to zero.
    pub const ZERO: Mat3<T> = Mat3 {
        x: Vec3::ZERO,
        y: Vec3::ZERO,
        z: Vec3::ZERO
    };

    /// Creates a new `Mat3` from base components.
    pub fn new(
        xx: T, xy: T, xz: T,
        yx: T, yy: T, yz: T,
        zx: T, zy: T, zz: T
        ) -> Self {
        
        Mat3 {
            x: Vec3(xx, xy, xz),
            y: Vec3(yx, yy, yz),
            z: Vec3(zx, zy, zz)
        }
    }
    
    /// Converts a `Mat3` to a `Mat3` of a different type.
    /// Returns `None` if the cast was unsuccessful.
    pub fn cast<U: Scalar>(&self) -> Option<Mat3<U>> {
        match (self[X].cast(), self[Y].cast(), self[Z].cast()) {
            (Some(x), Some(y), Some(z)) => Some(Mat3 { x, y, z }),
            _                           => None
        }
    }

    /// Gets a column.
    /// Useful since matrix indexing is row-major.
    pub fn get_column(&self, axis: Axis3) -> Vec3<T> {
        match axis {
            X      => Vec3(self[X][X], self[Y][X], self[Z][X]),
            Y      => Vec3(self[X][Y], self[Y][Y], self[Z][Y]),
            Z      => Vec3(self[X][Z], self[Y][Z], self[Z][Z]),
            NoAxis => panic!("`NoAxis` is not a valid index!")
        }
    }
    
    /// Sets a column.
    /// Useful since matrix indexing is row-major.
    pub fn set_column(&mut self, axis: Axis3, column: Vec3<T>) -> () {
        match axis {
            X => {
                self[X][X] = column[X]; self[Y][X] = column[Y]; self[Z][X] = column[Z];
            }
            Y => {
                self[X][Y] = column[X]; self[Y][Y] = column[Y]; self[Z][Y] = column[Z];
            }
            Z => {
                self[X][Z] = column[X]; self[Y][Z] = column[Y]; self[Z][Z] = column[Z];
            }
            NoAxis => panic!("`NoAxis` is not a valid index!")
        }
    }
    
    /// Constructs a new `Mat3` which represents scale, with no rotation or shear applied.
    pub fn from_scale(scale: Vec3<T>) -> Self {
        let t0: T = T::zero();
        Mat3 {
            x: Vec3(scale[X], t0, t0),
            y: Vec3(t0, scale[Y], t0),
            z: Vec3(t0, t0, scale[Z])
        }
    }
    
    /// Outputs a modified version of this matrix that was scaled up or down to `scale`.
    pub fn scaled(&self, scale: Vec3<T>) -> Mat3<T> {
        Mat3 {
            x: self[X] * scale[X],
            y: self[Y] * scale[Y],
            z: self[Z] * scale[Z]
        }
    }
    
    /// Multiplies all of the values of the matrix by a given scalar value.
    pub fn scalar_mult(&self, other: T) -> Mat3<T> {
        Mat3 {
            x: self[X] * other,
            y: self[Y] * other,
            z: self[Z] * other
        }
    }
    
    /// Returns the transposed dot product of this matrix and a vector with the axis specified.
    pub fn tdot_axis(&self, vec: Vec3<T>, axis: Axis3) -> T {
        match axis {
            X      => self[X][X] * vec[X] + self[Y][X] * vec[Y] + self[Z][X] * vec[Z],
            Y      => self[X][Y] * vec[X] + self[Y][Y] * vec[Y] + self[Z][Y] * vec[Z],
            Z      => self[X][Z] * vec[X] + self[Y][Z] * vec[Y] + self[Z][Z] * vec[Z],
            NoAxis => panic!("`NoAxis` is not a valid index!")
        }
    }
    
    /// Rotates and scales a given vector according to this Mat3.
    pub fn xform(&self, vec: Vec3<T>) -> Vec3<T> {
        Vec3(
			self[X].dot(vec),
			self[Y].dot(vec),
			self[Z].dot(vec)
        )
    }
    
    /// Reverses a rotation transformation that this `Mat3` has applied on the given vector.
    /// TODO: Utilize gaussian elimintation to reverse scaling as well, as scaling is not yet
    /// supported by this function.
    pub fn inv_xform(&self, vec: Vec3<T>) -> Vec3<T> {
        Vec3(
			(self[X][X] * vec[X]) + (self[Y][X] * vec[Y]) + (self[Z][X] * vec[Z]),
			(self[X][Y] * vec[X]) + (self[Y][Y] * vec[Y]) + (self[Z][Y] * vec[Z]),
			(self[X][Z] * vec[X]) + (self[Y][Z] * vec[Y]) + (self[Z][Z] * vec[Z])
        )
    }

    /// Dot multiplies two matrices together, which transforms the `other` matrix by this matrix.
    /// Useful if you want to center the origin of one object on another.
    pub fn dot(&self, other: &Mat3<T>) -> Mat3<T> {
        Mat3 {
			x: Vec3(other.tdot_axis(self[X], X), other.tdot_axis(self[X], Y), other.tdot_axis(self[X], Z)),
			y: Vec3(other.tdot_axis(self[Y], X), other.tdot_axis(self[Y], Y), other.tdot_axis(self[Y], Z)),
			z: Vec3(other.tdot_axis(self[Z], X), other.tdot_axis(self[Z], Y), other.tdot_axis(self[Z], Z))
        }
    }

    /// Returns a transposed version of this matrix.
    pub fn transposed(&self) -> Mat3<T> {
        Mat3 {
            x: self.get_column(X),
            y: self.get_column(Y),
            z: self.get_column(Z)
        }
    }

    /// Determines if this matrix is symmetrical.
    /// Symmetrical matrices have interesting properties such as being equal to their transpose.
    pub fn is_symmetrical(&self) -> bool {
        self == &self.transposed()
    }
}

impl <T: SignedScalar> Mat3<T> {
    
    /// Any Mat3 multiplied by this will have its X components negated.
    /// The matrix itself looks like this:
    /// ```text
    /// Mat3(
    ///     -1, 0, 0,
    ///      0, 1, 0,
    ///      0, 0, 1
    /// )
    /// ```
    pub const FLIP_X: Mat3<T> = Mat3 {
        x: Vec3::LEFT,
        y: Vec3::UP,
        z: Vec3::BACK
    };
    
    /// Any Mat3 multiplied by this will have its Y components negated.
    /// The matrix itself looks like this:
    /// ```text
    /// Mat3(
    ///     1,  0, 0,
    ///     0, -1, 0,
    ///     0,  0, 1
    /// )
    /// ```
    pub const FLIP_Y: Mat3<T> = Mat3 {
        x: Vec3::RIGHT,
        y: Vec3::DOWN,
        z: Vec3::BACK
    };
    
    /// Any Mat3 multiplied by this will have its Z components negated.
    /// The matrix itself looks like this:
    /// ```text
    /// Mat3(
    ///     1, 0,  0,
    ///     0, 1,  0,
    ///     0, 0, -1
    /// )
    /// ```
    pub const FLIP_Z: Mat3<T> = Mat3 {
        x: Vec3::RIGHT,
        y: Vec3::UP,
        z: Vec3::FORWARD
    };

    /// Computes the Determinant, which can be used for a number of reasons in higher maths:
    /// - If the determinant is `0`, then the matrix cannot be inverted.
    /// - If the determinant is negative, then the matrix represents a negative scale.
    pub fn determinant(&self) -> T {
        self[X][X] * (self[Y][Y] * self[Z][Z] - self[Z][Y] * self[Y][Z]) -
        self[Y][X] * (self[X][Y] * self[Z][Z] - self[Z][Y] * self[X][Z]) +
        self[Z][X] * (self[X][Y] * self[Y][Z] - self[Y][Y] * self[X][Z])
    }

    /// Returns the cofactor of the matrix.
    pub fn cofactor(&self, row1: Axis3, col1: Axis3, row2: Axis3, col2: Axis3) -> T {
        self[row1][col1] * self[row2][col2] - self[row1][col2] * self[row2][col1]
    }
}

impl <T: FloatScalar> Mat3<T> {
    
    /// Creates a Matrix from an angle and axis of rotation.
    pub fn from_angle(angle: T, axis: SignedAxis3) -> Self {
        Self::from_angle_free(angle, axis.to_vec3())
    }

    /// Creates a Matrix from an angle and axis of rotation.
    /// Unlike `from_angle`, this acceps a custom Vec3 for the axis.
    pub fn from_angle_free(angle: T, axis: Vec3<T>) -> Self {
        
        /*
         * Code Ported from Godot: https://github.com/godotengine/godot/blob/master/core/math/basis.cpp#L849
         * Mathematical Formula found here: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_angle
         */

        let     axis_sq: Vec3<T> = axis.pow2();
        let     cosine:  T       = angle.cos();
        let mut matrix:  Mat3<T> = Mat3::ZERO;
        
        matrix[X][X] = axis_sq[X] + cosine * (T::one() - axis_sq[X]);
        matrix[Y][Y] = axis_sq[Y] + cosine * (T::one() - axis_sq[Y]);
        matrix[Z][Z] = axis_sq[Z] + cosine * (T::one() - axis_sq[Z]);

        let sine: T = angle.sin();
        let t:    T = T::one() - cosine;

        let mut xyzt: T = axis[X] * axis[Y] * t;
        let mut zyxs: T = axis[Z] * sine;
        matrix[X][Y] = xyzt - zyxs;
        matrix[Y][X] = xyzt + zyxs;

        xyzt = axis[X] * axis[Z] * t;
        zyxs = axis[Y] * sine;
        matrix[X][Z] = xyzt + zyxs;
        matrix[Z][X] = xyzt - zyxs;

        xyzt = axis[Y] * axis[Z] * t;
        zyxs = axis[X] * sine;
        matrix[Y][Z] = xyzt - zyxs;
        matrix[Z][Y] = xyzt + zyxs;

        matrix
    }

    /// Creates a Matrix from an Euler angle (in radians) and specified order.
    /// More specifically:
    /// - X should be pitch.
    /// - Y should be yaw.
    /// - Z should be roll.
    pub fn from_euler_angle(euler: Vec3<T>, order: EulerOrder) -> Self {
        
        /*
         * Code Ported from Godot: https://github.com/godotengine/godot/blob/master/core/math/basis.cpp#L653
         */

        let t0: T = T::zero();
        let t1: T = T::one();

        let (mut c, mut s): (T, T);
        
        c = euler[X].cos();
        s = euler[X].sin();
        let xmat: Mat3<T> = Mat3::new(t1, t0, t0, t0, c, -s, t0, s, c);

        c = euler[Y].cos();
        s = euler[Y].sin();
        let ymat: Mat3<T> = Mat3::new(c, t0, s, t0, t1, t0, -s, t0, c);

        c = euler[Z].cos();
        s = euler[Z].sin();
        let zmat: Mat3<T> = Mat3::new(c, -s, t0, s, c, t0, t0, t0, t1);

        match order {
            EulerOrder::OrderXYZ => xmat * (ymat * zmat),
            EulerOrder::OrderXZY => xmat * zmat * ymat,
            EulerOrder::OrderYXZ => ymat * xmat * zmat,
            EulerOrder::OrderYZX => ymat * zmat * xmat,
            EulerOrder::OrderZXY => zmat * xmat * ymat,
            EulerOrder::OrderZYX => zmat * ymat * xmat
        }
    }

    /// Creates a new `Mat3` with a rotation such that its forward axis will target a specified
    /// position.
    /// By default, the -Z axis is treated as forward (implies +X is right).
    /// This can be changed to utilize the model front (which implies +X is left).
    /// # Note
    /// The up axis (+Y) points as close to the 'up' vector as possible while staying perpendicular
    /// to the forward axis.
    /// The returned matrix is orthonormalized. See `orthonormalized()`.
    pub fn looking_at(target: Vec3<T>, up: Vec3<T>, mode: LookingAtMode) -> Self {
        let mut v_z: Vec3<T> = target.normalized();
        if mode == LookingAtMode::ModelFront {
            v_z = -v_z;
        }
        
        let mut v_x: Vec3<T> = up.cross(v_z);
                v_x          = v_x.normalized();
        
        let v_y: Vec3<T> = v_z.cross(v_x);

        Mat3 {
            x: v_x,
            y: v_y,
            z: v_z
        }
    }
    
    /// Gets the current Euler angle described by this matrix, in radians.
    /// - The X component is the pitch.
    /// - The Y component is the yaw.
    /// - The Z component is the roll.
    /// Euler angles may be more intuitive but for 3D math it is preferred to use Quaternions due
    /// to a phenomenon known as gimbal lock.
    pub fn to_euler(&self, order: EulerOrder) -> Vec3<T> {
        match order {
            EulerOrder::OrderXYZ => to_euler_xyz(self),
            EulerOrder::OrderXZY => to_euler_xzy(self),
            EulerOrder::OrderYXZ => to_euler_yxz(self),
            EulerOrder::OrderYZX => to_euler_yzx(self),
            EulerOrder::OrderZXY => to_euler_zxy(self),
            EulerOrder::OrderZYX => to_euler_zyx(self)
        }
    }

    /// Gets the absolute value of this matrix's scale.
    /// If the determinant is negative, then this scale is as well.
    pub fn get_unsigned_scale(&self) -> Vec3<T> {
        Vec3(
			Vec3(self[X][X], self[Y][X], self[Z][X]).length(),
			Vec3(self[X][Y], self[Y][Y], self[Z][Y]).length(),
			Vec3(self[X][Z], self[Y][Z], self[Z][Z]).length()
        )
    }

    /// Gets this matrix's scale.
    pub fn get_scale(&self) -> Vec3<T> {
        let det_sign: T = self.determinant().signum();
	    self.get_unsigned_scale() * det_sign
    }

    /// Returns the inverse of this matrix.
    pub fn inverse(&self) -> Self {
        let co: [T; 3] = [self.cofactor(Y, Y, Z, Z), self.cofactor(Y, Z, Z, X), self.cofactor(Y, X, Z, Y)];
        
        let det: T = self[X][X] * co[0] +
                     self[X][Y] * co[1] +
                     self[X][Z] * co[2];

        let s: T = T::one() / det;

        Mat3 {
            x: Vec3(co[0] * s, self.cofactor(X, Z, Z, Y) * s, self.cofactor(X, Y, Y, Z) * s),
            y: Vec3(co[1] * s, self.cofactor(X, X, Z, Z) * s, self.cofactor(X, Z, Y, X) * s),
            z: Vec3(co[2] * s, self.cofactor(X, Y, Z, X) * s, self.cofactor(X, X, Y, Y) * s)
        }
    }

    /// Rotates this matrix by a given angle in radians across a given axis.
    /// Positive values rotate this basis clockwise around the axis, while negative values rotate it counterclockwise.
    pub fn rotated(&self, angle: T, axis: SignedAxis3) -> Mat3<T> {
        self.rotated_free(angle, axis.to_vec3())
    }
    
    /// Rotates this matrix by a given angle in radians across a given axis.
    /// Positive values rotate this basis clockwise around the axis, while negative values rotate it counterclockwise.
    /// Compared to `rotated`, this function gives you the freedom to specify a custom axis vector.
    pub fn rotated_free(&self, angle: T, axis: Vec3<T>) -> Mat3<T> {
        Mat3::from_angle_free(angle, axis) * self
    }

    /// Linearly interpolates from one matrix to another.
    pub fn lerp(&self, other: &Mat3<T>, t: T) -> Mat3<T> {
        Mat3 {
            x: self[X].lerp(other[X], t),
            y: self[Y].lerp(other[Y], t),
            z: self[Z].lerp(other[Z], t)
        }
    }
    
    /// Spherically interpolates between two matrices.
    pub fn slerp(&self, _other: &Mat3<T>, _t: T) -> Mat3<T> {
        todo!()

        /*
        // Consider scale - Use Quaternions
        let from: Quaternion = Quaternion::from_mat3(this);
        let to:   Quaternion = Quaternion::from_mat3(other);

        let o: Mat3<T> = from.slerp(to, t);
        o[X] *= self[X].length().lerp(other[X].length(), t);
        o[Y] *= self[Y].length().lerp(other[Y].length(), t);
        o[Z] *= self[Z].length().lerp(other[Z].length(), t);

        o*/
    }


    /// Returns an orthonormalized Matrix where:
    /// - The matrix is orthogonal: The axes are perpendicular to each other.
    /// - The matrix is normalized: The axes all have a length of one.
    /// It is useful to call this method to avoid rounding errors on a rotating matrix.
    pub fn orthonormalized(&self) -> Mat3<T> {
        
        /*
         * Gram-Schmidt Process
         */

        let mut x: Vec3<T> = self.get_column(X);
        let mut y: Vec3<T> = self.get_column(Y);
        let mut z: Vec3<T> = self.get_column(Z);

        x = x.normalized();
        y = y - x * (x.dot(y));
        y = y.normalized();
        z = z - x * (x.dot(z)) - y * (y.dot(z));
        z = z.normalized();

        Mat3 { x, y, z }
    }

    /// Determines if this Matrix is conformal, which essentially means that:
    /// - The Matrix is orthogonal: All axes are perpendicular to each other.
    /// - The Matrix is uniform: Each axis share the same length.
    /// If a Matrix is conformal, it features no distortions.
    pub fn is_conformal(&self) -> bool {
        let x: Vec3<T>  = self.get_column(X);
	    let y: Vec3<T>  = self.get_column(Y);
	    let z: Vec3<T>  = self.get_column(Z);
	    let x_len_sq: T = x.magnitude_squared();
	    
        x_len_sq.approx_eq(y.magnitude_squared()) && x_len_sq.approx_eq(z.magnitude_squared()) &&
        x.dot(y).approx_zero() && x.dot(z).approx_zero() && y.dot(z).approx_zero()
    }

    /// Determines if this Matrix is approximately equal to another Matrix by calling `approx_eq`
    /// between all members.
    pub fn approx_eq(&self, other: &Mat3<T>) -> bool {
        self[X][X].approx_eq(other[X][X]) && self[X][Y].approx_eq(other[X][Y]) && self[X][Z].approx_eq(other[X][Z]) &&  
        self[Y][X].approx_eq(other[Y][X]) && self[Y][Y].approx_eq(other[Y][Y]) && self[Y][Z].approx_eq(other[Y][Z]) &&  
        self[Z][X].approx_eq(other[Z][X]) && self[Z][Y].approx_eq(other[Z][Y]) && self[Z][Z].approx_eq(other[Z][Z])  
    }

    /// Returns whether this Matrix is finite.
    pub fn is_finite(&self) -> bool {
        self[X].is_finite() && self[Y].is_finite() && self[Z].is_finite()
    }
}


/*
 * Matrix to Euler
 *      Conversions
 *
 *  This code is ported from Godot and is based on the formulas found in:
 *  https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
 */


fn to_euler_xyz<T: FloatScalar>(mat: &Mat3<T>) -> Vec3<T> {
    
    // Euler angles in XYZ convention.
    // See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    //
    // rot =  cy*cz          -cy*sz           sy
    //        cz*sx*sy+cx*sz  cx*cz-sx*sy*sz -cy*sx
    //       -cx*cz*sy+sx*sz  cz*sx+cx*sy*sz  cx*cy
    
    let t0:      T = T::zero();
    let t1:      T = T::one();
    let t2:      T = T::from(2).unwrap();
    let epsilon: T = T::from(scalar::EPSILON).unwrap();

    let mut euler: Vec3<T> = t0.vec3();
    let     sy:    T       = mat[X][Z];
    
    if sy < (t1 - epsilon) {
        if sy > - (t1 - epsilon) {
            
            // If this is a pure Y rotation, return the simplest form for human readability.
            if mat[Y][X] == t0 && mat[X][Y] == t0 && mat[Y][Z] == t0 && mat[Z][Y] == t0 && mat[Y][Y] == t1 {
                euler[X] = t0;
                euler[Y] = mat[X][Z].atan2(mat[X][X]);
                euler[Z] = t0;
            } else {
                euler[X] = (-mat[Y][Z]).atan2(mat[Z][Z]);
                euler[Y] = sy.asin();
                euler[Z] = (-mat[X][Y]).atan2(mat[X][X]);
            }
        } else {
            euler[X] = mat[Z][Y].atan2(mat[Y][Y]);
            euler[Y] = -T::PI() / t2;
            euler[Z] = t0;
        }
    } else {
        euler[X] = mat[Z][Y].atan2(mat[Y][Y]);
        euler[Y] = T::PI() / t2;
        euler[Z] = t0;
    }
    
    euler
}

fn to_euler_xzy<T: FloatScalar>(mat: &Mat3<T>) -> Vec3<T> {
    
    // Euler angles in XZY convention.
    // See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    //
    // rot =  cz*cy             -sz             cz*sy
    //        sx*sy+cx*cy*sz    cx*cz           cx*sz*sy-cy*sx
    //        cy*sx*sz          cz*sx           cx*cy+sx*sz*sy
    
    let t0:      T = T::zero();
    let t1:      T = T::one();
    let t2:      T = T::from(2).unwrap();
    let epsilon: T = T::from(scalar::EPSILON).unwrap();

    let mut euler: Vec3<T> = t0.vec3();
    let     sz:    T       = mat[X][Y];

    if sz < (t1 - epsilon) {
        if sz > -(t1 - epsilon) {
            euler[X] = mat[Z][Y].atan2(mat[Y][Y]);
            euler[Y] = mat[X][Z].atan2(mat[X][X]);
            euler[Z] = (-sz).asin();
        } else {
            euler[X] = -mat[Y][Z].atan2(mat[Z][Z]);
            euler[Y] = t0;
            euler[Z] = T::PI() / t2;
        }
    } else {
        euler[X] = -mat[Y][Z].atan2(mat[Z][Z]);
        euler[Y] = t0;
        euler[Z] = -T::PI() / t2;
    }
    
    euler
}

fn to_euler_yxz<T: FloatScalar>(mat: &Mat3<T>) -> Vec3<T> {
    
    // Euler angles in YXZ convention.
    // See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    //
    // rot =  cy*cz+sy*sx*sz    cz*sy*sx-cy*sz        cx*sy
    //        cx*sz             cx*cz                 -sx
    //        cy*sx*sz-cz*sy    cy*cz*sx+sy*sz        cy*cx
    
    // TODO:
    // Based on the implementation here:
    // https://github.com/godotengine/godot/blob/master/core/math/basis.cpp#L457
    // Figure out the discrepancy with conversions.

    let t0:      T = T::zero();
    let t1:      T = T::one();
    let t05:     T = T::from(0.5).unwrap();
    let epsilon: T = T::from(scalar::EPSILON).unwrap();
    
    let mut euler: Vec3<T> = t0.vec3();
    let     m12:   T       = mat[Y][Z];

    if m12 < (t1 - epsilon) {
        if m12 > -(t1 - epsilon) {
            
            // If this is a pure X rotation, return the simplest form for human readability.
            if mat[Y][X] == t0 && mat[X][Y] == t0 && mat[X][Z] == t0 && mat[Z][X] == t0 && mat[X][X] == t1 {
                euler[X] = (-m12).atan2(mat[Y][Y]);
                euler[Y] = t0;
                euler[Z] = t0;
            } else {
                euler[X] = (-m12).asin();
                euler[Y] = mat[X][Z].atan2(mat[Z][Z]);
                euler[Z] = mat[Y][X].atan2(mat[Y][Y]);
            }
        } else {
            euler[X] = T::PI() * t05;
            euler[Y] = mat[X][Y].atan2(mat[X][X]);
            euler[Z] = t0;
        }
    } else {
        euler[X] = -T::PI() * t05;
        euler[Y] = -mat[X][Y].atan2(mat[X][X]);
        euler[Z] = t0;
    }

    euler
}

fn to_euler_yzx<T: FloatScalar>(mat: &Mat3<T>) -> Vec3<T> {
    
    // Euler angles in YZX convention.
    // See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    //
    // rot =  cy*cz             sy*sx-cy*cx*sz     cx*sy+cy*sz*sx
    //        sz                cz*cx              -cz*sx
    //        -cz*sy            cy*sx+cx*sy*sz     cy*cx-sy*sz*sx

    let t0:      T = T::zero();
    let t1:      T = T::one();
    let t2:      T = T::from(2).unwrap();
    let epsilon: T = T::from(scalar::EPSILON).unwrap();
    
    let mut euler: Vec3<T> = t0.vec3();
    let     sz:    T       = mat[Y][X];
    
    if sz < (t1 - epsilon) {
        if sz > -(t1 - epsilon) {
            euler[X] = (-mat[Y][Z]).atan2(mat[Y][Y]);
            euler[Y] = (-mat[Z][X]).atan2(mat[X][X]);
            euler[Z] = sz.asin();
        } else {
            euler[X] = mat[Z][Y].atan2(mat[Z][Z]);
            euler[Y] = t0;
            euler[Z] = -T::PI() / t2;
        }
    } else {
        euler[X] = mat[Z][Y].atan2(mat[Z][Z]);
        euler[Y] = t0;
        euler[Z] = T::PI() / t2;
    }
    
    euler
}

fn to_euler_zxy<T: FloatScalar>(mat: &Mat3<T>) -> Vec3<T> {
    
    // Euler angles in ZXY convention.
    // See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    //
    // rot =  cz*cy-sz*sx*sy    -cx*sz                cz*sy+cy*sz*sx
    //        cy*sz+cz*sx*sy    cz*cx                 sz*sy-cz*cy*sx
    //        -cx*sy            sx                    cx*cy
    
    let t0:      T = T::zero();
    let t1:      T = T::one();
    let t2:      T = T::from(2).unwrap();
    let epsilon: T = T::from(scalar::EPSILON).unwrap();
    
    let mut euler: Vec3<T> = t0.vec3();
    let     sx:    T       = mat[Z][Y];
    
    if sx < (t1 - epsilon) {
        if sx > -(t1 - epsilon) {
            euler[X] = sx.asin();
            euler[Y] = (-mat[Z][X]).atan2(mat[Z][Z]);
            euler[Z] = (-mat[X][Y]).atan2(mat[Y][Y]);
        } else {
            euler[X] = -T::PI() / t2;
            euler[Y] = mat[X][Z].atan2(mat[X][X]);
            euler[Z] = t0;
        }
    } else {
        euler[X] = T::PI() / t2;
        euler[Y] = mat[X][Z].atan2(mat[X][X]);
        euler[Z] = t0;
    }
    
    euler
}

fn to_euler_zyx<T: FloatScalar>(mat: &Mat3<T>) -> Vec3<T> {
    
    // Euler angles in ZYX convention.
    // See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    //
    // rot =  cz*cy             cz*sy*sx-cx*sz        sz*sx+cz*cx*cy
    //        cy*sz             cz*cx+sz*sy*sx        cx*sz*sy-cz*sx
    //        -sy               cy*sx                 cy*cx
    
    let t0:      T = T::zero();
    let t1:      T = T::one();
    let t2:      T = T::from(2).unwrap();
    let epsilon: T = T::from(scalar::EPSILON).unwrap();
    
    let mut euler: Vec3<T> = t0.vec3();
    let     sy:    T       = mat[Z][X];
    
    if sy < (t1 - epsilon) {
        if sy > -(t1 - epsilon) {
            euler[X] = mat[Z][Y].atan2(mat[Z][Z]);
            euler[Y] = (-sy).asin();
            euler[Z] = mat[Y][X].atan2(mat[X][X]);
        } else {
            euler[X] = t0;
            euler[Y] = T::PI() / t2;
            euler[Z] = -mat[X][Y].atan2(mat[Y][Y]);
        }
    } else {
        euler[X] = t0;
        euler[Y] = -T::PI() / t2;
        euler[Z] = -mat[X][Y].atan2(mat[Y][Y]);
    }
    
    euler
}


/*
    Global Operations
        Base Arithmetic
*/


impl <T: Scalar> Mul for Mat3<T> {
    type Output = Mat3<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.dot(&rhs)
    }
}

impl <T: FloatScalar> Neg for Mat3<T> {
    type Output = Mat3<T>;
    fn neg(self) -> Self::Output {
        self.inverse()
    }
}


/*
    Global Operations
        Reference Arithmetic
*/


impl <'a, T: Scalar> Mul<&'a Self> for &'a Mat3<T> {
    type Output = Mat3<T>;
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self.dot(rhs)
    }
}


/*
    Global Operations
        Reference vs Base Arithmetic
*/


impl <T: Scalar> Mul<&Self> for Mat3<T> {
    type Output = Mat3<T>;
    fn mul(self, rhs: &Self) -> Self::Output {
        self.dot(&rhs)
    }
}


/*
    Global Operations
        Base vs Vector Arithmetic
*/


impl <T: Scalar> Mul<Vec3<T>> for Mat3<T> {
    type Output = Vec3<T>;
    fn mul(self, rhs: Vec3<T>) -> Self::Output {
        self.xform(rhs)
    }
}


/*
    Global Operations
        Reference vs Vector Arithmetic
*/


impl <T: Scalar> Mul<Vec3<T>> for &Mat3<T> {
    type Output = Vec3<T>;
    fn mul(self, rhs: Vec3<T>) -> Self::Output {
        self.xform(rhs)
    }
}


/*
    Global Operations
        Base vs Scalar Arithmetic
*/


impl <T: Scalar> Mul<T> for Mat3<T> {
    type Output = Mat3<T>;
    fn mul(self, rhs: T) -> Self::Output {
        self.scalar_mult(rhs)
    }
}


/*
    Global Operations
        Reference vs Scalar Arithmetic
*/


impl <T: Scalar> Mul<T> for &Mat3<T> {
    type Output = Mat3<T>;
    fn mul(self, rhs: T) -> Self::Output {
        self.scalar_mult(rhs)
    }
}


/*
    Global Operations
        Assignment & Arithmetic
*/


impl <T: Scalar> MulAssign for Mat3<T> {
    fn mul_assign(&mut self, rhs: Self) -> () {
        *self = self.dot(&rhs)
    }
}


/*
    Global
        Behaviours
*/


impl <T: Scalar> Default for Mat3<T> {
    fn default() -> Self {
        Mat3::IDENTITY
    }
}

#[cfg(feature = "alloc")]
impl <T: Scalar> Display for Mat3<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> fmt::Result {
        let mut highest_len: usize = 0;
        
        for axis_row in [X, Y, Z] {
            for axis_col in [X, Y, Z] {
                highest_len = Scalar::max(highest_len, format!("{}", self[axis_row][axis_col]).len());
            }
        }

        write!(f, "Mat3(")?;
        for axis_row in [X, Y, Z] {
            write!(f, "\n\t")?;
            for axis_col in [X, Y, Z] {
                let number_str: &str = &format!("{}", self[axis_row][axis_col]);
                let padding:    &str = &" ".repeat(highest_len - number_str.len());

                write!(f, "{number_str},{padding} ")?;
            }
        }
        write!(f, "\n)")
    }
}
