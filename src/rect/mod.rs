//===================================================================================================================================================================================//
//
//  /$$$$$$$                        /$$    
// | $$__  $$                      | $$    
// | $$  \ $$  /$$$$$$   /$$$$$$$ /$$$$$$  
// | $$$$$$$/ /$$__  $$ /$$_____/|_  $$_/  
// | $$__  $$| $$$$$$$$| $$        | $$    
// | $$  \ $$| $$_____/| $$        | $$ /$$
// | $$  | $$|  $$$$$$$|  $$$$$$$  |  $$$$/
// |__/  |__/ \_______/ \_______/   \___/
//
//===================================================================================================================================================================================//

//?
//? Created by LunaticWyrm467 and others.
//? 
//? All code is licensed under the MIT license.
//? Feel free to reproduce, modify, and do whatever.
//?

//!
//! The rect module contains the definitions of the various bounding rect types, such as `Rect2` and `Rect3`.
//! Also contains global traits and functions that are shared between all bounding rect types.
//!

mod r2d;

use crate::scalar::{ Scalar, FloatScalar, SignedScalar };
use crate::vector::{ VectorAbstract, Vector, SignedVector, FloatVector };

pub use r2d::{ Side2, Rect2 };


/*
    Rect
        Trait
*/


pub trait RectAbstract<T: Scalar, V: VectorAbstract<T, V>, R: RectAbstract<T, V, R>>
where Self:
    Clone + PartialEq + PartialOrd + Default + std::fmt::Display + std::fmt::Debug
{}

pub trait Rect<T: Scalar, V: Vector<T, V, A>, R: Rect<T, V, R, A, S>, A, S>: RectAbstract<T, V, R> {

    //=====// Constructors //=====//
    /// Creates a new rectangle with the given position and size.
    /// This is the same thing as doing `Rect(position, size)` and is included for trait functionality.
    fn new(position: V, size: V) -> R;

    /// Constructs a new rectangle with all of its vectors set to zero.
    fn zeros_like() -> R {
        Self::new(V::zeros_like(), V::zeros_like())
    }

    /// Constructs a new rectangle with its size vector set to one and its position vector set to zero.
    fn unit() -> R {
        Self::unit_at(V::zeros_like())
    }

    /// Constructs a unit rectangle at a given position.
    fn unit_at(position: V) -> R {
        Self::new(position, V::ones_like())
    }

    /// Creates a rectangle of a given size located at 0,0.
    fn of_size(size: V) -> R {
        Self::new(V::zeros_like(), size)
    }

    /// Creates a new rectangle that encompasses all points in the provided vector.
    fn encompass_points(points: &Vec<V>) -> R;

    /// Creates a new rectangle that encompasses all rectangles in the provided vector.
    fn encompass_rects(rects: &Vec<R>) -> R {
        if rects.is_empty() {   // Special case for empty vector
            return Self::zeros_like();
        }
        
        let mut merge: R = rects[0].to_owned();
        for i in 1..rects.len() {
            merge = merge.merge(&rects[i]);
        }
        merge
    }


    //=====// Getters //=====//
    /// A simple identity function. Useful for trait implementations where trait bounds need to be kept.
    fn identity(&self) -> &R;

    /// Returns the rectangle's position.
    fn position(&self) -> &V;

    /// Returns a mutable reference to the rectangle's position.
    fn position_mut(&mut self) -> &mut V;

    /// Returns the rectangle's size.
    fn size(&self) -> &V;

    /// Returns a mutable reference to the rectangle's size.
    fn size_mut(&mut self) -> &mut V;


    //=====// Setters //=====//
    /// Sets the rectangle's position.
    fn set_position(&mut self, position: V);

    /// Sets the rectangle's size.
    fn set_size(&mut self, size: V);


    //=====// Geometry //=====//
    /// Computes the center of this rectangle.
    fn center(&self) -> V {
        self.position().to_owned() + (self.size().to_owned() / T::from(2.0).unwrap())
    }

    /// Gets the rectangle's 'end' corner, which is either its bottom-right corner if the size is positive, or its top-left corner if the size is negative.
    fn end(&self) -> V {
        self.position().to_owned() + self.size().to_owned()
    }

    /// Computes a selected vertex of this rectangle based on the given idx.
    /// # Panics
    /// This function will panic if the idx is out of bounds.
    fn vertex(&self, idx: usize) -> V;

    /// Gets the longest axis as an enum.
    fn longest_axis(&self) -> A;

    /// Gets the longest axis's length.
    fn longest_axis_length(&self) -> T {
        self.axis_length(self.longest_axis())
    }

    /// Gets the shortest axis as an enum.
    fn shortest_axis(&self) -> A;

    /// Gets the shortest axis's length.
    fn shortest_axis_length(&self) -> T {
        self.axis_length(self.shortest_axis())
    }

    /// Gets the size of an axis.
    fn axis_length(&self, axis: A) -> T;

    /// Returns whether this rectangle completely encompasses a given rectangle on all sides.
    fn encompasses(&self, other: &R) -> bool {
        self.position() < other.position() && other.end() <= self.end()
    }

    /// Returns whether this rectangle contains a given point in local coordinates.
    fn encompasses_point(&self, point: V) -> bool {
        self.position() <= &point && point < self.end()
    }

    /// Expands this rectangle up to the inclusion of the provided point in local coordinates and returns the result.
    fn expand_to_include(&self, point: V) -> R;

    /// Returns the n-dimensional measure of this rectangle.
    /// For a 2D rectangle, this is the area - and is the same thing as using the Rect2's `area()` function.
    /// For a 3D rectangle (or a box), this is the volume - and is the same thing as using the Rect3's `volume()` function.
    fn measure(&self) -> T {
        self.size().product()
    }

    /// Returns whether this rectangle has a valid n-dimensional measure which is positive and non-zero.
    fn has_valid_measure(&self) -> bool {
        self.size().to_owned() > V::zeros_like()
    }

    /// Expands and grows this rectangle by the given amount in all directions.
    fn grow(&self, amount: T) -> R {
        Self::new(self.position().to_owned() - amount, self.size().to_owned() + (amount * T::from(2).unwrap()))
    }

    /// Grows each side individually via a provided Rect.
    fn grow_by(&self, amount: &R) -> R {
        Self::new(self.position().to_owned() - amount.position().to_owned(), self.size().to_owned() + amount.size().to_owned() + amount.position().to_owned())
    }

    /// Grows a specific side of this rectangle by the given amount.
    fn grow_side(&self, side: S, amount: T) -> R;

    /// Merges two rectangles together and returns a rectangle that encompasses both of them.
    fn merge(&self, other: &R) -> R {
        let mut encompassing_rect: R = R::zeros_like();

        encompassing_rect.set_position(other.position().v_min(self.position()));
        encompassing_rect.set_size(other.end().v_max(&self.end()));

        // Make the position relative again in local space and return it.
        encompassing_rect.set_size(encompassing_rect.size().to_owned() - encompassing_rect.position().to_owned());
        encompassing_rect
    }

    /// Returns a rectangle describing the shape of an intersection between this rectangle and another.
    /// If there is no intersection, returns None.
    fn intersection(&self, other: &R) -> Option<R> {
        
        // First check if there even is an intersection to begin with.
        if !self.intersects(other, false) {
            return None;
        }

        // Compute the cross-section of the two rectangles' positions and sizes.
        let self_end:  V = self.end();
        let other_end: V = other.end();

        let new_position:  V = other.position().v_max(self.position());
        let cross_section: R = R::new(
            new_position.to_owned(),
            other_end.v_min(&self_end) - new_position
        );

        Some(cross_section)
    }

    /// Returns whether this rectangle intersects another.
    /// If `including_borders` is false, then a valid intersection will not include shapes that only touch by border.
    fn intersects(&self, other: &R, including_borders: bool) -> bool;
}

pub trait SignedRect<T: SignedScalar, V: SignedVector<T, V, A>, R: SignedRect<T, V, R, A, S>, A, S>: Rect<T, V, R, A, S> {

    /// Establishes a new Rectangle with the same size as the original, but with its position moved to the top-left corner
    /// in terms of local coordinates.<br>
    /// If the size is negative, the size will be made positive and the end point will be made the bottom-right corner.
    fn abs(&self) -> R {
        let end:      V = self.end();
        let top_left: V = self.position().v_min(&end);
        Self::new(top_left, self.size().abs())
    }
}

pub trait FloatRect<T: FloatScalar, V: FloatVector<T, V, A>, R: FloatRect<T, V, R, A, S>, A, S>: SignedRect<T, V, R, A, S> {

    /// Returns whether this rectangle is approximately equal to another.
    fn approx_eq(&self, other: &R) -> bool {
        self.position().approx_eq(other.position()) && self.size().approx_eq(other.size())
    }

    /// Returns whether this rectangle is finite in all capacity.
    fn is_finite(&self) -> bool {
        self.position().is_finite() && self.size().is_finite()
    }

    /// Returns whether this rectangle contains NaN values.
    fn is_nan(&self) -> bool {
        self.position().is_nan() || self.size().is_nan()
    }
}