//===================================================================================================================================================================================//
//
//  /$$$$$$$                        /$$    
// | $$__  $$                      | $$    
// | $$  \ $$  /$$$$$$   /$$$$$$  /$$$$$$  
// | $$$$$$$/ /$$__  $$ /$$__  $$|_  $$_/  
// | $$__  $$| $$  \ $$| $$  \ $$  | $$    
// | $$  \ $$| $$  | $$| $$  | $$  | $$ /$$
// | $$  | $$|  $$$$$$/|  $$$$$$/  |  $$$$/
// |__/  |__/ \______/  \______/    \___/
//
//===================================================================================================================================================================================//

//?
//? Created by LunaticWyrm467 and others.
//? 
//? All code is licensed under the MIT license.
//? Feel free to reproduce, modify, and do whatever.
//?

//!
//! The root file of the library. Nothing special here.
//!

#![no_std]
#[cfg(feature = "alloc")]
extern crate alloc;

/// The scalar module contains traits and functions that are added to basic primitive types.
/// While you can import from this module, scalar traits will automatically be imported from the `prelude` module.
pub mod scalar;
pub mod vector;
pub mod mat;
pub mod rect;

/// Generally speaking, you'll want to use the prelude module to get all of the traits and functions you'll need.
/// This does not include any of the individual types. If you want those, import from the `vector` and/or `rect` modules.
pub mod prelude {
    pub use crate::scalar::{ Scalar, IntScalar, SignedScalar, FloatScalar };
    pub use crate::vector::{ Vector, IntVector, SignedVector, FloatVector, Vectorized2D, Vectorized3D };
    pub use crate::rect::{ Rect, SignedRect, FloatRect };
}


/*
    Vectorized
        Trait
*/

/// We expose this private trait to the whole library so our library's generic traits can use it,
/// but we don't allow it to be used outside of the library since there are more readable options
/// such as `Vectorized2D`, `Vectorized3D`, and `Vectorized4D` whose functions are more
/// approachable.
mod vectorized {
    use crate::vector::VectorAbstract;
    use crate::scalar::Scalar;

    pub trait Vectorized<T: Scalar + Vectorized<T, V>, V: VectorAbstract<T, V>>: Clone + Copy {
        
        /// Returns a scalar if the type is a scalar, otherwise returns None.
        fn attempt_get_scalar(self) -> Option<T>;

        /// Converts a type or tuple of types to a suitable Vector representation.
        fn dvec(self) -> V;
    }
}
