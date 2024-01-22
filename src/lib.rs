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

/// The scalar module contains traits and functions that are added to basic primitive types.
/// While you can import from this module, scalar traits will automatically be imported from the `prelude` module.
pub mod scalar;
pub mod vector;
pub mod rect;

/// Generally speaking, you'll want to use the prelude module to get all of the traits and functions you'll need.
/// This does not include any of the individual types. If you want those, import from the `vector` and/or `rect` modules.
pub mod prelude {
    pub use crate::scalar::{ Scalar, IntScalar, SignedScalar, FloatScalar };
    pub use crate::vector::{ Vector, IntVector, SignedVector, FloatVector };
    pub use crate::rect::{ Rect, SignedRect, FloatRect };
}