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

/// Generally speaking, you'll want to use the prelude module to get all of the traits and functions you'll need.
/// This does not include any of the individual types. If you want those, import from the `vector` and `scalar` modules.
pub mod prelude {
    pub use crate::vector::{ IntVector, SignedVector, FloatVector };
    pub use crate::scalar::{ IntScalar, SignedScalar, FloatScalar };
}