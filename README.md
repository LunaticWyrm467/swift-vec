# SwiftVec
![GitHub License](https://img.shields.io/github/license/LunaticWyrm467/SwiftVec)

**SwiftVec** is an easy-to-use, intuitive vector maths library with a ton
of functionality for game development, physics simulations, and other potential use cases.

**⚠️WARNING⚠️**<br>
This crate is in its infancy! 3D and 4D vectors are missing, along with other planned functionality.
Beware of bugs!

## Getting Started!
Simply either run `cargo add swift_vec` at the terminal directed towards the directory of your project,
or add `swift_vec = X.X` to your `cargo.toml` file.

To show off some basic functionality of what this crate allows for;
```rust
use swift_vec::prelude::*;
use swift_vec::vector::Vec2;

fn main() {

  // Vectors support all primitive numerical types.
  let vec_i32:   Vec2<i32>   = Vec2::ones_like();
  let vec_u32:   Vec2<u32>   = Vec2::of(2);
  let vec_usize: Vec2<usize> = Vec2::of(3);
  let vec_f64:   Vec2<f64>   = Vec2::of(4.0);

  // Vectors can be cast to other types, and can be manipulated just like any other numerical data.
  let avg: Vec2<f32> = (vec_i32.cast() + vec_u32.cast() + vec_usize.cast() + vec_f64.cast()) / 4.0;
  
  assert_eq!(avg, Vec2(2.5, 2.5));

  // Several operations are implemented, such as dot/cross products, magnitude/normalization, etc.
  let dot: f64 = Vec2(5.0, 3.0).dot(&Vec2(1.0, 2.0));
  let mag: f64 = Vec2(3.0, 4.0).magnitude();
  
  assert_eq!(dot, 11.0);
  assert_eq!(mag, 5.0);
  assert_eq!(Vec2(3.0, 4.0).normalized().magnitude(), 1.0);

  // Interpolation is added to both scalar and vector types.
  let a: Vec2<f32> = Vec2(1.0, 2.0);
  let b: Vec2<f32> = Vec2(3.0, 3.0);
  let c: Vec2<f32> = a.lerp(&b, 0.5);

  assert_eq!(c, Vec2(2.0, 2.5));
  assert_eq!(1.0.lerp(2.0, 0.5), 1.5);

  // min(), max(), and clamp() functions are also implemented for floating point scalars.
  assert_eq!(1.0.min(2.0), 1.0);
  assert_eq!(1.0.max(2.0), 2.0);
  assert_eq!(1.0.clamp(0.0, 2.0), 1.0);
}
```

## Features
- ℹ️ Simple yet intuitive syntax. No messy constructors!
- ➕ Standard vector arithmetic and operations.
- ⛛ Trigonometric functions and angle manipulation.
- ↗️ Standard vector operations such as `magnitude()`, `normalize()`, `dot()`, `cross()`, etc.
- 🪞 Reflection and refraction functions.
- 🌍 Geometric comparisons and operations such as `distance_to()`, `slide()`, etc.
- 🐌 Different interpolation methods such as `lerp()`, `bezier_sample()`, `cubic_interpolate()`, and `cubic_interpolate_in_time()`.
- 📚 Aliases for common functions, such as `length()` for `magnitude()`.
