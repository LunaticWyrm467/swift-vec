# SwiftVec
![Crates.io License](https://img.shields.io/crates/l/swift_vec?color=green)
[![Crates.io Version](https://img.shields.io/crates/v/swift_vec)](https://crates.io/crates/swift_vec)
[![Documentation](https://docs.rs/swift_vec/badge.svg)](https://docs.rs/swift_vec)

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
use swift_vec::vector::{ Vec2, Axis2 };

fn main() {

    // Supports tuple destructuring and field indexing.
    let Vec2(x, y): Vec2<i32> = Vec2(1, 0);
    match Vec2(x, y) {
        Vec2( 0,  1) => println!("Up"),
        Vec2( 0, -1) => println!("Down"),
        Vec2(-1,  0) => println!("Left"),
        Vec2( 1,  0) => println!("Right"),
        _            => println!("Other")
    }

    let test_vec:    Vec2<i32> = Vec2(1, 0);
    let argmax_axis: Axis2     = test_vec.argmax();
    let argmax_val:  i32       = test_vec.get(argmax_axis);

    assert_eq!(argmax_axis, Axis2::X);
    assert_eq!(argmax_val, 1);

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

We also provide functionality for rectangles and their associated geometric functions;
```rust
use swift_vec::prelude::*;
use swift_vec::vector::{ Vec2, Axis2 };
use swift_vec::rect::{ Rect2, Side2 };

fn main() {

    // Just like vectors, rectangles can be destructured and indexed.
    let Rect2(position, dimensions): Rect2<i32> = Rect2(Vec2(1, 1), Vec2(3, 6));
    let rect:                        Rect2<i32> = Rect2(position, dimensions);

    let longest_axis:   Axis2 = rect.longest_axis();
    let longest_length: i32   = rect.longest_axis_length();

    assert_eq!(longest_axis,   Axis2::Y);
    assert_eq!(longest_length, 6);

    // There are checks in place for determining whether rectangles intersect, and to allow for the
    // computation of their cross-section.
    let rect_a: Rect2<f32> = Rect2::from_offsets(-5.0, -5.0, 5.0, 5.0);
    let rect_b: Rect2<f32> = Rect2::from_components(-10.0, -10.0, 7.5, 7.5);

    assert_eq!(rect_a.intersects(&rect_b, false), true);   // `include_borders` is set to false - not that it matters here.
    assert_eq!(rect_a.intersection(&rect_b).unwrap(), Rect2(Vec2(-5.0, -5.0), Vec2(2.5, 2.5)));

    let smaller_rect: Rect2<isize> = Rect2::unit();
    let bigger_rect:  Rect2<i64>   = Rect2(Vec2(-32, -32), Vec2(64, 64));

    assert_eq!(bigger_rect.encompasses(&smaller_rect.cast()), true);   // Casting is supported.
    assert_eq!(smaller_rect.encompasses(&bigger_rect.cast()), false);

    // Rectangles can be checked to see if they contain a point.
    let platform: Rect2<i16> = Rect2(Vec2(0, 0), Vec2(100, 100));
    let point:    Vec2<i16>  = Vec2(50, 50);

    assert_eq!(platform.encompasses_point(point), true);

    // Rectangles can be merged and their shape can be manipulated.
    let rect_a: Rect2<i32> = Rect2::from_components(-3, -3, 3, 3);
    let rect_b: Rect2<i32> = Rect2::from_components(3, 3, 3, 3);
    let merged: Rect2<i32> = rect_a.merge(&rect_b);
    
    assert_eq!(merged, Rect2(Vec2(-3, -3), Vec2(9, 9)));

    let base_rect: Rect2<i32> = Rect2::unit();
    let mod_rect:  Rect2<i32> = base_rect.grow_side(Side2::Top, 5);

    assert_eq!(mod_rect, Rect2(Vec2(0, -5), Vec2(1, 6)));
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
