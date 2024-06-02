use std::f32::consts as f32consts;
use std::f64::consts as f64consts;

use swift_vec::prelude::*;
use approx::assert_relative_eq;
use swift_vec::vector::Vectorized;

#[test]
fn vec2_angle_to() {

    use swift_vec::vector::Vec2;
    
    // Create four directional vectors
    let up:    Vec2<f32> = Vec2::up();
    let down:  Vec2<f32> = Vec2::down();
    let left:  Vec2<f32> = Vec2::left();
    let right: Vec2<f32> = Vec2::right();

    // Test the angle between the vectors
    assert_relative_eq!(up.angle_to(up),    0.0);
    assert_relative_eq!(up.angle_to(down),  f32consts::PI);
    assert_relative_eq!(up.angle_to(left), -f32consts::FRAC_PI_2);
    assert_relative_eq!(up.angle_to(right), f32consts::FRAC_PI_2);
}

#[test]
fn vec2_rotate() {

    use swift_vec::vector::Vec2;

    // Create a northbound vector and rotate the vector 90 degrees counterclockwise.
    // Check that the vector is now westbound.
    let north:   Vec2<f64> = Vec2::up();
    let rotated: Vec2<f64> = north.rotated(f64consts::FRAC_PI_2);

    assert!(rotated.approx_eq(Vec2::left()));

    // Create a vector from the rotation of 180 degrees and rotate the vector 90 degrees clockwise.
    // Check that the vector is now of a 90 degree angle.
    let rotated_180: Vec2<f64> = Vec2::from_angle(180f64.to_radians());
    let rotated_90:  Vec2<f64> = rotated_180.rotated(-f64consts::FRAC_PI_2);

    assert_relative_eq!(rotated_90.angle().to_degrees(), 90f64);
}

#[test]
fn vec2_log() {

    use swift_vec::vector::Vec2;

    // Create a vector of 10,
    // then compute the logarithm of base 10 of the vector.
    let vec: Vec2<usize> = Vec2(10, 10);
    let log: Vec2<usize> = vec.log(10);

    // Check that the result is 1.
    assert_eq!(log, Vec2(1, 1));
}

#[test]
fn vec2_interpolation() {

    use swift_vec::vector::Vec2;

    // Cubically interpolate between two vectors.
    let a: Vec2<f32> = Vec2::of(1.0);
    let b: Vec2<f32> = Vec2::of(2.0);

    let pre_start:     Vec2<f32> = Vec2::of(-5.0);
    let post_terminal: Vec2<f32> = Vec2::of( 3.0);

    let c_025: Vec2<f32> = a.cubic_interpolate(b, pre_start, post_terminal, 0.25);
    let c_050: Vec2<f32> = a.cubic_interpolate(b, pre_start, post_terminal, 0.50);
    let c_075: Vec2<f32> = a.cubic_interpolate(b, pre_start, post_terminal, 0.75);

    assert!(c_025.approx_eq(Vec2(1.601563, 1.601563)));
    assert!(c_050.approx_eq(Vec2(1.8125,   1.8125  )));
    assert!(c_075.approx_eq(Vec2(1.867188, 1.867188)));

    // Test the interpolation in time.
    let terminal_t:      f32 = -3.0;
    let pre_start_t:     f32 =  0.0;
    let post_terminal_t: f32 =  2.0;

    let ct_025: Vec2<f32> = a.cubic_interpolate_in_time(b, pre_start, post_terminal, 0.25, terminal_t, pre_start_t, post_terminal_t);
    let ct_050: Vec2<f32> = a.cubic_interpolate_in_time(b, pre_start, post_terminal, 0.50, terminal_t, pre_start_t, post_terminal_t);
    let ct_075: Vec2<f32> = a.cubic_interpolate_in_time(b, pre_start, post_terminal, 0.75, terminal_t, pre_start_t, post_terminal_t);

    assert!(ct_025.approx_eq(Vec2(-2.378125, -2.378125)));
    assert!(ct_050.approx_eq(Vec2(-0.425,    -0.425   )));
    assert!(ct_075.approx_eq(Vec2( 0.990625,  0.990625)));
}

#[test]
fn vec2_global() {

    use swift_vec::prelude::*;
    use swift_vec::vector::{ Vec2, Axis2 };

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
    let vec_i32:   Vec2<i32>   = 1.dvec();
    let vec_u32:   Vec2<u32>   = Vec2::of(2);
    let vec_usize: Vec2<usize> = Vec2::of(3);
    let vec_f64:   Vec2<f64>   = Vec2::of(4.0);

    // Vectors can be cast to other types, and can be manipulated just like any other numerical data.
    let avg: Vec2<f32> = (vec_i32.cast().unwrap() + vec_u32.cast().unwrap() + vec_usize.cast().unwrap() + vec_f64.cast().unwrap()) / 4.0;
    
    assert_eq!(avg, Vec2(2.5, 2.5));

    // Several operations are implemented, such as dot/cross products, magnitude/normalization, etc.
    let dot: f64 = Vec2(5.0, 3.0).dot(Vec2(1.0, 2.0));
    let mag: f64 = Vec2(3.0, 4.0).magnitude();
    
    assert_eq!(dot, 11.0);
    assert_eq!(mag, 5.0);
    assert_eq!(Vec2(3.0, 4.0).normalized().magnitude(), 1.0);

    // Interpolation is added to both scalar and vector types.
    let a: Vec2<f32> = Vec2(1.0, 2.0);
    let b: Vec2<f32> = Vec2(3.0, 3.0);
    let c: Vec2<f32> = a.lerp(b, 0.5);

    assert_eq!(c, Vec2(2.0, 2.5));
    assert_eq!(1.0.lerp(2.0, 0.5), 1.5);

    // min(), max(), and clamp() functions are also implemented for floating point scalars.
    assert_eq!(1.0.min(2.0), 1.0);
    assert_eq!(1.0.max(2.0), 2.0);
    assert_eq!(1.0.clamp(0.0, 2.0), 1.0);
}

#[test]
fn rect2_global() {

    use swift_vec::prelude::*;
    use swift_vec::vector::{ Vec2, Axis2 };
    use swift_vec::rect::{ Rect2, Side2 };

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

    assert_eq!(bigger_rect.encompasses(&smaller_rect.cast().unwrap()), true);   // Casting is supported.
    assert_eq!(smaller_rect.encompasses(&bigger_rect.cast().unwrap()), false);

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
