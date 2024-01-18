use std::f32::consts as f32consts;
use std::f64::consts as f64consts;

use swift_vec::prelude::*;
use approx::assert_relative_eq;

#[test]
fn vec2_angle_to() {

    use swift_vec::vector::Vec2;
    
    // Create four directional vectors
    let up:    Vec2<f32> = Vec2::up();
    let down:  Vec2<f32> = Vec2::down();
    let left:  Vec2<f32> = Vec2::left();
    let right: Vec2<f32> = Vec2::right();

    // Test the angle between the vectors
    assert_relative_eq!(up.angle_to(&up),    0.0);
    assert_relative_eq!(up.angle_to(&down),  f32consts::PI);
    assert_relative_eq!(up.angle_to(&left), -f32consts::FRAC_PI_2);
    assert_relative_eq!(up.angle_to(&right), f32consts::FRAC_PI_2);
}

#[test]
fn vec2_rotate() {

    use swift_vec::vector::Vec2;

    // Create a northbound vector and rotate the vector 90 degrees counterclockwise.
    // Check that the vector is now westbound.
    let north:   Vec2<f64> = Vec2::up();
    let rotated: Vec2<f64> = north.rotated(f64consts::FRAC_PI_2);

    assert!(rotated.approx_eq(&Vec2::left()));

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

    let c_025: Vec2<f32> = a.cubic_interpolate(&b, &pre_start, &post_terminal, 0.25);
    let c_05:  Vec2<f32> = a.cubic_interpolate(&b, &pre_start, &post_terminal, 0.50);
    let c_075: Vec2<f32> = a.cubic_interpolate(&b, &pre_start, &post_terminal, 0.75);

    assert!(c_025.approx_eq(&Vec2(1.601563, 1.601563)));
    assert!(c_05.approx_eq( &Vec2(1.8125,   1.8125  )));
    assert!(c_075.approx_eq(&Vec2(1.867188, 1.867188)));

    // Test the interpolation in time.
    let terminal_t:      f32 = -3.0;
    let pre_start_t:     f32 =  0.0;
    let post_terminal_t: f32 =  2.0;

    let ct_025: Vec2<f32> = a.cubic_interpolate_in_time(&b, &pre_start, &post_terminal, 0.25, terminal_t, pre_start_t, post_terminal_t);
    let ct_05:  Vec2<f32> = a.cubic_interpolate_in_time(&b, &pre_start, &post_terminal, 0.50, terminal_t, pre_start_t, post_terminal_t);
    let ct_075: Vec2<f32> = a.cubic_interpolate_in_time(&b, &pre_start, &post_terminal, 0.75, terminal_t, pre_start_t, post_terminal_t);

    assert!(ct_025.approx_eq(&Vec2(-2.378125, -2.378125)));
    assert!(ct_05.approx_eq( &Vec2(-0.425,    -0.425   )));
    assert!(ct_075.approx_eq(&Vec2( 0.990625,  0.990625)));
}