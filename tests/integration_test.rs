use std::f32::consts as f32consts;
use std::f64::consts as f64consts;

use swift_vec;
use approx::assert_relative_eq;

#[test]
fn vec2_angle_to() {

    use swift_vec::vector::v2d::Vec2;
    use swift_vec::vector::FloatVector;
    
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

    use swift_vec::vector::v2d::Vec2;
    use swift_vec::vector::FloatVector;

    // Create a northbound vector
    let north: Vec2<f64> = Vec2::up();

    // Rotate the vector 90 degrees counterclockwise.
    let rotated: Vec2<f64> = north.rotated(f64consts::FRAC_PI_2);

    // Check that the vector is now westbound.
    assert!(rotated.approx_eq(&Vec2::left()));
}

#[test]
fn vec2_log() {

    use swift_vec::vector::v2d::Vec2;
    use swift_vec::vector::IntVector;

    // Create a vector of 10.
    let vec: Vec2<usize> = Vec2(10, 10);

    // Compute the logarithm of base 10 of the vector.
    let log: Vec2<usize> = vec.log(10);

    // Check that the result is 1.
    assert_eq!(log, Vec2(1, 1));
}