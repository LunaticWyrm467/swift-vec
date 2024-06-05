

/*
 * Vec3
 *      Tests
 */


#[test]
fn vec3_global() {
    use swift_vec::prelude::*;
    use swift_vec::vector::{ Vec2, Vec3, Axis3 };

    // Supports tuple destructuring and field indexing.
    let Vec3(x, y, z): Vec3<i32> = Vec3(0, 0, 1);
    match Vec3(x, y, z) {
        Vec3( 0,  0,  1) => println!("Front"),
        Vec3( 0,  0, -1) => println!("Back"),
        Vec3( 0,  1,  0) => println!("Up"),
        Vec3( 0, -1,  0) => println!("Down"),
        Vec3(-1,  0,  0) => println!("Left"),
        Vec3( 1,  0,  0) => println!("Right"),
        _                => println!("Other")
    }

    let test_vec:    Vec3<i32> = Vec3(1, 0, 2);
    let argmax_axis: Axis3     = test_vec.argmax();
    let argmax_val:  i32       = test_vec[argmax_axis];

    assert_eq!(argmax_axis, Axis3::Z);
    assert_eq!(argmax_val, 2);

    // Vectors support all primitive numerical types and support multiple construction methods.
    let vec_i32:   Vec3<i32>   = 1.vec3();
    let vec_u32:   Vec3<u32>   = (1, 2, 3).vec3();
    let vec_usize: Vec3<usize> = (3, Vec2(2, 3)).vec3();
    let vec_f64:   Vec3<f64>   = (Vec2(5.0, 6.0), 4.0).vec3();

    // Vectors can be cast to other types, and can be manipulated just like any other numerical data.
    let avg: Vec3<f32> = (vec_i32.cast().unwrap() + vec_u32.cast().unwrap() + vec_usize.cast().unwrap() + vec_f64.cast().unwrap()) / 4.0;
    
    assert_eq!(avg, Vec3(2.5, 2.75, 2.75));

    // Several operations are implemented, such as dot/cross products, magnitude/normalization, etc.
    let dot: f64 = Vec3(5.0, 4.0, 3.0).dot(Vec3(1.0, 2.0, 3.0));
    let mag: f64 = Vec3(3.0, 4.0, 5.0).magnitude();
    
    assert_eq!(dot, 22.0);
    assert_eq!(mag, 5.0 * 2.0f64.sqrt());
    assert!(Vec3(3.0, 4.0, 5.0).normalized().magnitude().approx_eq(1.0));

    // Interpolation is added to vector types.
    let a: Vec3<f32> = Vec3(1.0, 2.0, 3.0);
    let b: Vec3<f32> = Vec3(3.0, 3.0, 3.0);
    let c: Vec3<f32> = a.lerp(b, 0.5);

    assert_eq!(c, Vec3(2.0, 2.5, 3.0));
}


/*
 * Matrix
 *      Tests
 */


#[test]
fn mat3_transforms_self() -> () {
    use core::f32::consts::TAU;

    use swift_vec::prelude::*;
    use swift_vec::vector::{ Vec3, SignedAxis3 };
    use swift_vec::mat::Mat3;

    /*
     * Test scaling here.
     */
    
    // Try creating a matrix from a scale.
    let control: Mat3<u32> = Mat3::new(
        2, 0, 0,
        0, 4, 0,
        0, 0, 8
    );

    let scale: Vec3<u32> = Vec3(2, 4, 8);
    let mat:   Mat3<u32> = Mat3::from_scale(scale);

    assert_eq!(mat, control);
    
    // Rotating the matrix in any way preserves its scale.
    let mut mat: Mat3<f32> = mat.cast().unwrap();
            mat            = mat.rotated(TAU / 2.0, SignedAxis3::YPos);
            mat            = mat.rotated(TAU / 4.0, SignedAxis3::XPos);
    
    assert!(mat.get_scale().approx_eq(Vec3(2.0, 4.0, 8.0)));
    
    // Try scaling.
    let control: Mat3<i32> = Mat3::new(
        0, 2, -2,
        0, 4, -4,
        0, 6, -6
    ).transposed();   // Note: Control computed via godot, which uses column-major order.

    let mut mat: Mat3<i32> = Mat3::new(
        1, 2, 3, 
	    1, 2, 3, 
	    1, 2, 3
    );

    mat = mat.scaled(Vec3(0, 2, -2));

    assert_eq!(mat, control);

    /*
     * Test rotation here.
     */

    // Test creating a matrix from a rotation.
    let control1: Mat3<f32> = Mat3::new(
        1.0,  0.0,  0.0,
        0.0, -0.0,  1.0,
        0.0, -1.0, -0.0
    ).transposed();   // Note: Control computed via godot, which uses column-major order.
    let control2: Mat3<f32> = Mat3::new(
         0.0, 1.0, 0.0,
        -1.0, 0.0, 0.0,
         0.0, 0.0, 1.0
    ).transposed();   // Note: Control computed via godot, which uses column-major order.
    let control3: Mat3<f32> = Mat3::new(
        0.837795,  0.162205, -0.521334,
        0.162205,  0.837795,  0.521334,
        0.521334, -0.521334,  0.67559
    ).transposed();   // Note: Control computed via godot, which uses column-major order.

    let mat1: Mat3<f32> = Mat3::from_angle(90.0_f32.to_radians(),  SignedAxis3::XPos);
    let mat2: Mat3<f32> = Mat3::from_angle(270.0_f32.to_radians(), SignedAxis3::ZNeg);
    let mat3: Mat3<f32> = Mat3::from_angle_free(47.5_f32.to_radians(), Vec3(0.5, 0.5, 0.0).normalized());

    assert!(mat1.approx_eq(&control1));
    assert!(mat2.approx_eq(&control2));
    assert!(mat3.approx_eq(&control3));

    // Test rotating a matrix here.
    let control: Mat3<f32> = Mat3::new(
         1.0, 0.0, -0.0,
        -0.0, 1.0, -0.0,
         0.0, 0.0,  1.0
    );
    
    let angle: f32 = TAU / 2f32;

    let mut mat: Mat3<f32> = Mat3::IDENTITY;
            mat            = mat.rotated(angle, SignedAxis3::YPos);  // Rotate around the up axis (yaw).
	        mat            = mat.rotated(angle, SignedAxis3::XPos);  // Rotate around the right axis (pitch).
	        mat            = mat.rotated(angle, SignedAxis3::ZPos);  // Rotate around the back axis (roll).
    
    assert!(mat.approx_eq(&control));

    /*
     * Test inversion here.
     */

    let control: Mat3<f32> = Mat3::new(
         0.452055,  0.041096, -0.39726,
        -0.054795, -0.09589,   0.260274,
        -0.041096,  0.178082, -0.054795
    );
    
    let mat: Mat3<f32> = Mat3::new(
        3.0, 5.0, 2.0,
        1.0, 3.0, 7.0,
        1.0, 6.0, 3.0
    );
    
    assert!(mat.inverse().approx_eq(&control));

    /*
     * Test normalization here.
     */
    
    let mut mat: Mat3<f32> = Mat3::IDENTITY.scaled(Vec3(3.0, 4.0, 6.0));
            mat            = mat.rotated(TAU, SignedAxis3::YPos);
	        mat            = mat.rotated(TAU, SignedAxis3::XPos);
	        mat            = mat.orthonormalized();
    
    assert!(Vec3(
        mat.x.length(),
        mat.y.length(),
        mat.z.length()
    ).approx_eq(1.0.vec3()));
}

#[test]
fn mat3_transforms_vector() -> () {
    use swift_vec::prelude::*;
    use swift_vec::vector::{ Vec3, SignedAxis3 };
    use swift_vec::mat::Mat3;
    
    let test_case_1: Vec3<f32> = Vec3(6.0, 2.0, 3.0);
    let test_case_2: Vec3<f32> = Vec3(180.0, -38.0, 0.0);
    let test_case_3: Vec3<f32> = Vec3(-1.0, -5.0, -2.0);

    /*
     * Rotations
     */
    
    // Test Transformations
    let rotation_1: Mat3<f32> = Mat3::from_angle(45.0f32.to_radians(),   SignedAxis3::XPos);
    let rotation_2: Mat3<f32> = Mat3::from_angle(176.0f32.to_radians(),  SignedAxis3::YNeg);
    let rotation_3: Mat3<f32> = Mat3::from_angle(-284.0f32.to_radians(), SignedAxis3::ZPos);
    
    let rotated_11: Vec3<f32> = &rotation_1 * test_case_1;
    let rotated_12: Vec3<f32> = &rotation_2 * test_case_1;
    let rotated_13: Vec3<f32> = &rotation_3 * test_case_1;
    
    let rotated_21: Vec3<f32> = &rotation_1 * test_case_2;
    let rotated_22: Vec3<f32> = &rotation_2 * test_case_2;
    let rotated_23: Vec3<f32> = &rotation_3 * test_case_2;
    
    let rotated_31: Vec3<f32> = &rotation_1 * test_case_3;
    let rotated_32: Vec3<f32> = &rotation_2 * test_case_3;
    let rotated_33: Vec3<f32> = &rotation_3 * test_case_3;
    
    assert!(rotated_11.approx_eq(Vec3(6.0, -0.70710670948029, 3.5355339050293)));
    assert!(rotated_12.approx_eq(Vec3(-6.19465398788452, 2.0, -2.57415342330933)));
    assert!(rotated_13.approx_eq(Vec3(-0.48905980587006, 6.30561828613281, 3.0)));
    
    assert!(rotated_21.approx_eq(Vec3(180.0, -26.8700580596924, -26.8700580596924)));
    assert!(rotated_22.approx_eq(Vec3(-179.561538696289, -38.0, 12.5561647415161)));
    assert!(rotated_23.approx_eq(Vec3(80.4171905517578, 165.460189819336, 0.0)));
    
    assert!(rotated_31.approx_eq(Vec3(-1.0, -2.12132024765015, -4.94974756240845)));
    assert!(rotated_32.approx_eq(Vec3(1.1370769739151, -5.0, 1.9253716468811)));
    assert!(rotated_33.approx_eq(Vec3(4.60955667495728, -2.1799054145813, -2.0)));

    // Test Inverse Transformations.
    let inv_rotated_11: Vec3<f32> = rotated_11 * &rotation_1;
    let inv_rotated_12: Vec3<f32> = rotated_12 * &rotation_2;
    let inv_rotated_13: Vec3<f32> = rotated_13 * &rotation_3;
    
    let inv_rotated_21: Vec3<f32> = rotated_21 * &rotation_1;
    let inv_rotated_22: Vec3<f32> = rotated_22 * &rotation_2;
    let inv_rotated_23: Vec3<f32> = rotated_23 * &rotation_3;
    
    let inv_rotated_31: Vec3<f32> = rotated_31 * &rotation_1;
    let inv_rotated_32: Vec3<f32> = rotated_32 * &rotation_2;
    let inv_rotated_33: Vec3<f32> = rotated_33 * &rotation_3;

    assert!(inv_rotated_11.approx_eq(test_case_1));
    assert!(inv_rotated_12.approx_eq(test_case_1));
    assert!(inv_rotated_13.approx_eq(test_case_1));
    
    assert!(inv_rotated_21.approx_eq(test_case_2));
    assert!(inv_rotated_22.approx_eq(test_case_2));
    assert!(inv_rotated_23.approx_eq(test_case_2));
    
    assert!(inv_rotated_31.approx_eq(test_case_3));
    assert!(inv_rotated_32.approx_eq(test_case_3));
    assert!(inv_rotated_33.approx_eq(test_case_3));

    /*
     * Scale
     */

    let scale_1: Mat3<f32> = Mat3::from_scale(Vec3(3.0, 1.0, 6.0));
    let scale_2: Mat3<f32> = Mat3::from_scale(Vec3(-2.0, 7.0, 180.0));
    let scale_3: Mat3<f32> = Mat3::from_scale(Vec3(-1.0, -284.0, -5.0));
    
    let scaled_11: Vec3<f32> = &scale_1 * test_case_1;
    let scaled_12: Vec3<f32> = &scale_2 * test_case_1;
    let scaled_13: Vec3<f32> = &scale_3 * test_case_1;
    
    let scaled_21: Vec3<f32> = &scale_1 * test_case_2;
    let scaled_22: Vec3<f32> = &scale_2 * test_case_2;
    let scaled_23: Vec3<f32> = &scale_3 * test_case_2;
    
    let scaled_31: Vec3<f32> = &scale_1 * test_case_3;
    let scaled_32: Vec3<f32> = &scale_2 * test_case_3;
    let scaled_33: Vec3<f32> = &scale_3 * test_case_3;

    assert!(scaled_11.approx_eq(Vec3( 18.0, 2.0,    18.0)));
    assert!(scaled_12.approx_eq(Vec3(-12.0, 14.0,   540.0)));
    assert!(scaled_13.approx_eq(Vec3(-6.0, -568.0, -15.0)));
    
    assert!(scaled_21.approx_eq(Vec3( 540.0, -38.0,    0.0)));
    assert!(scaled_22.approx_eq(Vec3(-360.0, -266.0,   0.0)));
    assert!(scaled_23.approx_eq(Vec3(-180.0,  10792.0, 0.0)));
    
    assert!(scaled_31.approx_eq(Vec3(-3.0, -5.0,   -12.0)));
    assert!(scaled_32.approx_eq(Vec3( 2.0, -35.0,  -360.0)));
    assert!(scaled_33.approx_eq(Vec3( 1.0,  1420.0, 10.0)));
    
    // TODO: Test Inverse Transformations.
    // (Once gaussian elimintation is implemented)
    /*
    let inv_scaled_11: Vec3<f32> = scaled_11 * &scale_1;
    let inv_scaled_12: Vec3<f32> = scaled_12 * &scale_2;
    let inv_scaled_13: Vec3<f32> = scaled_13 * &scale_3;
    
    let inv_scaled_21: Vec3<f32> = scaled_21 * &scale_1;
    let inv_scaled_22: Vec3<f32> = scaled_22 * &scale_2;
    let inv_scaled_23: Vec3<f32> = scaled_23 * &scale_3;
    
    let inv_scaled_31: Vec3<f32> = scaled_31 * &scale_1;
    let inv_scaled_32: Vec3<f32> = scaled_32 * &scale_2;
    let inv_scaled_33: Vec3<f32> = scaled_33 * &scale_3;
    
    assert!(inv_scaled_11.approx_eq(test_case_1));
    assert!(inv_scaled_12.approx_eq(test_case_1));
    assert!(inv_scaled_13.approx_eq(test_case_1));
    
    assert!(inv_scaled_21.approx_eq(test_case_2));
    assert!(inv_scaled_22.approx_eq(test_case_2));
    assert!(inv_scaled_23.approx_eq(test_case_2));
    
    assert!(inv_scaled_31.approx_eq(test_case_3));
    assert!(inv_scaled_32.approx_eq(test_case_3));
    assert!(inv_scaled_33.approx_eq(test_case_3));
    */
}

#[test]
fn mat3_euler() -> () {
    use core::f32::consts::TAU;

    use swift_vec::vector::Vec3;
    use swift_vec::mat::{ EulerOrder, Mat3 };
    
    // We generate two matrices to ensure that the euler representation is the same.
    let control: Mat3<f32> = Mat3 {
        x: Vec3(1.0,  0.0,  0.0),
        y: Vec3(0.0, -0.0,  1.0),
        z: Vec3(0.0, -1.0, -0.0)
    }.transposed();   // Note: Control computed via godot, which uses column-major order.

    let euler:     Vec3<f32> = Vec3(TAU / 4.0, 0.0, 0.0);
    let mat:       Mat3<f32> = Mat3::from_euler_angle(euler, EulerOrder::OrderYXZ);
    let new_euler: Vec3<f32> = mat.to_euler(EulerOrder::OrderYXZ);
    let new_mat:   Mat3<f32> = Mat3::from_euler_angle(new_euler, EulerOrder::OrderYXZ);

    assert!(mat.approx_eq(&control));
    assert!(new_mat.approx_eq(&mat));
}
