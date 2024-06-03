
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
