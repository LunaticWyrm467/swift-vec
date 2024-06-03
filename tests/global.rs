
#[test]
pub fn global() -> () {
    use swift_vec::prelude::*;
    use swift_vec::vector::{ Vec2, Vec3, Axis2 };
    use Axis2::*;   // It is recommended to import from the Axis enums if you're going to be
                    // indexing a lot.

    // Supports tuple destructuring and field indexing.
    let Vec2(x, y): Vec2<i32> = Vec2(1, 0);
    match Vec2(x, y) {
        Vec2( 0,  1) => println!("Up"),
        Vec2( 0, -1) => println!("Down"),
        Vec2(-1,  0) => println!("Left"),
        Vec2( 1,  0) => println!("Right"),
        _            => println!("Other")
    }

    let mut test_vec:    Vec2<i32> = Vec2(1, 0);
    let     argmax_axis: Axis2     = test_vec.argmax();
    let     argmax_val:  i32       = test_vec[argmax_axis];

    test_vec[X] = 2;   // You could always use tuple fields (`test_vec.0`) but this is more readable.
    test_vec[Y] = 3;

    assert_eq!(argmax_axis, Axis2::X);
    assert_eq!(argmax_val,  1);
    assert_eq!(test_vec,    Vec2(2, 3));
    
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
    let a: Vec2<f32> = Vec2(1.0, 2.0);
    let b: Vec2<f32> = Vec2(3.0, 3.0);
    let c: Vec2<f32> = a.lerp(b, 0.5);

    assert_eq!(c, Vec2(2.0, 2.5));
    
    let a: Vec2<f32> = 1.0.vec2();
    let b: Vec2<f32> = 2.0.vec2();

    let pre_a:  Vec2<f32> = -5.0.vec2();
    let post_b: Vec2<f32> =  3.0.vec2();

    let c_025: Vec2<f32> = a.cubic_interpolate(b, pre_a, post_b, 0.25);
    let c_050: Vec2<f32> = a.cubic_interpolate(b, pre_a, post_b, 0.50);
    let c_075: Vec2<f32> = a.cubic_interpolate(b, pre_a, post_b, 0.75);

    assert!(c_025.approx_eq(Vec2(1.601563, 1.601563)));
    assert!(c_050.approx_eq(Vec2(1.8125,   1.8125  )));
    assert!(c_075.approx_eq(Vec2(1.867188, 1.867188)));
}
