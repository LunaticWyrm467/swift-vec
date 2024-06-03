
#[test]
pub fn scalar_global() -> () {
    use swift_vec::prelude::*;
    
    // Interpolation is added to floats.
    assert_eq!(1.0.lerp(2.0, 0.5), 1.5);

    // min(), max(), and clamp() functions are also implemented for floating point scalars.
    assert_eq!(1.0.min(2.0), 1.0);
    assert_eq!(1.0.max(2.0), 2.0);
    assert_eq!(1.0.clamp(0.0, 2.0), 1.0);
}
