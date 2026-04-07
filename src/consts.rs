/// Standard epsilon, maximum relative precision of IEEE 754 double-precision floating point numbers (64 bit) e.g., 2^-53.
/// 
/// Note: This is the rounding machine epsilon, not interval machine epsilon.
pub const STANDARD_EPSILON: f64 = 0.00000000000000011102230246251565;


/// Limiting difference between the harmonic series and the natural logarithm.
pub const EULER_MASCHERONI_CONSTANT: f64 = 0.5772156649;


/// Polynomial coefficients for approximating ln of the Gamma function.
pub const GAMMA_DK: &[f64] = &[
    2.48574089138753565546e-5, 1.05142378581721974210, -3.45687097222016235469, 4.51227709466894823700, -2.98285225323576655721, 1.05639711577126713077,
    -1.95428773191645869583e-1, 1.70970543404441224307e-2, -5.71926117404305781283e-4, 4.63399473359905636708e-6, -2.71994908488607703910e-9,
];


/// Auxiliary variable when evaluating the ln of the Gamma function.
pub const GAMMA_R: f64 = 10.900511;


/// Constant used inside golden ratio numerical engine. The golden ratio is `1.0 : 0.5 + 0.5 * 5.sqrt()`.
/// Therefore, the ratio `1.0 / (1.5 + 0.5 * 5.sqrt())` is used to calculate the fractions in which to divide intervals.
pub const GR_RATIO: f64 = 2.618033988749895; // 1.5 + 0.5 * 5.0_f64.sqrt();


/// Default step between consequtive points for the gradient computation.
pub const H: f64 = std::f64::EPSILON * 1.0e10;