use crate::error::DigiFiError;
use crate::consts::{GAMMA_DK, GAMMA_R};


/// # Description
/// Natural logarithm of the Gamma function.
///
/// # Input
/// - `z`: Real part of a complex number
///
/// # Output
/// - Natural logarithm of the Gamma function
///
/// # LaTeX Formula
/// - \\Gamma(z) = \\int^{\\infty}_{0}t^{z-1}e^{-t}dt
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Gamma_function>
/// - Original Source: N/A
pub fn ln_gamma(z: f64) -> f64 {
    let ln_2_sqrt_e_over_pi: f64 = (2.0 * (std::f64::consts::E / std::f64::consts::PI).sqrt()).ln();
    if z < 0.5 {
        let s = GAMMA_DK.iter().enumerate().skip(1).fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - z));
        std::f64::consts::PI.ln() - (std::f64::consts::PI * z).sin().ln() - s.ln() - ln_2_sqrt_e_over_pi
            - (0.5 - z) * ((0.5 - z + GAMMA_R) / std::f64::consts::E).ln()
    } else {
        let s = GAMMA_DK.iter().enumerate().skip(1).fold(GAMMA_DK[0], |s, t| s + t.1 / (z + t.0 as f64 - 1.0));
        s.ln() + ln_2_sqrt_e_over_pi + (z - 0.5) * ((z - 0.5 + GAMMA_R) / std::f64::consts::E).ln()
    }
}


/// # Description
/// Gamma function is the most common extension of the factorial function to complex numbers.
///
/// # Input
/// - `z`: Real part of a complex number
///
/// # Output
/// - Evaluation of Gamma function at point `z`
///
/// # LaTeX Formula
/// - \\Gamma(z) = \\int^{\\infty}_{0}t^{z-1}e^{-t}dt
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Gamma_function>
/// - Original Source: N/A
///
/// # Examples
/// 
/// ```rust
/// use digifi::utilities::{TEST_ACCURACY, maths_utils::factorial};
/// use digifi::statistics::gamma;
/// 
/// // Gamma(1)
/// assert!((gamma(1.0) - 1.0).abs() < TEST_ACCURACY);
/// 
/// // Gamma(3/2)
/// let theoretical_result: f64 = std::f64::consts::PI.sqrt() * (factorial(factorial(3 - 2)) as f64) / 2.0_f64.powf((3.0 - 1.0) / 2.0);
/// assert!((gamma(3.0/2.0) - theoretical_result).abs() < TEST_ACCURACY);
/// ```
pub fn gamma(z: f64) -> f64 {
    ln_gamma(z).exp()
}


/// # Description
/// Gamma function with an integral limit defined over the range `(0, x)`.
///
/// # Input
/// - `z`: Real part of a complex number
/// - `x`: Upper integral limit of the lower incomplete Gamma function
/// - `n_terms`: Number of terms to use in the approximation (Default is 20)
///
/// # Output
/// - Evaluation of lower incomplete Gamma function
///
/// # Errors
/// - Returns an error if the value of `x` is negative.
///
/// # LaTeX Formula
/// - \\gamma(z, x) = \\int^{x}_{0}t^{z-1}e^{-t}dt
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Incomplete_gamma_function>
/// - Original Source: N/A
///
/// # Examples
/// 
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::lower_incomplete_gamma;
/// 
/// // Gamma_{lower}(1, x) = 1 - e^{-x}
/// assert!((lower_incomplete_gamma(1.0, 3.0, None).unwrap() - (1.0 - (-3.0_f64).exp())).abs() < TEST_ACCURACY);
/// ```
pub fn lower_incomplete_gamma(z: f64, x: f64, n_terms: Option<usize>) -> Result<f64, DigiFiError> {
    if x < 0.0 {
        return Err(DigiFiError::ParameterConstraint { title: "Lower Incomplete Gamma Function".to_owned(), constraint: "The value of `x` must be non-negative.".to_owned(), });
    }
    let n_terms: usize = n_terms.unwrap_or(20);
    let mut result: f64 = 0.0;
    for k in 0..n_terms {
        let mut denominator: f64 = 1.0;
        if k == 0 {
            denominator *= z;
        } else {
            for i in 0..(k+1) {
                denominator *= z + (i as f64);
            }
        } 
        result += x.powi(k as i32) / denominator;
    }
    Ok(x.powf(z) * (-x).exp() * result)
}


/// # Description
/// Gamma function with an integral limit defined over the range `(x, infinity)`.
///
/// # Input
/// - `z`: Real part of a complex number
/// - `x`: Lower integral limit of the upper incomplete Gamma function
/// - `n_terms`: Number of terms to use in the approximation (Default is 20)
///
/// # Output
/// - Evaluation of upper incomplete Gamma function
///
/// # Errors
/// - Returns an error if the value of `x` is negative.
///
/// # LaTeX Formula
/// - \\Gamma(z, x) = \\int^{\\infty}_{x}t^{z-1}e^{-t}dt
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Incomplete_gamma_function>
/// - Original Source: N/A
///
/// # Examples
/// 
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{gamma, upper_incomplete_gamma};
/// 
/// // Gamma_{upper}(s, 0) = Gamma(s)
/// assert!((upper_incomplete_gamma(4.0, 0.0, Some(30)).unwrap() - gamma(4.0)).abs() < TEST_ACCURACY);
/// 
/// // Gamma_{upper}(1, x) = e^{-x}
/// assert!((upper_incomplete_gamma(1.0, 3.0, Some(30)).unwrap() - (-3.0_f64).exp()).abs() < TEST_ACCURACY);
/// ```
pub fn upper_incomplete_gamma(z: f64, x: f64, n_terms: Option<usize>) -> Result<f64, DigiFiError> {
    if x < 0.0 {
        return Err(DigiFiError::ParameterConstraint { title: "Upper Incomplete Gamma Function".to_owned(), constraint: "The value of `x` must be non-negative.".to_owned(), });
    }
    Ok(gamma(z) - lower_incomplete_gamma(z, x, n_terms)?)
}


/// # Description
/// Digamma function, which is the logarithmic derivative of the gamma function.
///
/// # Input
/// - `z`: Positive real part of a complex number
///
/// # Output
/// - Evaluation of Digamma function
///
/// # Errors
/// - Returns an error if the value of `z` is non-positive.
///
/// # LaTeX Formula
/// - \\psi(z) = \\frac{d}{dz}ln\\Gamma(Z)
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Digamma_function>
/// - Original Source: N/A
/// 
/// /// # Examples
///
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::consts::EULER_MASCHERONI_CONSTANT;
/// use digifi::statistics::digamma;
/// 
/// // Digamma(1) = -gamma
/// assert!((digamma(1.0).unwrap() + EULER_MASCHERONI_CONSTANT).abs() < 1_000.0 * TEST_ACCURACY);
/// 
/// // Digamma(0.5) = -2ln(2) - gamma
/// assert!((digamma(0.5).unwrap() + (2.0 * 2.0_f64.ln() + EULER_MASCHERONI_CONSTANT)).abs() < 1_000.0 * TEST_ACCURACY);
/// 
/// // Digamma(1/3) = -Pi/(2 sqrt(3)) - 3 ln(3)/2 - gamma
/// assert!((digamma(1.0/3.0).unwrap() + (std::f64::consts::PI/(2.0 * 3.0_f64.sqrt()) + 3.0 * 3.0_f64.ln() / 2.0 + EULER_MASCHERONI_CONSTANT)).abs() < 1_000.0 * TEST_ACCURACY);
/// ```
pub fn digamma(z: f64) -> Result<f64, DigiFiError> {
    if z <= 0.0 {
        return Err(DigiFiError::ParameterConstraint { title: "Digamma Function".to_owned(), constraint: "The value of `z` must be positive.".to_owned(), });
    }
    let mut shifted_z: f64 = z;
    let mut result: f64 = 0.0;
    while shifted_z < 6.0 {
        result -= 1.0 / shifted_z;
        shifted_z += 1.0;
    }
    let z: f64 = shifted_z;
    let proxy_result: f64 = z.ln() - 1.0/(2.0 * z) - 1.0/(12.0 * z.powi(2) + 1.0/(120.0 * z.powi(4)) - 1.0/(252.0 * z.powi(6)) + 1.0/(240.0 * z.powi(8)) - 1.0/(132.0 * z.powi(10)) + 691.0/(32760.0 * z.powi(12)) - 1.0/(12.0 * z.powi(14)));
    Ok(proxy_result + result)
}


#[cfg(test)]
mod tests {
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_gamma() -> () {
        use crate::utilities::maths_utils::factorial;
        use crate::statistics::gamma::gamma;
        // Gamma(1)
        assert!((gamma(1.0) - 1.0).abs() < TEST_ACCURACY);
        // Gamma(3/2)
        let theoretical_result: f64 = std::f64::consts::PI.sqrt() * (factorial(factorial(3 - 2)) as f64) / 2.0_f64.powf((3.0 - 1.0) / 2.0);
        assert!((gamma(3.0/2.0) - theoretical_result).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_upper_incomplete_gamma() -> () {
        use crate::statistics::gamma::lower_incomplete_gamma;
        // Gamma_{lower}(1, x) = 1 - e^{-x}
        assert!((lower_incomplete_gamma(1.0, 3.0, None).unwrap() - (1.0 - (-3.0_f64).exp())).abs() < TEST_ACCURACY);
    }

    #[test]
    fn uni_test_upper_incomplete_gamma() -> () {
        use crate::statistics::gamma::{gamma, upper_incomplete_gamma};
        // Gamma_{upper}(s, 0) = Gamma(s)
        assert!((upper_incomplete_gamma(4.0, 0.0, Some(30)).unwrap() - gamma(4.0)).abs() < TEST_ACCURACY);
        // Gamma_{upper}(1, x) = e^{-x}
        assert!((upper_incomplete_gamma(1.0, 3.0, Some(30)).unwrap() - (-3.0_f64).exp()).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_digamma() -> () {
        use crate::consts::EULER_MASCHERONI_CONSTANT;
        use crate::statistics::gamma::digamma;
        // Digamma(1) = -gamma
        assert!((digamma(1.0).unwrap() + EULER_MASCHERONI_CONSTANT).abs() < 1_000.0 * TEST_ACCURACY);
        // Digamma(0.5) = -2ln(2) - gamma
        assert!((digamma(0.5).unwrap() + (2.0 * 2.0_f64.ln() + EULER_MASCHERONI_CONSTANT)).abs() < 1_000.0 * TEST_ACCURACY);
        // Digamma(1/3) = -Pi/(2 sqrt(3)) - 3 ln(3)/2 - gamma
        assert!((digamma(1.0/3.0).unwrap() + (std::f64::consts::PI/(2.0 * 3.0_f64.sqrt()) + 3.0 * 3.0_f64.ln() / 2.0 + EULER_MASCHERONI_CONSTANT)).abs() < 1_000.0 * TEST_ACCURACY);
    }
}