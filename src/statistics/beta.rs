use ndarray::Array1;
use crate::error::DigiFiError;
use crate::consts;
use crate::statistics::gamma::{ln_gamma, gamma};


/// Natural logarithm of the Beta function.
/// 
/// # Input
/// - `a`: Real part of complex number
/// - `b`: Real part of complex number
/// 
/// # Output
/// - Natural logarithm of the Beta function
/// 
/// # Errors
/// - Returns an error if the argument `a` is not positive
/// - Returns an error if the argument `b` is not positive
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Beta_function>
/// - Original Source: N/A
pub fn ln_beta(a: f64, b: f64) -> Result<f64, DigiFiError> {
    let error_title: String = String::from("Regularized Incomplete Beta function");
    if a <= 0.0 {
        return Err(DigiFiError::ParameterConstraint { title: error_title, constraint: "The argument `a` must be positive.".to_owned(), });
    }
    if b <= 0.0 {
        return Err(DigiFiError::ParameterConstraint { title: error_title, constraint: "The argument `b` must be positive.".to_owned(), });
    }
    Ok(ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b))
}


/// Beta function (or Euler integral of the first kind) is a special function that is closely related to the gamma function and to
/// binomial coefficients.
///
/// # Input
/// - `a`: Real part of complex number
/// - `b`: Real part of complex number
///
/// # Output
/// - Evaluation of Beta function
///
/// # Errors
/// - Returns an error if the argument `a` is not positive.
/// - Returns an error if the argument `b` is not positive.
///
/// # LaTeX Formula
/// - B(z_{1},z_{2}) = \\int^{1}_{0}t^{z_{1}-1}(1-t)^{z_2{}-1}dt
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Beta_function>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::beta;
/// 
/// // B(1, x) = 1/x
/// assert!((beta(1.0, 3.0).unwrap() - (1.0 / 3.0)).abs() < TEST_ACCURACY);
/// 
/// // B(x, 1-x) = Pi / sin(Pi*x)
/// assert!((beta(0.5, 0.5).unwrap() - std::f64::consts::PI).abs() < TEST_ACCURACY);
/// ```
pub fn beta(a: f64, b: f64) -> Result<f64, DigiFiError> {
    ln_beta(a, b).map(|v| v.exp() )
}


/// Generalization of Beta function with an upper integral bound that can be set.
///
/// # Input
/// - `x`: Upper bound of the definite integral
/// - `a`: Real part of complex number
/// - `b`: Real part of complex number
///
/// # Output
/// - Evaluation of incomplete Beta function
///
/// # Errors
/// - Returns an error if the argument `a` is not positive.
/// - Returns an error if the argument `b` is not positive.
/// - Returns an error if the argument `x` is not in the range `[0, 1]`.
///
/// # LaTeX Formula
/// - B(x;a,b) = \\int^{x}_{0}t^{a-1}(1-t)^{b-1}dt
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Beta_function>
/// - Original Source: N/A
pub fn incomplete_beta(x: f64, a: f64, b: f64) -> Result<f64, DigiFiError> {
    Ok(beta(a, b)? * regularized_incomplete_beta(x, a, b)?)
}


/// Regularized incomplete Beta function acts as a cumulative distribution of the Beta distribution.
///
/// # Input
/// - `x`: Upper bound of the definite integral
/// - `a`: Real part of complex number
/// - `b`: Real part of complex number
///
/// # Output
/// - Evaluation of regularized incomplete Beta function
///
/// # Errors
/// - Returns an error if the argument `a` is not positive.
/// - Returns an error if the argument `b` is not positive.
/// - Returns an error if the argument `x` is not in the range `[0, 1]`.
///
/// # LaTeX Formula
/// - I_{x}(a,b) = \\frac{B(x;a,b)}{B(a,b)}
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Beta_function>
/// - Original Source: N/A
///
/// # Examples
/// 
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::regularized_incomplete_beta;
/// 
/// // I_{0}(a, b) = 0
/// assert!(regularized_incomplete_beta(0.0, 0.2, 0.3).unwrap() < TEST_ACCURACY);
/// 
/// // I_{1}(a, b) = 1
/// assert!((regularized_incomplete_beta(1.0, 0.2, 0.3).unwrap() - 1.0 ).abs() < TEST_ACCURACY);
/// 
/// // I_{x}(a, 1) = x^{a}
/// assert!((regularized_incomplete_beta(0.5, 2.0, 1.0).unwrap() - 0.5_f64.powi(2)).abs() < TEST_ACCURACY);
/// 
/// // I_{x}(1, b) = 1 - (1 - x)^{b}
/// assert!((regularized_incomplete_beta(0.5, 1.0, 3.0).unwrap() - (1.0 - 0.5_f64.powi(3))).abs() < TEST_ACCURACY);
/// ```
pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> Result<f64, DigiFiError> {
    let error_title: String = String::from("Regularized Incomplete Beta function");
    if a <= 0.0 {
        return Err(DigiFiError::ParameterConstraint { title: error_title, constraint: "The argument `a` must be positive.".to_owned(), });
    }
    if b <= 0.0 {
        return Err(DigiFiError::ParameterConstraint { title: error_title, constraint: "The argument `b` must be positive.".to_owned(), });
    }
    if !(0.0..=1.0).contains(&x) {
        return Err(DigiFiError::ParameterConstraint { title: error_title, constraint: "The argument `x` must contain values in the range `[0, 1]`.".to_owned(), })
    }
    let bt: f64 = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp()
    };
    let symm_transform: bool = x >= (a + 1.0) / (a + b + 2.0);
    let fpmin: f64 = f64::MIN_POSITIVE / consts::STANDARD_EPSILON;
    let (mut x, mut a, mut b) = (x, a, b);
    if symm_transform {
        (x, a, b) = (1.0 - x, b, a);
    }
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let (mut c, mut d) = (1.0, 1.0 - qab * x / qap);
    if d.abs() < fpmin { d = fpmin; }
    d = 1.0 / d;
    let mut h: f64 = d;
    for m in 1..141 {
        let m: f64 = f64::from(m);
        let m2: f64 = m * 2.0;
        let mut aa: f64 = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin { d = fpmin; }
        c = 1.0 + aa / c;
        if c.abs() < fpmin { c = fpmin; }
        d = 1.0 / d;
        h = h * d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin { d = fpmin; }
        c = 1.0 + aa / c;
        if c.abs() < fpmin { c = fpmin; }
        d = 1.0 / d;
        let del: f64 = d * c;
        h *= del;
        if (del - 1.0).abs() <= consts::STANDARD_EPSILON {
            return if symm_transform {
                Ok(1.0 - bt * h / a)
            } else {
                Ok(bt * h / a)
            };
        }
    }
    if symm_transform {
        Ok(1.0 - bt * h / a)
    } else {
        Ok(bt * h / a)
    }
}


/// Inverse of the regularized incomplete Beta function.
///
/// # Input
/// - `x`: Value of regularized incomplete Beta function
/// - `a`: Real part of complex number
/// - `b`: Real part of complex number
///
/// # Output
/// - Inverse of the regularized incomplete Beta function
///
/// # Errors
/// - Returns an error if the argument `a` is not positive.
/// - Returns an error if the argument `b` is not positive.
/// - Returns an error if the argument `x` is not in the range `[0, 1]`.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function>
/// - Original Source: <http://www.jstor.org/stable/2346798>
/// 
/// # Examples
/// 
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::inverse_regularized_incomplete_beta;
/// 
/// // I_{0}(a, b) = 0
/// assert!(inverse_regularized_incomplete_beta(0.0, 0.2, 0.3).unwrap() < TEST_ACCURACY);
/// 
/// // I_{1}(a, b) = 1
/// assert!((inverse_regularized_incomplete_beta(1.0, 0.2, 0.3).unwrap() - 1.0 ).abs() < TEST_ACCURACY);
/// 
/// // I_{x}(a, 1) = x^{a}
/// assert!((inverse_regularized_incomplete_beta(0.5_f64.powi(2), 2.0, 1.0).unwrap() - 0.5).abs() < TEST_ACCURACY);
/// 
/// // I_{x}(1, b) = 1 - (1 - x)^{b}
/// assert!((inverse_regularized_incomplete_beta(1.0 - 0.5_f64.powi(3), 1.0, 3.0).unwrap() - 0.5).abs() < TEST_ACCURACY);
/// ```
pub fn inverse_regularized_incomplete_beta(mut x: f64, mut a: f64, mut b: f64) -> Result<f64, DigiFiError> {
    let error_title: String = String::from("Inverse Regularized Incomplete Beta function");
    if a <= 0.0 {
        return Err(DigiFiError::ParameterConstraint { title: error_title, constraint: "The argument `a` must be positive.".to_owned(), });
    }
    if b <= 0.0 {
        return Err(DigiFiError::ParameterConstraint { title: error_title, constraint: "The argument `b` must be positive.".to_owned(), });
    }
    if !(0.0..=1.0).contains(&x) {
        return Err(DigiFiError::ParameterConstraint { title: error_title, constraint: "The argument `x` must contain values in the range `[0, 1]`.".to_owned(), })
    }
    let ln_beta: f64 = ln_beta(a, b)?;
    // Remark AS R83: http://www.jstor.org/stable/2347779
    const SAE: i32 = -30;
    const FPU: f64 = 1e-30; // 10^SAE
    if x == 0.0 { return Ok(0.0); }
    if x == 1.0 { return Ok(1.0); }
    let mut p: f64;
    let mut q: f64;
    let flip: bool = 0.5 < x;
    if flip {
        p = a;
        a = b;
        b = p;
        x = 1.0 - x;
    }
    p = (-(x * x).ln()).sqrt();
    q = p - (2.30753 + 0.27061 * p) / (1.0 + (0.99229 + 0.04481 * p) * p);
    if 1.0 < a && 1.0 < b {
        // Remark AS R19 and Algorithm AS 109: http://www.jstor.org/stable/2346887
        let r: f64 = (q * q - 3.0) / 6.0;
        let s: f64 = 1.0 / (2.0 * a - 1.0);
        let t: f64 = 1.0 / (2.0 * b - 1.0);
        let h: f64 = 2.0 / (s + t);
        let w: f64 = q * (h + r).sqrt() / h - (t - s) * (r + 5.0 / 6.0 - 2.0 / (3.0 * h));
        p = a / (a + b * (2.0 * w).exp());
    } else {
        let mut t: f64 = 1.0 / (9.0 * b);
        t = 2.0 * b * (1.0 - t + q * t.sqrt()).powf(3.0);
        if t <= 0.0 {
            p = 1.0 - ((((1.0 - x) * b).ln() + ln_beta) / b).exp();
        } else {
            t = 2.0 * (2.0 * a + b - 1.0) / t;
            if t <= 1.0 {
                p = (((x * a).ln() + ln_beta) / a).exp();
            } else {
                p = 1.0 - 2.0 / (t + 1.0);
            }
        }
    }
    p = p.clamp(0.0001, 0.9999);
    // Remark AS R83: http://www.jstor.org/stable/2347779
    let e: i32 = (-5.0 / a / a - 1.0 / x.powf(0.2) - 13.0) as i32;
    let acu: f64 = if e > SAE { f64::powi(10.0, e) } else { FPU };
    let mut pnext: f64;
    let mut qprev: f64 = 0.0;
    let mut sq: f64 = 1.0;
    let mut prev: f64 = 1.0;
    'outer: loop {
        // Remark AS R19 and Algorithm AS 109: http://www.jstor.org/stable/2346887
        q = regularized_incomplete_beta(p, a, b)?;
        q = (q - x) * (ln_beta + (1.0 - a) * p.ln() + (1.0 - b) * (1.0 - p).ln()).exp();
        // Remark AS R83: http://www.jstor.org/stable/2347779
        if q * qprev <= 0.0 {
            prev = if sq > FPU { sq } else { FPU };
        }
        // Remark AS R19 and Algorithm AS 109: http://www.jstor.org/stable/2346887
        let mut g: f64 = 1.0;
        loop {
            loop {
                let adj: f64 = g * q;
                sq = adj * adj;
                if sq < prev {
                    pnext = p - adj;
                    if (0.0..=1.0).contains(&pnext) {
                        break;
                    }
                }
                g /= 3.0;
            }
            if prev <= acu || q * q <= acu {
                p = pnext;
                break 'outer;
            }
            if pnext != 0.0 && pnext != 1.0 {
                break;
            }
            g /= 3.0;
        }
        if pnext == p {
            break;
        }
        p = pnext;
        qprev = q;
    }
    if flip { Ok(1.0 - p) } else { Ok(p) }
}


/// Extension of the Beta function with more than two arguments.
/// 
/// # Input
/// - `a`: Array of positive real numbers
/// 
/// # Output
/// - Evaluation of the multivariate Beta function
/// 
/// # Errors
/// - Returns an error if one of the values of argumetn `a` is non-positive
/// 
/// # LaTeX Formula
/// B(a_{1}, ..., a_{n}) = \\frac{\\Gamma(a_{1})...\\Gamma(a_{n})}{\\Gamma(a_{1} + ... + a_{n})}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function>
/// - Original Source: N/A
pub fn multivariate_beta(a: &Array1<f64>) -> Result<f64, DigiFiError> {
    for a_ in a.iter() {
        if a_ <= &0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: "Multivariate Beta Function".to_owned(),
                constraint: "The argument `a` must be positive.".to_owned(),
            });
        }
    }
    let (a_cum, numerator) = a.iter().fold((0.0, 1.0), |(a_cum, numerator), a| {
        (a_cum + a, numerator * gamma(*a))
    } );
    Ok(numerator / gamma(a_cum))
}


#[cfg(test)]
mod tests {
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_beta() -> () {
        use crate::statistics::beta::beta;
        // B(1, x) = 1/x
        assert!((beta(1.0, 3.0).unwrap() - (1.0 / 3.0)).abs() < TEST_ACCURACY);
        // B(x, 1-x) = Pi / sin(Pi*x)
        assert!((beta(0.5, 0.5).unwrap() - std::f64::consts::PI).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_regularized_incomplete_beta() -> () {
        use crate::statistics::beta::regularized_incomplete_beta;
        // I_{0}(a, b) = 0
        assert!(regularized_incomplete_beta(0.0, 0.2, 0.3).unwrap() < TEST_ACCURACY);
        // I_{1}(a, b) = 1
        assert!((regularized_incomplete_beta(1.0, 0.2, 0.3).unwrap() - 1.0 ).abs() < TEST_ACCURACY);
        // I_{x}(a, 1) = x^{a}
        assert!((regularized_incomplete_beta(0.5, 2.0, 1.0).unwrap() - 0.5_f64.powi(2)).abs() < TEST_ACCURACY);
        // I_{x}(1, b) = 1 - (1 - x)^{b}
        assert!((regularized_incomplete_beta(0.5, 1.0, 3.0).unwrap() - (1.0 - 0.5_f64.powi(3))).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_incomplete_regularized_incomplete_beta() -> () {
        use crate::statistics::beta::inverse_regularized_incomplete_beta;
        // I_{0}(a, b) = 0
        assert!(inverse_regularized_incomplete_beta(0.0, 0.2, 0.3).unwrap() < TEST_ACCURACY);
        // I_{1}(a, b) = 1
        assert!((inverse_regularized_incomplete_beta(1.0, 0.2, 0.3).unwrap() - 1.0 ).abs() < TEST_ACCURACY);
        // I_{x}(a, 1) = x^{a}
        assert!((inverse_regularized_incomplete_beta(0.5_f64.powi(2), 2.0, 1.0).unwrap() - 0.5).abs() < TEST_ACCURACY);
        // I_{x}(1, b) = 1 - (1 - x)^{b}
        assert!((inverse_regularized_incomplete_beta(1.0 - 0.5_f64.powi(3), 1.0, 3.0).unwrap() - 0.5).abs() < TEST_ACCURACY);
    }
}