use std::{borrow::Borrow, iter::zip};
use crate::error::DigiFiError;
use crate::utilities::{NUMERICAL_CORRECTION, compare_len};


/// Factorial of n.
/// 
/// # Input
/// -`n`: Input variable
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Factorial>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::factorial;
///
/// assert_eq!(3628800, factorial(10));
/// assert_eq!(1, factorial(0));
/// ```
pub fn factorial(n: u128) -> u128 {
    (1..=n).product()
}


/// Rising factorial (Pochhammer function) of x.
///
/// # Input
/// - `x`: Input variable
/// - `n`: Number of factors in the rising factorial
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Falling_and_rising_factorials>
/// - Original Source: N/A
///
/// /// # Examples
///
/// ```rust
/// use digifi::utilities::rising_factorial;
///
/// assert_eq!(rising_factorial(3, 4), 360);
/// ```
pub fn rising_factorial(x: u128, n: u128) -> u128 {
    let mut result: u128 = 1;
    for k in 0..n {
        result *= x + k;
    }
    result
}


/// Error function computed with the Taylor expansion.
/// 
/// Input
/// - `x`: Input variable
/// - `n_terms`: Number of terms to use in the approximation (Default is 20)
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Error_function>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::{TEST_ACCURACY, erf};
///
/// assert!((erf(1.0, None) - 0.8427007929497149).abs() < 10.0 * TEST_ACCURACY);
/// ```
pub fn erf(x: f64, n_terms: Option<usize>) -> f64 {
    let n_terms: usize = n_terms.unwrap_or(20);
    let total: f64 = (0..n_terms).into_iter().fold(0.0, |total, n| {
        let exp: i32 = (2 * n + 1) as i32;
        total + (-1.0_f64).powi(n as i32) * x.powi(exp) / (factorial(n as u128) as f64 * (2 * n + 1) as f64)
    } );
    (2.0 / f64::sqrt(std::f64::consts::PI)) * total
}


/// Inverse error function computed with the Taylor expansion.
/// 
/// # Input
/// - `z`: Input variable
/// - `n_terms`: Number of terms to use in the approximation (Default is 20)
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Error_function#Inverse_functions>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::{TEST_ACCURACY, erfinv};
///
/// let approximation: f64 = erfinv(0.8427007929497149, Some(100));
/// assert!((approximation - 1.0).abs() < TEST_ACCURACY);
/// ```
pub fn erfinv(z: f64, n_terms: Option<usize>) -> f64 {
    let n_terms: usize = n_terms.unwrap_or(20);
    let pi_sqrt_half_z: f64 = f64::sqrt(std::f64::consts::PI) * z / 2.0;
    let mut c: Vec<f64> = Vec::with_capacity(n_terms);
    (0..n_terms).into_iter().fold(0.0, |total, k| {
        let two_k_plus_one: i32 = (2 * k + 1) as i32;
        let c_sum: f64 = match k {
            0 => 1.0,
            _ => (0..k).into_iter().fold(0.0, |total, m| total + c[m] * c[k-1-m] / ((m + 1) * (2 * m + 1)) as f64 )
        };
        c.push(c_sum);
        total + c_sum / (two_k_plus_one as f64) * pi_sqrt_half_z.powi(two_k_plus_one)
    } )
}


/// The distance between two points in a Euclidean space.
/// 
/// # Input
/// - `v_1`: Coordinate of the first point.
/// - `v_2`: Coordinate of the second point.
/// 
/// # LaTeX Formula
/// - d(x,y) = \\sqrt{\\sum^{n}_{i=1}(x_{i} - y_{i})^{2}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Euclidean_distance>
/// - Original Source: N/A
pub fn euclidean_distance<T, I>(v_1: T, v_2: T) -> Result<f64, DigiFiError>
where
    T: Iterator<Item = I> + ExactSizeIterator,
    I: Borrow<f64>,
{
    compare_len(&v_1, &v_2, "v_1", "v_2")?;
    Ok(zip(v_1, v_2).fold(0.0, |sum, (v1, v2)| { sum + (v1.borrow() - v2.borrow()).powi(2) } ).sqrt())

}


/// Numerical differentiation (symmetric difference quotient) of a function.
///
/// # Input
/// - `f`: Function to differentiate
/// - `x`: Point at which the derivative is takem
/// - `h`: Interval between consequtive values of `x`
///
/// # LaTeX Formula
/// - \\frac{df}{dx} = \\frac{f(x+h) - f(x-h)}{2h}
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Numerical_differentiation>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::{TEST_ACCURACY, derivative};
///
/// let f = |x: f64| { x.powi(2) + 3.0*x - 2.0 };
/// assert!((derivative(f, 3.0, 0.00000001) - 9.0).abs() < 1_000.0 * TEST_ACCURACY);
/// ```
pub fn derivative<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}


/// Numerical solution (composite trapezoidal rule) of a definite integral of a function.
///
/// # Input
/// - `f`: Function to perform definite integration on
/// - `start`: Lower bound of the definite integral
/// - `end`: Upper bound of the definite integral
/// - `n_intervals`: Number of intervals to use in the composite trapezoidal rule
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Numerical_integration>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::{TEST_ACCURACY, definite_integral};
///
/// let f = |x: f64| { x.exp() };
///
/// // Both bounds are finite
/// let integral: f64 = definite_integral(f, 0.0, 10.0, 1_000_000);
/// assert!((integral - 22_025.46579).abs() < 50_000_000.0 * TEST_ACCURACY);
///
/// // Lower bound is infinite and upper bound is finite
/// let integral: f64 = definite_integral(f, f64::NEG_INFINITY, 0.0, 1_000_000);
/// assert!((integral - 1.0).abs() < 1_000.0 * TEST_ACCURACY);
///
/// // Lower bound is finite and upper bound is infinite
/// let f = |x: f64| { (-x).exp() };
/// let integral: f64 = definite_integral(f, 0.0, f64::INFINITY, 1_000_000);
/// assert!((integral - 1.0).abs() < 1_000.0 * TEST_ACCURACY);
///
/// // Both bounds are infinite
/// let f = |x: f64| { (-x.powi(2) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt() };
/// let integral: f64 = definite_integral(f, f64::NEG_INFINITY, f64::INFINITY, 1_000_000);
/// assert!((integral - 1.0).abs() < 1_000.0 * TEST_ACCURACY);
/// ```
pub fn definite_integral<F: Fn(f64) -> f64>(f: F, start: f64, end: f64, n_intervals: usize) -> f64 {
    let n: f64 = n_intervals as f64;
    let a: f64;
    let b: f64;
    // Transform the integral bounds from infinite to finite
    let g: Box<dyn Fn(f64) -> f64> = match (start, end) {
        (f64::NEG_INFINITY, f64::INFINITY) => {
            a = -1.0 + NUMERICAL_CORRECTION;
            b = 1.0 - NUMERICAL_CORRECTION;
            Box::new(move |t: f64| { f(t/(1.0 - t.powi(2))) * (1.0 + t.powi(2)) / (1.0 - t.powi(2)).powi(2) })
        },
        (f64::NEG_INFINITY, v) => {
            a = NUMERICAL_CORRECTION;
            b = 1.0 - NUMERICAL_CORRECTION;
            Box::new(move |t: f64| { f(v - (1.0 - t)/t) / t.powi(2) })
        },
        (v, f64::INFINITY) => {
            a = NUMERICAL_CORRECTION;
            b = 1.0 - NUMERICAL_CORRECTION;
            Box::new(move |t: f64| { f(v + t/(1.0 - t)) / (1.0 - t).powi(2) })
        },
        (a_, b_) => {
            match a_ {
                0.0 => { a = NUMERICAL_CORRECTION; },
                _ => {a = a_; },
            }
            match b_ {
                0.0 => { b = NUMERICAL_CORRECTION; },
                _ => { b = b_; },
            }
            Box::new(move |t: f64| { f(t) })
        },
    };
    // Composite trapezoidal rule
    let scale_factor: f64 = (b - a) / n;
    let result: f64 = (1..(n_intervals - 1)).into_iter().fold(g(a)/2.0 + g(b)/2.0, |result, i| result + g(a + (i as f64) * scale_factor) );
    scale_factor * result
}


#[cfg(test)]
mod tests {
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_factorial() -> () {
        use crate::utilities::maths_utils::factorial;
        assert_eq!(3628800, factorial(10));
        assert_eq!(1, factorial(0));
    }

    #[test]
    fn unit_test_rising_factorial() -> () {
        use crate::utilities::maths_utils::rising_factorial;
        assert_eq!(rising_factorial(3, 4), 360);
    }

    #[test]
    fn unit_test_erf() -> () {
        use crate::utilities::maths_utils::erf;
        assert!((erf(1.0, None) - 0.8427007929497149).abs() < 10.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_erfinv() -> () {
        use crate::utilities::maths_utils::erfinv;
        let approximation: f64 = erfinv(0.8427007929497149, Some(100));
        assert!((approximation - 1.0).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_derivative() -> () {
        use crate::utilities::maths_utils::derivative;
        let f = |x: f64| { x.powi(2) + 3.0*x - 2.0 };
        assert!((derivative(f, 3.0, 0.00000001) - 9.0).abs() < 1_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_definite_integral() -> () {
        use crate::utilities::maths_utils::definite_integral;
        let f = |x: f64| { x.exp() };
        // Both bounds are finite
        let integral: f64 = definite_integral(f, 0.0, 10.0, 1_000_000);
        assert!((integral - 22_025.46579).abs() < 50_000_000.0 * TEST_ACCURACY);
        // Lower bound is infinite and upper bound is finite
        let integral: f64 = definite_integral(f, f64::NEG_INFINITY, 0.0, 1_000_000);
        assert!((integral - 1.0).abs() < 1_000.0 * TEST_ACCURACY);
        // Lower bound is finite and upper bound is infinite
        let f = |x: f64| { (-x).exp() };
        let integral: f64 = definite_integral(f, 0.0, f64::INFINITY, 1_000_000);
        assert!((integral - 1.0).abs() < 1_000.0 * TEST_ACCURACY);
        // Both bounds are infinite
        let f = |x: f64| { (-x.powi(2) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt() };
        let integral: f64 = definite_integral(f, f64::NEG_INFINITY, f64::INFINITY, 1_000_000);
        assert!((integral - 1.0).abs() < 1_000.0 * TEST_ACCURACY);
    }
}