/// # Desciption
/// Factorial of n.
/// 
/// # Input
/// -n: Input variable
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Factorial
/// - Original Source: N/A
pub fn factorial(n: u128) -> u128 {
    (1..=n).product()
}


/// # Description
/// Error function computed with the Taylor expansion.
/// 
/// Input
/// - x: Input variable
/// - n_terms: Number of terms to use in a Taylor's expansion of the error function
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Error_function
/// - Original Source: N/A
pub fn erf(x: f64, n_terms: usize) -> f64 {
    let mut total: f64 = 0.0;
    let mut sign: f64 = 1.0;
    for n in 0..n_terms {
        // Safe conversion of usize to i32, panic if it doesn't fit
        let exp: i32 = (2 * n + 1) as i32;
        total += sign * x.powi(exp) / (factorial(n as u128) as f64 * (2 * n + 1) as f64);
        // Flip the sign for the next term
        sign *= -1.0;
    }
    (2.0 / f64::sqrt(std::f64::consts::PI)) * total
}


/// # Description
/// Inverse error function computed with the Taylor expansion.
/// 
/// # Input
/// - z: Input variable
/// - n_terms: Number of terms to use in a Taylor's expansion of the error function
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Error_function#Inverse_functions
/// - Original Source: N/A
pub fn erfinv(z: f64, n_terms: usize) -> f64 {
    let mut total: f64 = 0.0;
    let mut c: Vec<f64> = Vec::<f64>::new();
    for k in 0..n_terms {
        let mut c_sum: f64 = 0.0;
        if k == 0 {
            c_sum = 1.0;
        } else {
            for m in 0..(k) {
                c_sum += c[m] * c[k-1-m] / ((m+1) * (2*m + 1)) as f64;
            }
        }
        c.push(c_sum);
        total += c_sum / ((2*k+1) as f64) * (f64::sqrt(std::f64::consts::PI) * z / 2.0).powi((2*k+1) as i32)
    }
    total
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
    fn unit_test_erf() -> () {
        use crate::utilities::maths_utils::erf;
        let approximation: f64 = erf(1.0, 10);
        assert!((approximation - 0.8427007929497149).abs() < TEST_ACCURACY*10.0);
    }

    #[test]
    fn unit_test_erfinv() -> () {
        use crate::utilities::maths_utils::erfinv;
        let approximation: f64 = erfinv(0.8427007929497149, 100);
        assert!((approximation - 1.0).abs() < TEST_ACCURACY);
    }
}