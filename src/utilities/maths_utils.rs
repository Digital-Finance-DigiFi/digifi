use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;


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
/// nCr:  n choose r
/// 
/// # Input
/// - n: Power of the binomial expansion
/// - r: Number of successes
/// 
/// # Panics
/// - Panics if the value of n is larger than r
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Binomial_coefficient
/// - Original Source: N/A
pub fn n_choose_r(n: u128, r: u128) -> u128 {
    if n < r {
        panic!("The value of variable n must be larger or equal to the value of variable r.");
    }
    factorial(n) / (factorial(n - r) * factorial(r))
}


/// # Description
/// Error function computed with the Taylor expansion.
/// 
/// Input
/// - x Input variables
/// - n_terms: Number of terms to use in a Taylor's expansion of the error function
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Error_function
/// - Original Source: N/A
pub fn erf(x: f64, n_terms: usize) -> f64 {
    let mut total = 0.0;
    let mut sign = 1.0;
    for n in 0..n_terms {
        // Safe conversion of usize to i32, panic if it doesn't fit
        let exp = (2 * n + 1).try_into().unwrap();
        total += sign * x.powi(exp) / (factorial(n as u128) as f64 * (2 * n + 1) as f64);
        // Flip the sign for the next term
        sign *= -1.0;
    }
    (2.0 / f64::sqrt(std::f64::consts::PI)) * total
}


/// # Description
/// Statistical model that estimates the linear dependency between a scalar response and one or more explanatory variables.
/// 
/// # Input
/// - x: Matrix of explanatory variables, where each matrix column corresponds to one variable.
/// - y: Observed response values
/// 
/// # Output
/// - Parameters of the linear regression model
/// 
/// # Panics
/// - Panics if the length of matrix does not match the length of vector y.
/// 
/// # LaTeX Formula
/// - y = X\\cdot\\beta
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Linear_regression
/// - Original Source: N/A
pub fn linear_regression(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    if x.dim().0 != y.len() {
        panic!("The length of x and y do not coincide.");
    }
    let square_matrix: Array2<f64> = x.t().dot(x);
    square_matrix.inv().expect("Failed to inverse the matrix x.").dot(&x.t().dot(&y.t()))
}


#[cfg(test)]
mod tests {
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_factorial() {
        use crate::utilities::maths_utils::factorial;
        assert_eq!(3628800, factorial(10));
        assert_eq!(1, factorial(0));
    }

    #[test]
    fn unit_test_n_choose_r() {
        use crate::utilities::maths_utils::n_choose_r;
        assert_eq!(10, n_choose_r(5, 2));
    }

    #[test]
    fn unit_test_erf() {
        use crate::utilities::maths_utils::erf;
        let approximation = erf(1.0, 10);
        // Allow some margin
        assert!((approximation - 0.8427007929497149).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_linear_regression() -> () {
        use ndarray::{Array1, Array2, array};
        use crate::utilities::maths_utils::linear_regression;
        let y: Array1<f64> = array![1.0, 2.0, 3.0];
        let x: Array2<f64> = array![[1.0, 3.0, 1.0], [4.0, 4.0, 1.0], [6.0, 5.0, 1.0]];
        let params = linear_regression(&x, &y);
        // Comparison vector was found using LinearRegression from sklearn
        let comparison: Array1<f64> = Array1::from(vec![-2.49556592e-16, 1.0, -2.0]);
        assert!((&params - &comparison).sum().abs() < TEST_ACCURACY);
        println!("Hello");
    }
}