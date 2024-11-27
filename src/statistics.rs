pub mod continuous_distributions;
pub mod discrete_distributions;


use std::io::Error;
use ndarray::{Array1, Array2};
// use ndarray_linalg::Inverse;
use nalgebra::DMatrix;
use num::complex::Complex;
use crate::utilities::{compare_array_len, input_error, data_error, other_error, MatrixConversion};
use crate::utilities::maths_utils::{FunctionEvalMethod, factorial, definite_integral};


pub enum ProbabilityDistributionType {
    Discrete,
    Continuous,
}


pub trait ProbabilityDistribution {
    /// # Description
    /// Mean.
    fn mean(&self) -> f64;

    /// # Description
    /// Median.
    fn median(&self) -> Vec<f64>;

    /// # Description
    /// Mode.
    fn mode(&self) -> Vec<f64>;

    /// # Description
    /// Variance.
    fn variance(&self) -> f64;

    /// # Description
    /// Skewness.
    fn skewness(&self) -> f64;

    /// # Description
    /// Excess kurtosis.
    fn excess_kurtosis(&self) -> f64;

    /// # Description
    /// Entropy.
    fn entropy(&self) -> f64;

    /// # Description
    /// Probability density function.
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, Error>;

    /// # Description
    /// Cummulative distribution function (CDF).
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, Error>;

    /// # Description
    /// Moment generating function (MGF).
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64>;

    /// # Description
    /// Characteristic function (CF).
    /// 
    /// Characteristic function is the Fourier transform of the PDF.
    fn cf(&self, t: &Array1<usize>) -> Array1<Complex<f64>>;

    /// # Description
    /// Inverse cumulative distribution function (CDF), else knwon as quantile function.
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, Error>;
}


/// # Description
/// Measure of joint variability of two random variables.
/// 
/// Cov(X,Y) = E[(X-E[X])(Y-E[Y])]
/// 
/// # Input
/// - `array_1`: First array
/// - `array_2`: Second array
/// 
/// # Output
/// - Covariance of two arrays
/// 
/// # Errors
/// - Returns an error if the lengths of array_1 and array_2 do not match
/// 
/// # LaTeX Formula
/// - Cov(X,Y) = \\frac{1}{n-ddof}\\sum^{n}_{i=1}(x_{i}-\\bar{x})(y_{i}-\\bar{y})
/// where \\bar{x} = \frac{1}{n}\\sum^{n}_{i=1}x_{i}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Covariance
/// - Original Source: N/A
pub fn covariance(array_1: &Array1<f64>, array_2: &Array1<f64>, ddof: usize) -> Result<f64, Error> {
    compare_array_len(array_1, array_2, "array_1", "array_2")?;
    let x: Array1<f64> = array_1 - array_1.mean().ok_or(other_error("Covariance: Mean of array_1 is not computed."))?;
    let y: Array1<f64> = array_2 - array_2.mean().ok_or(other_error("Covariance: Mean of array_2 is not computed."))?;
    Ok((&x * &y).sum() / (x.len() - ddof) as f64)
}


/// # Description
/// Calculates the skewness of a given data array. Skewness measures the asymmetry of the probability distribution of a real-valued random variable about its mean.
/// 
/// The skewness value can be positive, zero, negative, or undefined.
/// 
/// Fisher-Pearson coefficient of skewness is used.
/// 
/// # Input
/// - `array`: An array for which the skewness is to be calculated
/// 
/// # Output
/// - Skewness value of the given data
/// 
/// # LaTeX Formula
/// - \\textit{Skewness} = \\frac{E[(X-\\mu)^{3}]}{(E[(X-\\mu)^{2}])^{\\frac{3}{2}}}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Skewness
/// - Original Source: N/A
pub fn skewness(array: &Array1<f64>) -> Result<f64, Error> {
    let difference: Array1<f64> = array - array.mean().ok_or(other_error("Skewness: Mean of array is not computed."))?;
    let numerator: f64 = difference.map(| i | i.powi(3)).mean().ok_or(other_error("Skewness: Mean of difference array is not computed."))?;
    let denominator: f64 = difference.map(| i | i.powi(2)).mean().ok_or(other_error("Skewness: Mean of difference array is not computed."))?.powf(3.0/2.0);
    if denominator == 0.0 {
        Ok(f64::NAN)
    } else {
        Ok(numerator / denominator)
    }
}


/// # Description
/// Computes the kurtosis of a given data array. Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable.
/// 
/// Higher kurtosis implies a heavier tail.
/// 
/// The calculation here does not subtract 3, hence this is the 'excess kurtosis'.
/// 
/// # Input
/// - `array`: An array for which the kurtosis is to be calculated
/// 
/// # Output
/// - Kurtosis value of the given data
/// 
/// # LaTeX Formula
/// - \\textit{Kurtosis} = \\frac{E[(X-\\mu)^{4}]}{(E[(X-\\mu)^{2}])^{2}}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Kurtosis
/// - Original Source: N/A
pub fn kurtosis(array: &Array1<f64>) -> Result<f64, Error> {
    let difference: Array1<f64> = array - array.mean().ok_or(other_error("Kurtosis: Mean of array is not computed."))?;
    let numerator: f64 = difference.map(| i | i.powi(2)).mean().ok_or(other_error("Kurtosis: Mean of difference array is not computed."))?;
    let denominator: f64 = difference.map(| i | i.powi(2)).mean().ok_or(other_error("Kurtosis: Mean of difference array is not computed."))?.powi(2);
    if denominator == 0.0 {
        Ok(f64::NAN)
    } else {
        Ok(numerator / denominator)
    }
}


/// # Description
/// nCr:  n choose r
/// 
/// # Input
/// - `n`: Power of the binomial expansion
/// - `r`: Number of successes
/// 
/// # Errors
/// - Returns an error if the value of n is larger than r
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Binomial_coefficient
/// - Original Source: N/A
pub fn n_choose_r(n: u128, r: u128) -> Result<u128, Error> {
    if n < r {
        return Err(input_error("n Chose r: The value of variable n must be larger or equal to the value of variable r."));
    }
    Ok(factorial(n) / (factorial(n - r) * factorial(r)))
}


/// # Description
/// Statistical model that estimates the linear dependency between a scalar response and one or more explanatory variables.
/// 
/// # Input
/// - `x`: Matrix of explanatory variables, where each matrix column corresponds to one variable.
/// - `y`: Observed response values
/// 
/// # Output
/// - Parameters of the linear regression model
/// 
/// # Errors
/// - Returns an error if the length of matrix does not match the length of vector y.
/// 
/// # LaTeX Formula
/// - y = X\\cdot\\beta
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Linear_regression
/// - Original Source: N/A
pub fn linear_regression(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    if x.dim().0 != y.len() {
        Err(data_error("Linear Regression: The length of x and y do not coincide."))?;
    }
    let square_matrix: Array2<f64> = x.t().dot(x);
    // Matrix inverse is done via nalgebra.
    let n_square_matrix: DMatrix<f64> = MatrixConversion::ndarray_to_nalgebra(&square_matrix);
    let n_inv_matrix: DMatrix<f64> = n_square_matrix.try_inverse().ok_or("Linear Regression: No matrix inverse exists to perform linear regression.")?;
    let inv_matrix: Array2<f64> = MatrixConversion::nalgebra_to_ndarray(&n_inv_matrix)?;
    Ok(inv_matrix.dot(&x.t().dot(&y.t())))
}


/// # Description
/// The ratio of the systematic variance to the total variance.
/// 
/// # Input
/// - `real_values`: Array of observed or empirical data
/// - `predicted_values`: Array data produced by the model
/// 
/// # Output
/// - r_square: The ratio of exaplained variance to all variance
/// 
/// # LaTeX Formula
/// - R^{2} = 1 - \\frac{\\sum_{i}(y_{i}-f_{i})^{2}}{\\sum_{i}(y_{i}-\\bar{y})^{2}}
/// where ym is the array of real values and f is the array of predicted values.
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Coefficient_of_determination
/// - Original Source: N/A
pub fn r_square(real_values: &Array1<f64>, predicted_values: &Array1<f64>) -> Result<f64, Error> {
    compare_array_len(&real_values, &predicted_values, "real_values", "predicted_values")?;
    let residual_sum: f64 = (real_values - predicted_values).map(|x| x.powi(2)).sum();
    let total_sum: f64 = (real_values - real_values.mean().ok_or(other_error("R Square: Could not compute the mean of real_values."))?).map(|x| x.powi(2)).sum();
    Ok(1.0 - residual_sum/total_sum)
}


/// # Description
/// Adjusted R-square for the upward bias in the R-square due to estimated values of the parameters used.
/// 
/// # Input
/// - `real_values`: Array of observed or empirical data
/// - `predicted_values`: Array data produced by the model
/// - `sample_size`: Number of points used in the model
/// - `k_variables`: Number of variables in the model
/// 
/// # Output
/// - R-square adjusted for upward estimation bias
/// 
/// # LaTeX Formula
/// - R^{2}_{A} = 1 - (1-R^{2})\\frac{n-1}{n-k-1}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
/// - Original Source: N/A
pub fn adjusted_r_square(real_values: &Array1<f64>, predicted_values: &Array1<f64>, sample_size: usize, k_variables: usize) -> Result<f64, Error> {
    Ok(1.0 - (1.0-r_square(real_values, predicted_values)?) * (sample_size as f64 - 1.0) / ((sample_size-k_variables) as f64 - 1.0))
}


/// # Description
/// Gamma function is the most common extension of the factorial function to complex numbers.
///
/// # Input
/// - `z`: Real part of a complex number
/// - `method`: Method for evaluationg the function
///
/// # Output
/// - Evaluation of Gamma function at point `z`
///
/// # LaTeX Formula
/// - \\Gamma(z) = \\int^{\\infty}_{0}t^{z-1}e^{-t}dt
///
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Gamma_function
/// - Original Source: N/A
pub fn gamma_function(z: f64, method: FunctionEvalMethod) -> f64 {
    match method {
        FunctionEvalMethod::Integral { n_intervals } => {
            let f = |t: f64| { t.powf(z - 1.0) * (-t).exp() };
            definite_integral(f, 0.0, f64::INFINITY, n_intervals)
        },
        FunctionEvalMethod::PowerSeries { n_terms } => {
            let mut result: f64 = 1.0;
            for n in 1..n_terms {
                let n_ : f64 = n as f64;
                result *= (1.0 + 1.0 / n_).powf(z) / (1.0 + z / n_);
            }
            result / z
        },
    }
}


/// # Description
/// Gamma function with an integral limit defined over the range (x, infinity).
///
/// # Input
/// - `z`: Real part of a complex number
/// - `x`: Lower integral limit of the upper incomplete Gamma function
/// - `n_intervals`: Number of intervals to use in the composite trapezoidal rule
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
/// - Wikipedia: https://en.wikipedia.org/wiki/Incomplete_gamma_function
/// - Original Source: N/A
pub fn upper_incomplete_gamma_function(z: f64, x: f64, n_intervals: usize) -> Result<f64, Error> {
    if x < 0.0 {
        return Err(input_error("Upper Incomplete Function: The value of x must be non-negative."));
    }
    let f = |t: f64| { t.powf(z - 1.0) * (-t).exp() };
    Ok(definite_integral(f, x, f64::INFINITY, n_intervals))
    
}


/// # Description
/// Gamma function with an integral limit defined over the range (0, x).
///
/// # Input
/// - `z`: Real part of a complex number
/// - `x`: Upper integral limit of the lower incomplete Gamma function
/// - `method`: Method for evaluationg the function
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
/// - Wikipedia: https://en.wikipedia.org/wiki/Incomplete_gamma_function
/// - Original Source: N/A
pub fn lower_incomplete_gamma_function(z: f64, x: f64, method: FunctionEvalMethod) -> Result<f64, Error> {
    if x < 0.0 {
        return Err(input_error("Lower Incomplete Function: The value of x must be non-negative."));
    }
    match method {
        FunctionEvalMethod::Integral { n_intervals } => {
            let f = |t: f64| { t.powf(z - 1.0) * (-t).exp() };
            Ok(definite_integral(f, 0.0, x, n_intervals))
        },
        FunctionEvalMethod::PowerSeries { n_terms } => {
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
        },
    }
}


/// # Description
/// Beta function (or Euler integral of the first kind) is a special function that is closely related to the gamma function and to
/// binomial coefficients.
///
/// # Input
/// - `z_1`: Real part of complex number
/// - `z_2`: Real part of complex number
/// - `n_intervals`: Number of intervals to use in the composite trapezoidal rule
///
/// # Output
/// - Evaluation of Beta function
///
/// # Errors
/// - Returns an error if the argument `z_1` is not positive.
/// - Returns an error if the argument `z_2` is not positive.
///
/// # LaTeX Formula
/// - B(z_{1},z_{2}) = \\int^{1}_{0}t^{z_{1}-1}(1-t)^{z_2{}-1}dt
///
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Beta_function
/// - Original Source: N/A
pub fn beta_function(z_1: f64, z_2: f64, n_intervals: usize) -> Result<f64, Error> {
    if z_1 <= 0.0 {
        return Err(input_error("Beta Function: The argument z_1 must be positive."));
    }
    if z_2 <= 0.0 {
        return Err(input_error("Beta Function: The argument z_2 must be positive."));
    }
    let f = |t: f64| { t.powf(z_1 - 1.0) * (1.0 - t).powf(z_2 - 1.0) };
    Ok(definite_integral(f, 0.0, 0.99999999999999, n_intervals))
}


/// # Description
/// Generalization of Beta function with an upper integral bound that can be set.
///
/// # Input
/// - `x`: Upper bound of the definite integral
/// - `a`: Real part of complex number
/// - `b`: Real part of complex number
/// - `n_intervals`: Number of intervals to use in the composite trapezoidal rule
///
/// # Output
/// - Evaluation of incomplete Beta function
///
/// # Errors
/// - Returns an error if the argument `a` is not positive.
/// - Returns an error if the argument `b` is not positive.
///
/// # LaTeX Formula
/// - B(x;a,b) = \\int^{x}_{0}t^{a-1}(1-t)^{b-1}dt
///
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Beta_function
/// - Original Source: N/A
pub fn incomplete_beta_function(x: f64, a: f64, b: f64, n_intervals: usize) -> Result<f64, Error> {
    if a <= 0.0 {
        return Err(input_error("Beta Function: The argument a must be positive."));
    }
    if b <= 0.0 {
        return Err(input_error("Beta Function: The argument b must be positive."));
    }
    let f = |t: f64| { t.powf(a - 1.0) * (1.0 - t).powf(b - 1.0) };
    Ok(definite_integral(f, 0.0, x, n_intervals))
}


/// # Description
/// Regularized incomplete Beta function acts as a cumulative distribution of the Beta distribution.
///
/// # Input
/// - `x`: Upper bound of the definite integral
/// - `a`: Real part of complex number
/// - `b`: Real part of complex number
/// - `n_intervals`: Number of intervals to use in the composite trapezoidal rule
///
/// # Output
/// - Evaluation of regularized incomplete Beta function
///
/// # Errors
/// - Returns an error if the argument `a` is not positive.
/// - Returns an error if the argument `b` is not positive.
///
/// # LaTeX Formula
/// - I_{x}(a,b) = \\frac{B(x;a,b)}{B(a,b)}
///
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Beta_function
/// - Original Source: N/A
pub fn regularized_incomplete_beta_function(x: f64, a: f64, b: f64, n_intervals: usize) -> Result<f64, Error> {
    if a <= 0.0 {
        return Err(input_error("Beta Function: The argument a must be positive."));
    }
    if b <= 0.0 {
        return Err(input_error("Beta Function: The argument b must be positive."));
    }
    Ok(incomplete_beta_function(x, a, b, n_intervals)? / beta_function(a, b, n_intervals)?)
}



#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};
    use crate::utilities::TEST_ACCURACY;
    use crate::statistics::FunctionEvalMethod;

    #[test]
    fn unit_test_covariance() -> () {
        use crate::statistics::covariance;
        let array_1: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let array_2: Array1<f64> = array![1.0, 5.0, -9.0, 2.0, 4.0];
        assert!((covariance(&array_1, &array_2, 0).unwrap() - array![-0.8, -4.4, 0.0, 1.4, 6.8].sum()/5.0).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_skewness() -> () {
        use crate::statistics::skewness;
        let array_: Array1<f64> = array![1.0, 1.0, 3.0, 4.0, 5.0];
        let numerator: f64 = array!(-5.832, -5.832, 0.008, 1.728, 10.648).mean().unwrap();
        let denominator: f64 = array!(3.24, 3.24, 0.04, 1.44, 4.84_f64).mean().unwrap().powf(3.0/2.0);
        assert!((skewness(&array_).unwrap() - numerator/denominator).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_kurtosis() -> () {
        use crate::statistics::kurtosis;
        let array_: Array1<f64> = array![1.0, 1.0, 3.0, 4.0, 5.0];
        let numerator: f64 = array!(3.24, 3.24, 0.04, 1.44, 4.84_f64).mean().unwrap();
        let denuminator: f64 = array!(3.24, 3.24, 0.04, 1.44, 4.84_f64).mean().unwrap().powi(2);
        assert!((kurtosis(&array_).unwrap() - numerator/denuminator).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_n_choose_r() {
        use crate::statistics::n_choose_r;
        assert_eq!(10, n_choose_r(5, 2).unwrap());
    }

    #[test]
    fn unit_test_linear_regression() -> () {
        use ndarray::{Array1, Array2, array};
        use crate::statistics::linear_regression;
        let y: Array1<f64> = array![1.0, 2.0, 3.0];
        let x: Array2<f64> = array![[1.0, 3.0, 1.0], [4.0, 4.0, 1.0], [6.0, 5.0, 1.0]];
        let params: Array1<f64> = linear_regression(&x, &y).unwrap();
        // The results were found using LinearRegression from sklearn
        let comparison: Array1<f64> = Array1::from(vec![-2.49556592e-16, 1.0, -2.0]);
        assert!((&params - &comparison).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_gamma_function() -> () {
        use crate::utilities::maths_utils::factorial;
        use crate::statistics::gamma_function;
        // Gamma(1)
        assert!((gamma_function(1.0, FunctionEvalMethod::Integral { n_intervals: 1_000_000 }) - 1.0).abs() < TEST_ACCURACY);
        assert!((gamma_function(1.0, FunctionEvalMethod::PowerSeries { n_terms: 20 }) - 1.0).abs() < TEST_ACCURACY);
        // Gamma(3/2)
        let theoretical_result: f64 = std::f64::consts::PI.sqrt() * (factorial(factorial(3 - 2)) as f64) / 2.0_f64.powf((3.0 - 1.0) / 2.0);
        assert!((gamma_function(3.0/2.0, FunctionEvalMethod::Integral { n_intervals: 1_000_000 }) - theoretical_result).abs() < TEST_ACCURACY);
        assert!((gamma_function(3.0/2.0, FunctionEvalMethod::PowerSeries { n_terms: 50 }) - theoretical_result).abs() < 1_000_000.0 * TEST_ACCURACY);
    }


    #[test]
    fn unit_test_upper_incomplete_gamma_function() -> () {
        use crate::statistics::{gamma_function, upper_incomplete_gamma_function};
        // Gamma_{upper}(s, 0) = Gamma(s)
        assert!((upper_incomplete_gamma_function(4.0, 0.0, 1_000_000).unwrap() - gamma_function(4.0, FunctionEvalMethod::Integral { n_intervals: 1_000_000 })).abs() < TEST_ACCURACY);
        // Gamma_{upper}(1, x) = e^{-x}
        assert!((upper_incomplete_gamma_function(1.0, 3.0, 1_000_000).unwrap() - (-3.0_f64).exp()).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_lower_incomplete_gamma_function() -> () {
        use crate::statistics::lower_incomplete_gamma_function;
        // Gamma_{lower}(1, x) = 1 - e^{-x}
        assert!((lower_incomplete_gamma_function(1.0, 3.0, FunctionEvalMethod::Integral { n_intervals: 1_000_000 }).unwrap() - (1.0 - (-3.0_f64).exp())).abs() < 100.0 * TEST_ACCURACY);
        assert!((lower_incomplete_gamma_function(1.0, 3.0, FunctionEvalMethod::PowerSeries { n_terms: 20 }).unwrap() - (1.0 - (-3.0_f64).exp())).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_beta_function() -> () {
        use crate::statistics::beta_function;
        // B(1, x) = 1/x
        assert!((beta_function(1.0, 3.0, 1_000_000).unwrap() - (1.0 / 3.0)).abs() < TEST_ACCURACY);
        // B(x, 1-x) = Pi / sin(Pi*x)
        assert!((beta_function(0.5, 0.5, 100_000_000).unwrap() - std::f64::consts::PI).abs() < 10_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_regularized_incomplete_beta_function() -> () {
        use crate::statistics::regularized_incomplete_beta_function;
        // I_{0}(a, b) = 0
        assert!(regularized_incomplete_beta_function(0.0, 0.2, 0.3, 1_000_000).unwrap() < TEST_ACCURACY);
        // I_{1}(a, b) = 1
        assert!((regularized_incomplete_beta_function(0.99999999999999, 0.2, 0.3, 1_000_000).unwrap() - 1.0 ).abs() < TEST_ACCURACY);
        // I_{x}(a, 1) = x^{a}
        assert!((regularized_incomplete_beta_function(0.5, 2.0, 1.0, 1_000_000).unwrap() - 0.5_f64.powi(2)).abs() < 10_000.0 * TEST_ACCURACY);
        // I_{x}(1, b) = 1 - (1 - x)^{b}
        assert!((regularized_incomplete_beta_function(0.5, 1.0, 3.0, 1_000_000).unwrap() - (1.0 - 0.5_f64.powi(3))).abs() < 10_000.0 * TEST_ACCURACY);
    }
}