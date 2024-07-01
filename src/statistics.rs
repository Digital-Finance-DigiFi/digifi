pub mod continuous_distributions;
pub mod discrete_distributions;


use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use num::complex::Complex;
use crate::utilities::compare_array_len;
use crate::utilities::maths_utils::factorial;


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
    fn pdf(&self, x: &Array1<f64>) -> Array1<f64>;

    /// # Description
    /// Cummulative distribution function (CDF).
    fn cdf(&self, x: &Array1<f64>) -> Array1<f64>;

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
    fn inverse_cdf(&self, p: &Array1<f64>) -> Array1<f64>;
}


/// # Description
/// Measure of joint variability of two random variables.
/// 
/// Cov(X,Y) = E[(X-E[X])(Y-E[Y])]
/// 
/// # Input
/// - array_1: First array
/// - array_2: Second array
/// 
/// # Output
/// - Covariance of two arrays
/// 
/// # Panics
/// - Panics if the lengths of array_1 and array_2 do not match
/// 
/// # LaTeX Formula
/// - Cov(X,Y) = \\frac{1}{n-ddof}\\sum^{n}_{i=1}(x_{i}-\\bar{x})(y_{i}-\\bar{y})
/// where \\bar{x} = \frac{1}{n}\\sum^{n}_{i=1}x_{i}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Covariance
/// - Original Source: N/A
pub fn covariance(array_1: &Array1<f64>, array_2: &Array1<f64>, ddof: usize) -> f64 {
    compare_array_len(array_1, array_2, "array_1", "array_2");
    let x: Array1<f64> = array_1 - array_1.mean().expect("Mean of array_1 is not computed.");
    let y: Array1<f64> = array_2 - array_2.mean().expect("Mean of array_2 is not computed.");
    (&x * &y).sum() / (x.len() - ddof) as f64
}


/// # Description
/// Calculates the skewness of a given data array. Skewness measures the asymmetry of the probability distribution of a real-valued random variable about its mean.
/// 
/// The skewness value can be positive, zero, negative, or undefined.
/// 
/// Fisher-Pearson coefficient of skewness is used.
/// 
/// # Input
/// - array: An array for which the skewness is to be calculated
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
pub fn skewness(array: &Array1<f64>) -> f64 {
    let difference: Array1<f64> = array - array.mean().expect("Mean of array is not computed.");
    let numerator: f64 = difference.map(| i | i.powi(3)).mean().expect("Mean of difference array is not computed.");
    let denominator: f64 = difference.map(| i | i.powi(2)).mean().expect("Mean of difference array is not computed.").powf(3.0/2.0);
    if denominator == 0.0 {
        f64::NAN
    } else {
        numerator / denominator
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
/// - array: An array for which the kurtosis is to be calculated
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
pub fn kurtosis(array: &Array1<f64>) -> f64 {
    let difference: Array1<f64> = array - array.mean().expect("Mean of array is not computed.");
    let numerator: f64 = difference.map(| i | i.powi(2)).mean().expect("Mean of difference array is not computed.");
    let denominator: f64 = difference.map(| i | i.powi(2)).mean().expect("Mean of difference array is not computed.").powi(2);
    if denominator == 0.0 {
        f64::NAN
    } else {
        numerator / denominator
    }
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


/// # Description
/// The ratio of the systematic variance to the total variance.
/// 
/// # Input
/// - real_values: Array of observed or empirical data
/// - predicted_values: Array data produced by the model
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
pub fn r_square(real_values: &Array1<f64>, predicted_values: &Array1<f64>) -> f64 {
    compare_array_len(&real_values, &predicted_values, "real_values", "predicted_values");
    let residual_sum: f64 = (real_values - predicted_values).map(|x| x.powi(2)).sum();
    let total_sum: f64 = (real_values - real_values.mean().expect("Could not compute the mean of real_values.")).map(|x| x.powi(2)).sum();
    1.0 - residual_sum/total_sum
}


/// # Description
/// Adjusted R-square for the upward bias in the R-square due to estimated values of the parameters used.
/// 
/// # Input
/// - real_values: Array of observed or empirical data
/// - predicted_values: Array data produced by the model
/// - sample_size: Number of points used in the model
/// - k_variables: Number of variables in the model
/// 
/// # Output
/// - adjusted_r_square: R-square adjusted for upward estimation bias
/// 
/// # LaTeX Formula
/// - R^{2}_{A} = 1 - (1-R^{2})\\frac{n-1}{n-k-1}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
/// - Original Source: N/A
pub fn adjusted_r_square(real_values: &Array1<f64>, predicted_values: &Array1<f64>, sample_size: usize, k_variables: usize) -> f64 {
    1.0 - (1.0-r_square(real_values, predicted_values)) * (sample_size as f64 - 1.0) / ((sample_size-k_variables) as f64 - 1.0)
}


#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_covariance() -> () {
        use crate::statistics::covariance;
        let array_1: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let array_2: Array1<f64> = array![1.0, 5.0, -9.0, 2.0, 4.0];
        assert!((covariance(&array_1, &array_2, 0) - array![-0.8, -4.4, 0.0, 1.4, 6.8].sum()/5.0).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_skewness() -> () {
        use crate::statistics::skewness;
        let array_: Array1<f64> = array![1.0, 1.0, 3.0, 4.0, 5.0];
        let numerator: f64 = array!(-5.832, -5.832, 0.008, 1.728, 10.648).mean().unwrap();
        let denominator: f64 = array!(3.24, 3.24, 0.04, 1.44, 4.84_f64).mean().unwrap().powf(3.0/2.0);
        assert!((skewness(&array_) - numerator/denominator).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_kurtosis() -> () {
        use crate::statistics::kurtosis;
        let array_: Array1<f64> = array![1.0, 1.0, 3.0, 4.0, 5.0];
        let numerator: f64 = array!(3.24, 3.24, 0.04, 1.44, 4.84_f64).mean().unwrap();
        let denuminator: f64 = array!(3.24, 3.24, 0.04, 1.44, 4.84_f64).mean().unwrap().powi(2);
        assert!((kurtosis(&array_) - numerator/denuminator).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_n_choose_r() {
        use crate::statistics::n_choose_r;
        assert_eq!(10, n_choose_r(5, 2));
    }

    #[test]
    fn unit_test_linear_regression() -> () {
        use ndarray::{Array1, Array2, array};
        use crate::statistics::linear_regression;
        let y: Array1<f64> = array![1.0, 2.0, 3.0];
        let x: Array2<f64> = array![[1.0, 3.0, 1.0], [4.0, 4.0, 1.0], [6.0, 5.0, 1.0]];
        let params = linear_regression(&x, &y);
        // The results were found using LinearRegression from sklearn
        let comparison: Array1<f64> = Array1::from(vec![-2.49556592e-16, 1.0, -2.0]);
        assert!((&params - &comparison).sum().abs() < TEST_ACCURACY);
    }
}