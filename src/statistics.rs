// Re-Exports
pub use self::gamma::{ln_gamma, gamma, lower_incomplete_gamma, upper_incomplete_gamma, digamma};
pub use self::beta::{ln_beta, beta, incomplete_beta, regularized_incomplete_beta, multivariate_beta};
pub use self::discrete_distributions::{BernoulliDistribution, BinomialDistribution, DiscreteUniformDistribution, PoissonDistribution};
pub use self::continuous_distributions::{
    ContinuousUniformDistribution, NormalDistribution, ExponentialDistribution, LaplaceDistribution, GammaDistribution, StudentsTDistribution,
};
pub use self::stat_tests::{ConfidenceLevel, ADFType, ADFResult, adf, CointegrationResult, cointegration, TTestResult, t_test_two_sample, t_test_lr};
pub use self::linear_regression_analysis::{LinearRegressionFeatureResult, LinearRegressionResult, LinearRegressionSettings, LinearRegressionAnalysis};


mod gamma;
mod beta;
pub mod continuous_distributions;
pub mod discrete_distributions;
pub mod stat_tests;
mod linear_regression_analysis;


use ndarray::{Array1, Array2};
use nalgebra::DMatrix;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;
use crate::utilities::{compare_array_len, MatrixConversion};
use crate::utilities::maths_utils::factorial;
use crate::utilities::loss_functions::{LossFunction, SSE};


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ProbabilityDistributionType {
    Discrete,
    Continuous,
}


/// # Description
/// Trait that describes all the available methods for a probability distribution.
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
    fn entropy(&self) -> Result<f64, DigiFiError>;

    /// # Description
    /// Probability density function.
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError>;

    /// # Description
    /// Cummulative distribution function (CDF).
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError>;

    /// # Description
    /// Moment generating function (MGF).
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64>;

    /// # Description
    /// Inverse cumulative distribution function (CDF), else knwon as quantile function.
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, DigiFiError>;
}


/// # Description
/// Measure of joint variability of two random variables.
/// 
/// Cov(X,Y) = E\[(X-E\[X\])(Y-E\[Y\])\]
/// 
/// # Input
/// - `array_1`: First array
/// - `array_2`: Second array
/// - `ddof`: Delta degrees of freedom (i.e., degrees of freedom = N - ddof)
/// 
/// # Output
/// - Covariance of two arrays
/// 
/// # Errors
/// - Returns an error if the lengths of array_1 and array_2 do not match.
/// 
/// # LaTeX Formula
/// - Cov(X,Y) = \\frac{1}{n-ddof}\\sum^{n}_{i=1}(x_{i}-\\bar{x})(y_{i}-\\bar{y})
/// where \\bar{x} = \frac{1}{n}\\sum^{n}_{i=1}x_{i}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Covariance>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, array};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::covariance;
///
/// let array_1: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let array_2: Array1<f64> = array![1.0, 5.0, -9.0, 2.0, 4.0];
///
/// assert!((covariance(&array_1, &array_2, 0).unwrap() - array![-0.8, -4.4, 0.0, 1.4, 6.8].sum()/5.0).abs() < TEST_ACCURACY);
/// ```
pub fn covariance(array_1: &Array1<f64>, array_2: &Array1<f64>, ddof: usize) -> Result<f64, DigiFiError> {
    let error_title: String = String::from("Covariance");
    compare_array_len(array_1, array_2, "array_1", "array_2")?;
    let x: Array1<f64> = array_1 - array_1.mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "array_1".to_owned(), })?;
    let y: Array1<f64> = array_2 - array_2.mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "array_2".to_owned(), })?;
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
/// - Wikipedia: <https://en.wikipedia.org/wiki/Skewness>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, array};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::skewness;
///
/// let array_: Array1<f64> = array![1.0, 1.0, 3.0, 4.0, 5.0];
/// let numerator: f64 = array!(-5.832, -5.832, 0.008, 1.728, 10.648).mean().unwrap();
/// let denominator: f64 = array!(3.24, 3.24, 0.04, 1.44, 4.84_f64).mean().unwrap().powf(3.0/2.0);
///
/// assert!((skewness(&array_).unwrap() - numerator/denominator).abs() < TEST_ACCURACY);
/// ```
pub fn skewness(array: &Array1<f64>) -> Result<f64, DigiFiError> {
    let error_title: String = String::from("Skewness");
    let difference: Array1<f64> = array - array.mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "array".to_owned(), })?;
    let numerator: f64 = difference.map(| i | i.powi(3)).mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "difference array".to_owned(), })?;
    let denominator: f64 = difference.map(| i | i.powi(2)).mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "difference array".to_owned(), })?.powf(3.0/2.0);
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
/// - Wikipedia: <https://en.wikipedia.org/wiki/Kurtosis>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, array};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::kurtosis;
///
/// let array_: Array1<f64> = array![1.0, 1.0, 3.0, 4.0, 5.0];
/// let numerator: f64 = array!(3.24, 3.24, 0.04, 1.44, 4.84_f64).mean().unwrap();
/// let denuminator: f64 = array!(3.24, 3.24, 0.04, 1.44, 4.84_f64).mean().unwrap().powi(2);
///
/// assert!((kurtosis(&array_).unwrap() - numerator/denuminator).abs() < TEST_ACCURACY);
/// ```
pub fn kurtosis(array: &Array1<f64>) -> Result<f64, DigiFiError> {
    let error_title: String = String::from("Kurtosis");
    let difference: Array1<f64> = array - array.mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "array".to_owned(), })?;
    let numerator: f64 = difference.map(| i | i.powi(2)).mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "difference array".to_owned(), })?;
    let denominator: f64 = difference.map(| i | i.powi(2)).mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "difference array".to_owned(), })?.powi(2);
    if denominator == 0.0 {
        Ok(f64::NAN)
    } else {
        Ok(numerator / denominator)
    }
}


/// # Description
/// Pearson correlation coefficient. Pearson correlation is a measure of linear correlation between two arrays.
///
/// # Input
/// - `array_1`: Array of data
/// - `array_2`: Array of data
///
/// # Output
// - Pearson correlation coefficient
///
/// # Errors
/// - Returns an error if the lengths of `array_1` and `array_2` do not match.
///
/// # LaTeX Formula
/// - \\rho(X,Y) = \\frac{cov(X,Y)}{\\sigma_{X}\\sigma_{Y}}
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>
/// - Original Source: <https://books.google.com/books?id=60aL0zlT-90C&pg=PA240>
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::pearson_correlation;
///
/// let array_1: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let array_2: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// assert!((pearson_correlation(&array_1, &array_2, 1).unwrap() - 1.0).abs() < TEST_ACCURACY);
///
/// let array_1: Array1<f64> = Array1::from_vec(vec![1.0, -1.0]);
/// let array_2: Array1<f64> = Array1::from_vec(vec![-1.0, 1.0]);
/// assert!((pearson_correlation(&array_1, &array_2, 1).unwrap() + 1.0).abs() < TEST_ACCURACY);
/// ```
pub fn pearson_correlation(array_1: &Array1<f64>, array_2: &Array1<f64>, ddof: usize) -> Result<f64, DigiFiError> {
    let cov: f64 = covariance(array_1, array_2, ddof)?;
    let sigma_1: f64 = covariance(array_1, array_1, ddof)?.sqrt();
    let sigma_2: f64 = covariance(array_2, array_2, ddof)?.sqrt();
    Ok(cov / (sigma_1 * sigma_2))
}


/// # Description
/// nCr:  n choose r
/// 
/// # Input
/// - `n`: Power of the binomial expansion
/// - `r`: Number of successes
/// 
/// # Errors
/// - Returns an error if the value of `n` is larger than `r`.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Binomial_coefficient>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::statistics::n_choose_r;
///
/// assert_eq!(10, n_choose_r(5, 2).unwrap());
/// ```
pub fn n_choose_r(n: u128, r: u128) -> Result<u128, DigiFiError> {
    if n < r {
        return Err(DigiFiError::ParameterConstraint { title: "n Choose r".to_owned(), constraint: "The value of variable n must be larger or equal to the value of variable r.".to_owned(), });
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
/// - Returns an error if the length of matrix `x` does not match the length of vector `y`.
/// 
/// # LaTeX Formula
/// - y = X\\cdot\\beta
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Linear_regression>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, Array2, array};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::linear_regression;
///
/// let y: Array1<f64> = array![1.0, 2.0, 3.0];
/// let x: Array2<f64> = array![[1.0, 3.0, 1.0], [4.0, 4.0, 1.0], [6.0, 5.0, 1.0]];
/// let params: Array1<f64> = linear_regression(&x, &y).unwrap();
///
/// // The results were found using LinearRegression from sklearn
/// let results: Array1<f64> = Array1::from(vec![-2.49556592e-16, 1.0, -2.0]);
/// assert!((&params - &results).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub fn linear_regression(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
    if x.dim().0 != y.len() {
        return Err(DigiFiError::UnmatchingLength { array_1: "x".to_owned(), array_2: "y".to_owned(), });
    }
    let square_matrix: Array2<f64> = x.t().dot(x);
    // Matrix inverse is done via nalgebra.
    let n_square_matrix: DMatrix<f64> = MatrixConversion::ndarray_to_nalgebra(&square_matrix);
    let n_inv_matrix: DMatrix<f64> = n_square_matrix.try_inverse()
        .ok_or(DigiFiError::Other { title: "Linear Regression".to_owned(), details: "No matrix inverse exists to perform linear regression.".to_owned(), })?;
    let inv_matrix: Array2<f64> = MatrixConversion::nalgebra_to_ndarray(&n_inv_matrix)?;
    Ok(inv_matrix.dot(&x.t().dot(&y.t())))
}


/// # Description
/// Standard error of linear regression coefficient, which is used for different statistical tests.
/// 
/// # Input
/// - `y`: Observed response values
/// - `y_prediction`: Values predicted by the linear regression
/// - `x`: Feature in linear regression whose coefficient's standard error is being computed
/// - `ddof`: Delta degrees of freedom (i.e., degrees of freedom = N - ddof)
/// 
/// # Output
/// - Standard error of the linear regression coefficient
/// 
/// # Errors
/// - Returns an error if the length of vector `x` does not match the length of vector `y_prediction`.
/// - Returns an error if the length of vector `x` does not match the length of vector `y`.
/// - Returns an error if there are fewed data points in `y` than `ddof`.
/// 
/// # LaTeX Formula
/// - s_{\\hat{\\beta}} = sqrt{\\frc{\\frac{1}{n-ddof}\\sum^{n}\_{i=1}\\hat{epsilon}^{2}_{i}}{\\sum^{n}\_{i=1}(x_{i} - \\bar{x})^{2}}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Simple_linear_regression#Normality_assumption>
/// - Original Source: N/A
pub fn se_lr_coefficient(y: &Array1<f64>, y_prediction: &Array1<f64>, x: &Array1<f64>, ddof: usize) -> Result<f64, DigiFiError> {
    let error_title: String = String::from("Standard Error of Linear Regression Coefficient");
    compare_array_len(x, y_prediction, "x", "y_prediction")?;
    compare_array_len(x, y, "x", "y")?;
    let loss_function: SSE = SSE;
    let denominator: f64 = match y.len().checked_sub(ddof) {
        Some(v) => v as f64, // Number of data points (degrees of freedom) minus constrained degrees of freedom by the model (i.e., number of features)
        None => return Err(DigiFiError::Other { title: error_title, details: "There are fewer data points in `y` array than `ddof`.".to_owned(), }),
    };
    let estimated_var: f64 = loss_function.loss_array(y, y_prediction)? / denominator;
    let mean_of_x: f64 = x.mean().ok_or(DigiFiError::MeanCalculation { title: error_title, series: "x".to_owned(), })?;
    let estimated_var_beta_denominator: f64 = x.map(|v| (v - mean_of_x).powi(2) ).sum();
    Ok((estimated_var / estimated_var_beta_denominator).sqrt())
}


/// # Description
/// Coefficient of determination, the ratio of the systematic variance to the total variance.
/// 
/// # Input
/// - `real_values`: Array of observed or empirical data
/// - `predicted_values`: Array data produced by the model
/// 
/// # Output
/// - R-squared: Coefficient of determination, the ratio of exaplained variance to all variance
/// 
/// # LaTeX Formula
/// - R^{2} = 1 - \\frac{\\sum_{i}(y_{i}-f_{i})^{2}}{\\sum_{i}(y_{i}-\\bar{y})^{2}}
/// where ym is the array of real values and f is the array of predicted values.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Coefficient_of_determination>
/// - Original Source: N/A
pub fn r_squared(real_values: &Array1<f64>, predicted_values: &Array1<f64>) -> Result<f64, DigiFiError> {
    compare_array_len(&real_values, &predicted_values, "real_values", "predicted_values")?;
    let residual_sum: f64 = (real_values - predicted_values).map(|x| x.powi(2)).sum();
    let total_sum: f64 = (real_values - real_values.mean()
        .ok_or(DigiFiError::MeanCalculation { title: "R Square".to_owned(), series: "real_values".to_owned(), })?).map(|x| x.powi(2)).sum();
    Ok(1.0 - residual_sum/total_sum)
}


/// # Description
/// Adjusted R-squared for the upward bias in the R-squared due to estimated values of the parameters used.
/// 
/// # Input
/// - `real_values`: Array of observed or empirical data
/// - `predicted_values`: Array data produced by the model
/// - `sample_size`: Number of points used in the model
/// - `k_variables`: Number of variables in the model
/// 
/// # Output
/// - R-squared adjusted for upward estimation bias
/// 
/// # LaTeX Formula
/// - R^{2}_{A} = 1 - (1-R^{2})\\frac{n-1}{n-k-1}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2>
/// - Original Source: N/A
pub fn adjusted_r_squared(real_values: &Array1<f64>, predicted_values: &Array1<f64>, sample_size: usize, k_variables: usize) -> Result<f64, DigiFiError> {
    Ok(1.0 - (1.0-r_squared(real_values, predicted_values)?) * (sample_size as f64 - 1.0) / ((sample_size-k_variables) as f64 - 1.0))
}


/// # Description
/// Ratio (quotient) of the variance of a parameter estimate when fitting a full model that includes other parameters to the variance of the parameter estimate
/// if the model is fit with only the parameter on its own.
/// 
/// # Input
/// - `xi`: Feature of the model whose VIF is being computed
/// - `xis`: Other features of the model, excluding `xi`
/// 
/// # Ouput
/// - Variance inflation Factor (VIF)
/// 
/// # LaTeX Formula
/// - VIF_{i} = \\frac{1}{1 - R^{2}_{i}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Variance_inflation_factor>
/// - Original Source: N/A
pub fn variance_inflation_factor(xis: &Vec<Array1<f64>>, xi: &Array1<f64>) -> Result<Option<f64>, DigiFiError> {
    let mut settings: LinearRegressionSettings = LinearRegressionSettings::disable_all();
    settings.add_constant = true;
    settings.enable_r_squared = true;
    let lr: LinearRegressionAnalysis = LinearRegressionAnalysis::new(settings);
    let lr_result: LinearRegressionResult = lr.run(xis, xi)?;
    match lr_result.r_squared {
        Some(r) => {
            if r == 1.0 { return Ok(None) } else { Ok(Some(1.0 / (1.0 - r))) }
        },
        None => Ok(None),
    }
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
    fn unit_test_pearson_correlation() -> () {
        use crate::statistics::pearson_correlation;
        let array_1: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let array_2: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!((pearson_correlation(&array_1, &array_2, 1).unwrap() - 1.0).abs() < TEST_ACCURACY);
        let array_1: Array1<f64> = Array1::from_vec(vec![1.0, -1.0]);
        let array_2: Array1<f64> = Array1::from_vec(vec![-1.0, 1.0]);
        assert!((pearson_correlation(&array_1, &array_2, 1).unwrap() + 1.0).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_n_choose_r() {
        use crate::statistics::n_choose_r;
        assert_eq!(10, n_choose_r(5, 2).unwrap());
    }

    #[test]
    fn unit_test_linear_regression() -> () {
        use ndarray::Array2;
        use crate::statistics::linear_regression;
        let y: Array1<f64> = array![1.0, 2.0, 3.0];
        let x: Array2<f64> = array![[1.0, 3.0, 1.0], [4.0, 4.0, 1.0], [6.0, 5.0, 1.0]];
        let params: Array1<f64> = linear_regression(&x, &y).unwrap();
        // The results were found using LinearRegression from sklearn
        let results: Array1<f64> = Array1::from(vec![-2.49556592e-16, 1.0, -2.0]);
        assert!((&params - &results).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }
}