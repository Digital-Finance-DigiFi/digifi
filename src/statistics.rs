//! # Statistics
//! 
//! Contains continuous and discrete probability distributions that are commonly used in finance, along with common statistical functions such as beta
//! and gamma functions, tests for statistical significance, and a tool for performing a linear regression analysis.


// Re-Exports
pub use self::gamma::{ln_gamma, gamma, lower_incomplete_gamma, upper_incomplete_gamma, digamma};
pub use self::beta::{ln_beta, beta, incomplete_beta, regularized_incomplete_beta, multivariate_beta};
pub use self::discrete_distributions::{BernoulliDistribution, BinomialDistribution, DiscreteUniformDistribution, PoissonDistribution};
pub use self::continuous_distributions::{
    ContinuousUniformDistribution, NormalDistribution, ExponentialDistribution, LaplaceDistribution, GammaDistribution, StudentsTDistribution,
    ParetoDistribution, LogNormalDistribution,
};
pub use self::stat_tests::{ConfidenceLevel, ADFType, ADFResult, adf, CointegrationResult, cointegration, TTestResult, TTestTwoSampleCase, t_test_two_sample, t_test_lr};
pub use self::linear_regression_analysis::{LinearRegressionFeatureResult, LinearRegressionResult, LinearRegressionSettings, LinearRegressionAnalysis};
pub use self::mediation_analysis::{
    BKMediationAnalysisStep, BKMediationAnalysisFinalStep, BKMediationAnalysisResult, BaronKennyMeriationAnalysis, SobelTestResult, sobel_test,
    MJYAnalysisStep, MJYAnalysisFinalStep, MJYAnalysisResult, MullerJuddYzerbytAnalysis,
};


mod gamma;
mod beta;
pub mod continuous_distributions;
pub mod discrete_distributions;
pub mod stat_tests;
mod linear_regression_analysis;
pub mod mediation_analysis;


use std::borrow::Borrow;
use ndarray::{Array1, Array2};
use nalgebra::DMatrix;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::{
    compare_len, MatrixConversion, FeatureCollection,
    maths_utils::factorial,
    loss_functions::{LossFunction, SSE},
};


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ProbabilityDistributionType {
    Discrete,
    Continuous,
}


/// Trait that describes all the available methods for a probability distribution.
pub trait ProbabilityDistribution {
    /// Type of probability distribution.
    fn distribution_type() -> ProbabilityDistributionType;

    /// Mean.
    fn mean(&self) -> f64;

    /// Median.
    fn median(&self) -> Vec<f64>;

    /// Mode.
    fn mode(&self) -> Vec<f64>;

    /// Variance.
    fn variance(&self) -> f64;

    /// Skewness.
    fn skewness(&self) -> f64;

    /// Excess kurtosis.
    fn excess_kurtosis(&self) -> f64;

    /// Entropy.
    fn entropy(&self) -> Result<f64, DigiFiError>;

    /// Probability density function.
    fn pdf(&self, x: f64) -> Result<f64, DigiFiError>;

    /// Cumulative distribution function (CDF).
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError>;

    /// Inverse cumulative distribution function (CDF), else knwon as quantile function.
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError>;

    /// Moment generating function (MGF).
    fn mgf(&self, t: usize) -> f64;

    /// Probability density function (PDF) applied to the array of values.
    fn pdf_iter<T, I>(&self, x: T) -> Result<Array1<f64>, DigiFiError>
    where
        T: Iterator<Item = I> + ExactSizeIterator,
        I: Borrow<f64>,
    {
        let mut p: Vec<f64> = Vec::with_capacity(x.len());
        for i in x {
            p.push(self.pdf(*i.borrow())?);
        }
        Ok(Array1::from_vec(p))
    }

    /// Cumulative distribution function (CDF) applied to the array of values.
    fn cdf_iter<T, I>(&self, x: T) -> Result<Array1<f64>, DigiFiError>
    where
        T: Iterator<Item = I> + ExactSizeIterator,
        I: Borrow<f64>,
    {
        let mut p: Vec<f64> = Vec::with_capacity(x.len());
        for i in x {
            p.push(self.cdf(*i.borrow())?);
        }
        Ok(Array1::from_vec(p))
    }

    /// Inverse cumulative distribution function (CDF) applied to the array of values.
    fn inverse_cdf_iter<T, I>(&self, p: T) -> Result<Array1<f64>, DigiFiError>
    where
        T: Iterator<Item = I> + ExactSizeIterator,
        I: Borrow<f64>,
    {
        let mut xs: Vec<f64> = Vec::with_capacity(p.len());
        for i in p {
            xs.push(self.inverse_cdf(*i.borrow())?);
        }
        Ok(Array1::from_vec(xs))
    }

    /// Moment generating function (MGF) applied to the array of values.
    fn mgf_iter<T, I>(&self, t: T) -> Result<Array1<f64>, DigiFiError>
    where
        T: Iterator<Item = I>,
        I: Borrow<usize>
    {
        let ts: Vec<f64> = t.map(|t| self.mgf(*t.borrow()) ).collect();
        Ok(Array1::from_vec(ts))
    }
}


/// This trait contains methods applicable to using probability distributions as risk measures.
pub trait RiskMeasure: ProbabilityDistribution + ErrorTitle {
    fn validate_alpha(&self, alpha: f64) -> Result<(), DigiFiError> {
        if (alpha < 0.0) || (1.0 < alpha) {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `alpha` must be in the range `[0, 1]`.".to_owned(),
            });
        }
        Ok(())
    }

    /// Measure of the risk of a portfolio estimating how much a portfolio can lose in a specified period.
    /// 
    /// Note: This function uses the convention where 95% V@R of $1 million means that $1 million is the maximum possible loss
    /// in the specified time horizon after excluding all worse outcomes whose combined probability is at most 5%.
    /// 
    /// Note: This V@R implementation assumes that the distribution provided is the distribution of losses (i.e., the positive numbers are losses,
    /// negative numbers are profits).
    /// 
    /// Note: The V@R is quoted as a positive number, if V@R is negative it implies that the portfolio has very high chance of making a profit.
    /// 
    /// # Input
    /// - `alpha`: Probability level for V@R
    /// - `losses_distribution`: Probability distribution object with an inverse CDF method
    /// 
    /// # Output
    /// - Value at risk (V@R)
    /// 
    /// # Errors
    /// - Returns an error if the argument `alpha` is not in the range \[0, 1\].
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Value_at_risk#>
    /// - Original Source: N/A
    fn value_at_risk(&self, alpha: f64) -> Result<f64, DigiFiError> {
        self.validate_alpha(alpha)?;
        Ok(self.inverse_cdf(alpha)?)
    }

    /// Measure of the risk of a portfolio that evaluates the expected return of a portfolio in the worst percentage of cases.
    /// 
    /// Note: This function uses the convention where ES at 95% is the expected shortfall of the 5% of worst cases.
    /// 
    /// # Input
    /// - `alpha`: Probability level for ES
    /// 
    /// # Output
    /// - Expected shortfall (ES)
    /// 
    /// # Errors
    /// - Returns an error if the argument `alpha` is not in the range \[0, 1\].
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Expected_shortfall>
    /// - Original Source: N/A
    fn expected_shortfall(&self, alpha: f64) -> Result<f64, DigiFiError>;

    fn value_at_risk_iter<T, I>(&self, alphas: T) -> Result<Array1<f64>, DigiFiError>
    where
        T: Iterator<Item = I>,
        I: Borrow<f64>,
    {
        alphas.map(|alpha| self.value_at_risk(*alpha.borrow()) ).collect()
    }

    fn expected_shortfall_iter<T, I>(&self, alphas: T) -> Result<Array1<f64>, DigiFiError>
    where
        T: Iterator<Item = I>,
        I: Borrow<f64>,
    {
        alphas.map(|alpha| self.expected_shortfall(*alpha.borrow()) ).collect()
    }
}


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
    compare_len(&array_1.iter(), &array_2.iter(), "array_1", "array_2")?;
    let array_1_mean: f64 = match array_1.mean() {
        Some(mean) => mean,
        None => return Err(DigiFiError::MeanCalculation { title: error_title, series: "array_1".to_owned(), }),
    };
    let array_2_mean: f64 = match array_2.mean() {
        Some(v) => v,
        None => return Err(DigiFiError::MeanCalculation { title: error_title, series: "array_2".to_owned(), })
    };
    Ok(
        array_1.iter()
            .zip(array_2.iter())
            .fold(0.0, |total, (x, y)| total + (x - array_1_mean) * (y - array_2_mean) ) / (array_1.len() - ddof) as f64
    )
}


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
    if array.len() <= 1 {
        return Err(DigiFiError::ValidationError { title: error_title, details: "The length of `array` must be at least `2`.".to_owned(), });
    }
    let mean: f64 = match array.mean() {
        Some(v) => v,
        None => return Err(DigiFiError::MeanCalculation { title: error_title, series: "array".to_owned(), }),
    };
    let (num, denom) = array.iter().fold((0.0, 0.0),  |(num, denom), v| (num + (v - mean).powi(3), denom + (v - mean).powi(2)) );
    let array_len: f64 = array.len() as f64;
    Ok((num / array_len) / (denom / array_len).powf(3.0/2.0))
}


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
    if array.len() <= 1 {
        return Err(DigiFiError::ValidationError { title: error_title, details: "The length of `array` must be at least `2`.".to_owned(), });
    }
    let mean: f64 = match array.mean() {
        Some(v) => v,
        None => return Err(DigiFiError::MeanCalculation { title: error_title, series: "array".to_owned(), }),
    };
    let (num, denom) = array.iter().fold((0.0, 0.0), |(num, denom), v| (num + (v - mean).powi(2), denom + (v - mean).powi(2)) );
    let array_len: f64 = array.len() as f64;
    Ok((num / array_len) / (denom / array_len).powi(2))
}


/// Changes the period for which the standard deviation applies by multiplying standard deviation by factor `sqrt(n)`.
/// 
/// Note: This is useful when you want to obtain estimated standard deviation for large period of time from data that is high frequency
/// (e.g., standard deviation of daily returns can be annualized by settings `n=252`)
/// 
/// # Input
/// - `std`: Standard deviation of data with higher frequency
/// - `n`: Number of data points in the desired re-sampled standard deviation.
/// 
/// # Output
/// - Standard deviation of returns over a certain period
pub fn change_frequency_std(std: f64, n: f64) -> f64 {
    std * n.sqrt()
}


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
    let ddof: f64 = ddof as f64;
    Ok(cov / (array_1.std(ddof) * array_2.std(ddof)))
}


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
        return Err(DigiFiError::ParameterConstraint {
            title: "n Choose r".to_owned(),
            constraint: "The value of variable n must be larger or equal to the value of variable r.".to_owned(),
        });
    }
    Ok(factorial(n) / (factorial(n - r) * factorial(r)))
}


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
    let n_square_matrix: DMatrix<f64> = MatrixConversion::ndarray_to_nalgebra(square_matrix);
    let n_inv_matrix: DMatrix<f64> = n_square_matrix.try_inverse()
        .ok_or(DigiFiError::Other { title: "Linear Regression".to_owned(), details: "No matrix inverse exists to perform linear regression.".to_owned(), })?;
    let inv_matrix: Array2<f64> = MatrixConversion::nalgebra_to_ndarray(n_inv_matrix)?;
    Ok(inv_matrix.dot(&x.t().dot(&y.t())))
}


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
    compare_len(&x.iter(), &y_prediction.iter(), "x", "y_prediction")?;
    compare_len(&x.iter(), &y.iter(), "x", "y")?;
    let loss_function: SSE = SSE;
    let denominator: f64 = match y.len().checked_sub(ddof) {
        Some(v) => v as f64, // Number of data points (degrees of freedom) minus constrained degrees of freedom by the model (i.e., number of features)
        None => return Err(DigiFiError::Other { title: error_title, details: "There are fewer data points in `y` array than `ddof`.".to_owned(), }),
    };
    let estimated_var: f64 = loss_function.loss_iter(y.iter(), y_prediction.iter())? / denominator;
    let mean_of_x: f64 = x.mean().ok_or(DigiFiError::MeanCalculation { title: error_title, series: "x".to_owned(), })?;
    let estimated_var_beta_denominator: f64 = x.map(|v| (v - mean_of_x).powi(2) ).sum();
    Ok((estimated_var / estimated_var_beta_denominator).sqrt())
}


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
    compare_len(&real_values.iter(), &predicted_values.iter(), "real_values", "predicted_values")?;
    let real_values_mean: f64 = real_values.mean()
        .ok_or(DigiFiError::MeanCalculation { title: "R Square".to_owned(), series: "real_values".to_owned(), })?;
    let (residual_sum, total_sum) = real_values.iter().zip(predicted_values.iter())
        .fold((0.0, 0.0), |(residual_sum, total_sum), (real, predicted)| {
            (residual_sum + (real - predicted).powi(2), total_sum + (real - real_values_mean).powi(2))
        } );
    Ok(1.0 - residual_sum / total_sum)
}


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
    Ok(1.0 - (1.0-r_squared(real_values, predicted_values)?) * (sample_size as f64 - 1.0) / ((sample_size - k_variables) as f64 - 1.0))
}


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
pub fn variance_inflation_factor(xis: &mut FeatureCollection, xi: &Array1<f64>) -> Result<Option<f64>, DigiFiError> {
    xis.add_constant = true;
    let mut settings: LinearRegressionSettings = LinearRegressionSettings::disable_all();
    settings.enable_r_squared = true;
    let lr: LinearRegressionAnalysis = LinearRegressionAnalysis::new(settings);
    let lr_result: LinearRegressionResult = lr.run(xis, xi)?;
    match lr_result.r_squared {
        Some(r) if r != 1.0 => Ok(Some(1.0 / (1.0 - r))),
        _ => Ok(None),
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