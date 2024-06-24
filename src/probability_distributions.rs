use ndarray::Array1;
use crate::utilities::compare_array_len;


pub enum ProbabilityDistributionType {
    Discrete,
    Continuous,
}


pub trait ProbabilityDistribution {
    /// # Description
    /// Probability density function.
    fn pdf() -> Array1<f64>;

    /// # Description
    /// Cummulative distribution function (CDF).
    fn cdf() -> Array1<f64>;

    /// # Description
    /// Moment generating function (MGF).
    fn mgf() -> Array1<f64>;

    /// # Description
    /// Characteristic function (CF).
    /// 
    /// Characteristic function is the Fourier transform of the PDF.
    fn cf() -> Array1<f64>;

    /// # Description
    /// Inverse cumulative distribution function (CDF), else knwon as quantile function.
    fn inverse_cdf() -> Array1<f64>;
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


#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_skewness() -> () {
        use crate::probability_distributions::skewness;
        let array_: Array1<f64> = array![1.0, 1.0, 3.0, 4.0, 5.0];
        let numerator: f64 = array!(-5.832, -5.832, 0.008, 1.728, 10.648).mean().unwrap();
        let denominator: f64 = array!(3.24, 3.24, 0.04, 1.44, 4.84_f64).mean().unwrap().powf(3.0/2.0);
        assert!((skewness(&array_) - numerator/denominator).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_kurtosis() -> () {
        use crate::probability_distributions::kurtosis;
        let array_: Array1<f64> = array![1.0, 1.0, 3.0, 4.0, 5.0];
        let numerator: f64 = array!(3.24, 3.24, 0.04, 1.44, 4.84_f64).mean().unwrap();
        let denuminator: f64 = array!(3.24, 3.24, 0.04, 1.44, 4.84_f64).mean().unwrap().powi(2);
        assert!((kurtosis(&array_) - numerator/denuminator).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_covariance() -> () {
        use crate::probability_distributions::covariance;
        let array_1: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let array_2: Array1<f64> = array![1.0, 5.0, -9.0, 2.0, 4.0];
        assert!((covariance(&array_1, &array_2, 0) - array![-0.8, -4.4, 0.0, 1.4, 6.8].sum()/5.0).abs() < TEST_ACCURACY);
    }
}