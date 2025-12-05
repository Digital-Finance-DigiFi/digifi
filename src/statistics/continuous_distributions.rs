#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::statistics::inverse_regularized_incomplete_beta;
use crate::utilities::{maths_utils::{erf, erfinv, derivative}, numerical_engines::nelder_mead};
use crate::statistics::{
    ProbabilityDistributionType, ProbabilityDistribution, RiskMeasure,
    gamma::{ln_gamma, gamma, lower_incomplete_gamma, digamma},
    beta::{beta, regularized_incomplete_beta},
};


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Methods and properties of continuous uniform distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Continuous_uniform_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, ContinuousUniformDistribution};
///
/// let dist: ContinuousUniformDistribution = ContinuousUniformDistribution::build(0.0, 1.0).unwrap();
/// let x: f64 = 0.6;
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
/// assert!((pdf_v - 1.0).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 0.6).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(0.6).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// ```
pub struct ContinuousUniformDistribution {
    /// Lower bound of the distribution
    a: f64,
    /// Upper bound of the distribution
    b: f64,
}

impl ContinuousUniformDistribution {
    /// Creates a new `ContinuousUniformDistribution` instance.
    /// 
    /// # Input
    /// - `a`: Lower bound of the distribution
    /// - `b`: Upper bound of the distribution
    /// 
    /// # Errors
    /// - Returns an error if the value of `a` is larger or equal to `b`.
    pub fn build(a: f64, b: f64) -> Result<Self, DigiFiError> {
        if b <= a {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `a` must be smaller or equal to the argument `b`.".to_owned(),
            });
        }
        Ok(ContinuousUniformDistribution { a, b, })
    }
}

impl ErrorTitle for ContinuousUniformDistribution {
    fn error_title() -> String {
        String::from("Continuous Uniform Distribution")
    }
}

impl ProbabilityDistribution for ContinuousUniformDistribution {
    fn distribution_type() -> ProbabilityDistributionType {
        ProbabilityDistributionType::Continuous
    }

    fn mean(&self) -> f64 {
        (self.a + self.b) / 2.0
    }

    fn median(&self) -> Vec<f64> {
        vec![(self.a + self.b) / 2.0]
    }

    fn mode(&self) -> Vec<f64> {
        vec![(self.a + self.b) / 2.0]
    }

    fn variance(&self) -> f64 {
        (self.b - self.a).powi(2) / 12.0
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn excess_kurtosis(&self) -> f64 {
        -6.0 / 5.0
    }

    fn entropy(&self) -> Result<f64, DigiFiError> {
        Ok((self.b - self.a).ln())
    }

    /// Calculates the Probability Density Function (PDF) for a continuous uniform distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF value at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Continuous_uniform_distribution#Probability_density_function>
    /// - Original Source: N/A
    fn pdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok(if (self.a <= x) && (x <= self.b) { 1.0 / (self.b - self.a) } else { 0.0 })
    }

    /// Computes the Cumulative Distribution Function (CDF) for a continuous uniform distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF value at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Continuous_uniform_distribution#Cumulative_distribution_function>
    /// - Original Sourcew: N/A
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok(if self.a <= x { ((x - self.a) / (self.b - self.a)).min(1.0) } else { 0.0 })
    }

    /// Calculates the Inverse Cumulative Distribution Function (Inverse CDF) for a continuous uniform distribution.
    /// 
    /// # Input
    /// - `p`: Probability value for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF value for the given probabilities
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError> {
        if (p < 0.0) || (1.0 < p) { return Ok(f64::NAN) }
        Ok(self.a + p * (self.b - self.a))
    }

    /// Computes the Moment Generating Function (MGF) for a continuous uniform distribution.
    /// 
    /// # Input
    /// - `t`: Input value for the MGF
    /// 
    /// # Output
    /// - MGF value at the given `t`
    fn mgf(&self, t: usize) -> f64 {
        if t != 0 {
            let x: f64 = t as f64;
            ((x*self.b).exp() - (x*self.a).exp()) / (x*(self.b - self.a))
        } else {
            1.0
        }
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Methods and properties of normal distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Normal_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, NormalDistribution};
///
/// let dist: NormalDistribution = NormalDistribution::build(0.0, 1.0).unwrap();
/// let x: f64 = 0.6;
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
/// assert!((pdf_v - 0.33322460289179967).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 0.7257468822499265).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(0.7257468822499265).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// ```
pub struct NormalDistribution {
    /// Mean of the distribution
    mu: f64,
    /// Standard deviation of the distribution
    sigma: f64,
}

impl NormalDistribution {
    /// Creates a new `NormalDistribution` instance.
    /// 
    /// # Input
    /// - `mu`: Mean of the distribution
    /// - `sigma`: Standard deviation of the distribution
    /// 
    /// # Errors
    /// - Returns an error if `sigma` is negative.
    pub fn build(mu: f64, sigma: f64) -> Result<Self, DigiFiError> {
        if sigma < 0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `sigma` must be non-negative.".to_owned(),
            });
        }
        Ok(NormalDistribution { mu, sigma, })
    }
}

impl ErrorTitle for NormalDistribution {
    fn error_title() -> String {
        String::from("Normal Distribution")
    }
}

impl ProbabilityDistribution for NormalDistribution {
    fn distribution_type() -> ProbabilityDistributionType {
        ProbabilityDistributionType::Continuous
    }

    fn mean(&self) -> f64 {
        self.mu
    }

    fn median(&self) -> Vec<f64> {
        vec![self.mu]
    }

    fn mode(&self) -> Vec<f64> {
        vec![self.mu]
    }

    fn variance(&self) -> f64 {
        self.sigma.powi(2)
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn excess_kurtosis(&self) -> f64 {
        0.0
    }

    fn entropy(&self) -> Result<f64, DigiFiError> {
        Ok((2.0 * std::f64::consts::PI * std::f64::consts::E * self.sigma.powi(2)).ln() / 2.0)
    }

    /// Calculates the Probability Density Function (PDF) of a normal distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF value at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Normal_distribution#Probability_density_function>
    /// - Original Source: N/A
    fn pdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok((((x - self.mu) / self.sigma).powi(2) / -2.0).exp() / (self.sigma * (2.0*std::f64::consts::PI).sqrt()))
    }

    /// Computes the Cumulative Distribution Function (CDF) for a normal distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF value at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function>
    /// - Original Source: N/A
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok((1.0 + erf((x - self.mu) / (self.sigma * 2.0_f64.sqrt()), None)) / 2.0)
    }

    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a normal distribution.
    /// 
    /// # Input
    /// - `p`: Probability value for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF value for the given probabilities
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError> {
        if (p < 0.0) || (1.0 < p) { return Ok(f64::NAN) }
        Ok(self.mu + self.sigma * 2.0_f64.sqrt() * erfinv(2.0 * p - 1.0, None) )

    }

    /// Calculates the Moment Generating Function (MGF) for a normal distribution.
    /// 
    /// # Input
    /// - `t`: Input value for the MGF
    /// 
    /// # Output
    /// - MGF value at the given `t`
    fn mgf(&self, t: usize) -> f64 {
        let x: f64 = t as f64;
        (self.mu * x + 0.5 * self.sigma.powi(2) * x.powi(2)).exp()
    }
}

impl RiskMeasure for NormalDistribution {
    fn expected_shortfall(&self, alpha: f64) -> Result<f64, DigiFiError> {
        self.validate_alpha(alpha)?;
        let std_norm_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0)?;
        Ok(self.mu + self.sigma * std_norm_dist.pdf(std_norm_dist.inverse_cdf(alpha)?)? / (1.0 - alpha))
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Methods and properties of exponential distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Exponential_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, ExponentialDistribution};
///
/// let dist: ExponentialDistribution = ExponentialDistribution::build(0.5).unwrap();
/// let x: f64 = 0.6;
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
/// assert!((pdf_v - 0.37040911034085894).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 0.2591817793182821).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(0.2591817793182821).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// ```
pub struct ExponentialDistribution {
    /// Rate parameter
    lambda: f64,
}

impl ExponentialDistribution {
    /// Creates a new `ExponentialDistribution` instance.
    /// 
    /// # Input
    /// - `lambda`: Rate parameter
    /// 
    /// # Errors
    /// - Returns an error if `lambda` is not positive.
    pub fn build(lambda: f64) -> Result<Self, DigiFiError> {
        if lambda <= 0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `lambda` must be positive.".to_owned(),
            });
        }
        Ok(ExponentialDistribution { lambda, })
    }
}

impl ErrorTitle for ExponentialDistribution {
    fn error_title() -> String {
        String::from("Exponential Distribution")
    }
}

impl ProbabilityDistribution for ExponentialDistribution {
    fn distribution_type() -> ProbabilityDistributionType {
        ProbabilityDistributionType::Continuous
    }

    fn mean(&self) -> f64 {
        1.0 / self.lambda
    }

    fn median(&self) -> Vec<f64> {
        vec![2.0_f64.ln() / self.lambda]
    }

    fn mode(&self) -> Vec<f64> {
        vec![0.0]
    }

    fn variance(&self) -> f64 {
        1.0 / self.lambda.powi(2)
    }

    fn skewness(&self) -> f64 {
        2.0
    }

    fn excess_kurtosis(&self) -> f64 {
        6.0
    }

    fn entropy(&self) -> Result<f64, DigiFiError> {
        Ok(1.0 - self.lambda.ln())
    }

    /// Calculates the Probability Density Function (PDF) for an exponential distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF value at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Exponential_distribution#Probability_density_function>
    /// - Original Source: N/A
    fn pdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok(self.lambda * (-self.lambda * x).exp())
    }

    /// Computes the Cumulative Distribution Function (CDF) for an exponential distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF value at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Exponential_distribution#Cumulative_distribution_function>
    /// - Original Source: N/A
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok(1.0 - (-self.lambda * x).exp())
    }

    /// Calculates the Inverse Cumulative Distribution Function (Inverse CDF) for an exponential distribution.
    /// 
    /// # Input
    /// - `p`: Probability value for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF value for the given probabilities
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError> {
        if (p < 0.0) || (1.0 < p) { return Ok(f64::NAN) }
        Ok((1.0 - p).ln() / -self.lambda)
    }

    /// Computes the Moment Generating Function (MGF) for an exponential distribution.
    /// 
    /// # Input
    /// - `t`: Input value for the MGF
    /// 
    /// # Output
    /// - MGF value at the given `t`
    fn mgf(&self, t: usize) -> f64 {
        if (t as f64) < self.lambda { self.lambda / (self.lambda - (t as f64)) } else { f64::NAN }
    }
}

impl RiskMeasure for ExponentialDistribution {
    fn expected_shortfall(&self, alpha: f64) -> Result<f64, DigiFiError> {
        self.validate_alpha(alpha)?;
        Ok((-(1.0 - alpha).ln() + 1.0) / self.lambda)
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Methods and properties of Laplace distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Laplace_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, LaplaceDistribution};
///
/// let dist: LaplaceDistribution = LaplaceDistribution::build(1.0, 2.0).unwrap();
/// let x: f64 = 0.6;
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
// assert!((pdf_v - 0.20468268826949546).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 0.4093653765389909).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(0.4093653765389909).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// ```
pub struct LaplaceDistribution {
    /// Location parameter, which is the peak of the distribution
    mu: f64,
    /// Scale parameter, which controls the spread of the distribution
    b: f64,
}

impl LaplaceDistribution {
    /// Creates a new `LaplaceDistribution` instance.
    ///
    /// # Input
    /// - `mu`: Location parameter, which is the peak of the distribution
    /// - `b`: Scale parameter, which controls the spread of the distribution
    /// 
    /// # Errors
    /// - Returns an error if `b` is not positive.
    pub fn build(mu: f64, b: f64) -> Result<Self, DigiFiError> {
        if b <= 0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `b` must be positive.".to_owned(),
            });
        }
        Ok(LaplaceDistribution { mu, b, })
    }
}

impl ErrorTitle for LaplaceDistribution {
    fn error_title() -> String {
        String::from("Laplace Distribution")
    }
}

impl ProbabilityDistribution for LaplaceDistribution {
    fn distribution_type() -> ProbabilityDistributionType {
        ProbabilityDistributionType::Continuous
    }
    
    fn mean(&self) -> f64 {
        self.mu
    }

    fn median(&self) -> Vec<f64> {
        vec![self.mu]
    }

    fn mode(&self) -> Vec<f64> {
        vec![self.mu]
    }

    fn variance(&self) -> f64 {
        2.0 * self.b.powi(2)
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn excess_kurtosis(&self) -> f64 {
        3.0
    }

    fn entropy(&self) -> Result<f64, DigiFiError> {
        Ok((2.0 * self.b * std::f64::consts::E).ln())
    }

    /// Calculates the Probability Density Function (PDF) for a Laplace distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF value at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Laplace_distribution#Probability_density_function>
    /// - Original Source: N/A
    fn pdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok((-(x - self.mu).abs() / self.b).exp() / (2.0 * self.b))
    }

    /// Computes the Cumulative Distribution Function (CDF) for a Laplace distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF value at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Laplace_distribution#Cumulative_distribution_function>
    /// - Original Source: N/A
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok(if x <= self.b { 0.5 * ((x - self.mu) / self.b).exp() } else { 1.0 - 0.5 * (-(x - self.mu) / self.b).exp() })
    }

    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Laplace distribution.
    /// 
    /// # Input
    /// - `p`: Probability value for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF value for the given probabilities
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError> {
        if (p < 0.0) || (1.0 < p) { return Ok(f64::NAN) }
        Ok(self.mu - self.b * (p - 0.5).signum() * (1.0 - 2.0 * (p - 0.5).abs()).ln())
    }

    /// Calculates the Moment Generating Function (MGF) for a Laplace distribution.
    /// 
    /// # Input
    /// - `t`: Input value for the MGF
    /// 
    /// # Output
    /// - MGF value at the given `t`
    fn mgf(&self, t: usize) -> f64 {
        if (t as f64).abs() < 1.0 / self.b {
            let x: f64 = t as f64;
            (self.mu * x).exp() / (1.0 - self.b.powi(2) * x.powi(2))
        } else {
            f64::NAN
        }
    }
}

impl RiskMeasure for LaplaceDistribution {
    fn expected_shortfall(&self, alpha: f64) -> Result<f64, DigiFiError> {
        self.validate_alpha(alpha)?;
        if alpha < 0.5 {
            Ok(self.mu + self.b * alpha * (1.0 - (2.0 * alpha).ln()) / (1.0 - alpha))
        } else {
            Ok(self.mu + self.b * (1.0 - (2.0 - 2.0 * alpha).ln()))
        }
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Methods and properties of Gamma distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Gamma_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, GammaDistribution};
///
/// let dist: GammaDistribution = GammaDistribution::build(0.5, 2.0).unwrap();
/// let x: f64 = 0.6;
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
/// assert!((pdf_v - 0.38154528938409304).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 0.5614219739190003).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(0.5614219739190003).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// ```
pub struct GammaDistribution {
    /// Shape parameter, which controls the shape of the distribution
    k: f64,
    /// Scale parameter, which controls the spread of the distribution
    theta: f64,
}

impl GammaDistribution {
    /// Creates a new `GammaDistribution` instance.
    ///
    /// # Input
    /// - `k`: Shape parameter, which controls the shape of the distribution
    /// - `theta`: Scale parameter, which controls the spread of the distribution
    /// 
    /// # Errors
    /// - Returns an error if `k` is not positive.
    /// - Returns an error if `theta` is not positive.
    pub fn build(k: f64, theta: f64) -> Result<Self, DigiFiError> {
        if k <= 0.0 {
            return Err(DigiFiError::ParameterConstraint { title: Self::error_title(), constraint: "The argument `k` must be positive.".to_owned(), });
        }
        if theta <= 0.0 {
            return Err(DigiFiError::ParameterConstraint { title: Self::error_title(), constraint: "The argument `theta` must be positive.".to_owned(), });
        }
        Ok(GammaDistribution { k, theta, })
    }
}

impl ErrorTitle for GammaDistribution {
    fn error_title() -> String {
        String::from("Gamma Distribution")
    }
}

impl ProbabilityDistribution for GammaDistribution {
    fn distribution_type() -> ProbabilityDistributionType {
        ProbabilityDistributionType::Continuous
    }

    fn mean(&self) -> f64 {
        self.k * self.theta
    }

    fn median(&self) -> Vec<f64> {
        vec![f64::NAN]
    }

    fn mode(&self) -> Vec<f64> {
        if 1.0 <= self.k {
            vec![(self.k - 1.0) * self.theta]
        } else {
            vec![0.0]
        }
    }

    fn variance(&self) -> f64 {
        self.k * self.theta.powi(2)
    }

    fn skewness(&self) -> f64 {
        2.0 / self.k.sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        6.0 / self.k
    }

    fn entropy(&self) -> Result<f64, DigiFiError> {
        let f = |z: f64| { gamma(z.ln()) };
        Ok(self.k + self.theta.ln() + gamma(self.k).ln() + (1.0 - self.k) * derivative(f, self.k, 0.00000001))
    }

    /// Calculates the Probability Density Function (PDF) for a Gamma distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF value at the given `x`
    fn pdf(&self, x: f64) -> Result<f64, DigiFiError> {
        if x < 0.0 { return Ok(f64::NAN) }
        Ok(x.powf(self.k - 1.0) * (-x / self.theta).exp() / (gamma(self.k) * self.theta.powf(self.k)))
    }

    /// Computes the Cumulative Distribution Function (CDF) for a Gamma distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF value at the given `x`
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError> {
        if x < 0.0 { return Ok(f64::NAN) }
        Ok(lower_incomplete_gamma(self.k, x/self.theta, Some(50))? / gamma(self.k))
    }

    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Gamma distribution.
    /// 
    /// # Input
    /// - `p`: Probability value for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF value for the given probabilities
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError> {
        if (p < 0.0) || (1.0 < p) { return Ok(f64::NAN) }
        let f = |x: &[f64]| { (self.cdf(x[0]).unwrap() - p).powi(2) };
        Ok(nelder_mead(f, vec![0.0], Some(1_000), Some(10_000), None, None, None)?[0])
    }

    /// Calculates the Moment Generating Function (MGF) for a Gamma distribution.
    /// 
    /// # Input
    /// - `t`: Input value for the MGF
    /// 
    /// # Output
    /// - MGF value at the given `t`
    fn mgf(&self, t: usize) -> f64 {
        if (t as f64).abs() < 1.0 / self.theta {
            (1.0 - self.theta * (t as f64)).powf(-self.k)
        } else {
            f64::NAN
        }
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Methods and properties of Student's t-distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Student's_t-distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, StudentsTDistribution};
///
/// let dist: StudentsTDistribution = StudentsTDistribution::build(2.0).unwrap();
/// let x: Array1<f64> = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
/// 
/// // PDF test (Values obtained using t.pdf() from SciPy)
/// let pdf_v: Array1<f64> = dist.pdf_iter(x.iter()).unwrap();
/// let tested_values: Array1<f64> = Array1::from(vec![0.06804138, 0.19245009, 0.35355339, 0.19245009, 0.06804138]);
/// assert!((pdf_v - tested_values).fold(true, |test, i| if *i < TEST_ACCURACY && test { true } else { false } ));
/// 
/// // CDF test (Values obtained using t.cdf() from SciPy)
/// let cdf_v: Array1<f64> = dist.cdf_iter(x.iter()).unwrap();
/// let tested_values: Array1<f64> = Array1::from_vec(vec![0.09175171, 0.21132487, 0.5, 0.78867513, 0.90824829]);
/// assert!((cdf_v - &tested_values).fold(true, |test, i| if *i < TEST_ACCURACY && test { true } else { false } ));
/// 
/// // Inverse CDF test
/// let icdf_v: Array1<f64> = dist.inverse_cdf_iter(tested_values.iter()).unwrap();
/// assert!((icdf_v - x).fold(true, |test, i| if *i < 10.0 * TEST_ACCURACY && test { true } else { false } ));
/// 
/// 
/// // Moderate degrees of freedom test (Central values)
/// let dist: StudentsTDistribution = StudentsTDistribution::build(6.0).unwrap();
/// let x: Array1<f64> = Array1::from_vec(vec![0.75, 0.8, 0.85]);
/// let tested_values: Array1<f64> = Array1::from_vec(vec![0.718, 0.906, 1.134]);
/// let icdf_v: Array1<f64> = dist.inverse_cdf_iter(x.iter()).unwrap();
/// assert!((icdf_v - &tested_values).fold(true, |test, i| if *i < 100_000.0 * TEST_ACCURACY && test { true } else { false } ));
/// 
/// // Moderate degrees of freedom test (Tail values)
/// let x: Array1<f64> = Array1::from_vec(vec![0.95, 0.975, 0.99]);
/// let tested_values: Array1<f64> = Array1::from_vec(vec![1.943, 2.447, 3.143]);
/// let icdf_v: Array1<f64> = dist.inverse_cdf_iter(x.iter()).unwrap();
/// assert!((icdf_v - &tested_values).fold(true, |test, i| if *i < 1_000_000.0 * TEST_ACCURACY && test { true } else { false } ));
/// 
/// 
/// // Large degrees of freedom (Approaching 30 degrees of freedom, df = 29)
/// let dist: StudentsTDistribution = StudentsTDistribution::build(29.0).unwrap();
/// let x: Array1<f64> = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
/// 
/// // PDF test (Values obtained using t.pdf() from SciPy)
/// let pdf_v: Array1<f64> = dist.pdf_iter(x.iter()).unwrap();
/// let tested_values: Array1<f64> = Array1::from(vec![0.05694135, 0.23785815, 0.39551858, 0.23785815, 0.05694135]);
/// assert!((pdf_v - tested_values).fold(true, |test, i| if *i < TEST_ACCURACY && test { true } else { false } ));
/// 
/// // CDF test (Values obtained using t.cdf() from SciPy)
/// let cdf_v: Array1<f64> = dist.cdf_iter(x.iter()).unwrap();
/// let tested_values: Array1<f64> = Array1::from_vec(vec![0.02747182, 0.16279099, 0.5, 0.83720901, 0.97252818]);
/// assert!((cdf_v - &tested_values).fold(true, |test, i| if *i < TEST_ACCURACY && test { true } else { false } ));
/// 
/// // Inverse CDF test
/// let icdf_v: Array1<f64> = dist.inverse_cdf_iter(tested_values.iter()).unwrap();
/// assert!((icdf_v - x).fold(true, |test, i| if *i < 1_000_000.0 * TEST_ACCURACY && test { true } else { false } ));
/// 
/// 
/// // Very large degrees of freedom (df = 124)
/// let dist: StudentsTDistribution = StudentsTDistribution::build(124.0).unwrap();
/// let x: Array1<f64> = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
/// 
/// // PDF test (Values obtained using t.pdf() from SciPy)
/// let pdf_v: Array1<f64> = dist.pdf_iter(x.iter()).unwrap();
/// let tested_values: Array1<f64> = Array1::from(vec![0.0547352, 0.24099831, 0.39813878, 0.24099831, 0.0547352]);
/// assert!((pdf_v - tested_values).fold(true, |test, i| if *i < TEST_ACCURACY && test { true } else { false } ));
/// 
/// // CDF test (Values obtained using t.cdf() from SciPy)
/// let cdf_v: Array1<f64> = dist.cdf_iter(x.iter()).unwrap();
/// let tested_values: Array1<f64> = Array1::from_vec(vec![0.02384271, 0.15962897, 0.5, 0.84037103, 0.97615729]);
/// assert!((cdf_v - &tested_values).fold(true, |test, i| if *i < TEST_ACCURACY && test { true } else { false } ));
/// 
/// // Inverse CDF test
/// let icdf_v: Array1<f64> = dist.inverse_cdf_iter(tested_values.iter()).unwrap();
/// assert!((icdf_v - x).fold(true, |test, i| if *i < 10_000_000.0 * TEST_ACCURACY && test { true } else { false } ));
/// ```
pub struct StudentsTDistribution {
    /// Degrees of freedom
    v: f64,
}

impl StudentsTDistribution {
    /// Creates a new `StudentsTDistribution` instance.
    ///
    /// # Input
    /// - `v`: Degrees of freedom
    /// 
    /// # Errors
    /// - Returns an error if `v` is not positive.
    pub fn build(v: f64) -> Result<Self, DigiFiError> {
        if v <= 0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `v` must be positive.".to_owned(),
            });
        }
        Ok(StudentsTDistribution { v, })
    }
}

impl ErrorTitle for StudentsTDistribution {
    fn error_title() -> String {
        String::from("Student's T Distribution")
    }
}

impl ProbabilityDistribution for StudentsTDistribution {
    fn distribution_type() -> ProbabilityDistributionType {
        ProbabilityDistributionType::Continuous
    }
    
    fn mean(&self) -> f64 {
        if 1.0 < self.v { 0.0 } else { f64::NAN }
    }

    fn median(&self) -> Vec<f64> {
        vec![0.0]
    }

    fn mode(&self) -> Vec<f64> {
        vec![0.0]
    }

    fn variance(&self) -> f64 {
        if 2.0 < self.v { self.v / (self.v - 2.0) } else if (1.0 < self.v) && (self.v <= 2.0) { f64::INFINITY } else { f64::NAN }
    }

    fn skewness(&self) -> f64 {
        if 3.0 < self.v { 0.0 } else { f64::NAN }
    }

    fn excess_kurtosis(&self) -> f64 {
        if 4.0 < self.v { 6.0 / (self.v - 4.0) } else if (2.0 < self.v) && (self.v <= 4.0) { f64::INFINITY } else { f64::NAN }
    }

    fn entropy(&self) -> Result<f64, DigiFiError> {
        let part_1: f64 = (digamma((self.v + 1.0) / 2.0)? - digamma(self.v / 2.0)?) * (self.v + 1.0) / 2.0;
        let part_2: f64 = (self.v.sqrt() * beta(self.v / 2.0, 0.5)?).ln();
        Ok(part_1 + part_2)
    }

    /// Calculates the Probability Density Function (PDF) for a Student's t-distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF value at the given `x`
    fn pdf(&self, x: f64) -> Result<f64, DigiFiError> {
        let v_: f64 = (self.v + 1.0) / 2.0;
        let num: f64 = gamma(v_) * (1.0 + x.powi(2) / self.v).powf(-v_);
        let denom: f64 = gamma(self.v / 2.0) * (std::f64::consts::PI * self.v).sqrt();
        Ok(num / denom)
    }

    /// Computes the Cumulative Distribution Function (CDF) for a Student's t-distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF value at the given `x`
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError> {
        if x == 0.0 {
            Ok(0.5)
        } else if x.is_infinite() {
            if x.is_sign_negative() { Ok(0.0) } else { Ok(1.0) }
        } else if x < 0.0  {
            Ok(0.5 * regularized_incomplete_beta(self.v / (self.v + x.powi(2)), self.v / 2.0, 0.5)?)
        } else {
            Ok(1.0 - 0.5 * regularized_incomplete_beta(self.v / (self.v + x.powi(2)), self.v / 2.0, 0.5)?)
        }
    }

    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Student's t-distribution.
    /// 
    /// # Input
    /// - `p`: Probability value for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF value for the given probabilities
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError> {
        if (p < 0.0) || (1.0 < p) { return Ok(f64::NAN) }
        // Shortcut approximations for some degrees of freedom
        match self.v {
            1.0 => Ok((std::f64::consts::PI * (p - 0.5)).tan()),
            2.0 => Ok(2.0 * (p - 0.5) * (2.0 / (4.0 * p * (1.0 - p))).sqrt()),
            4.0 => {
                let alpha: f64 = 4.0 * p * (1.0 - p);
                let q: f64 = (alpha.sqrt().acos() / 3.0).cos() / alpha.sqrt();
                let sign: f64 = if (p - 0.5) < 0.0 { -1.0 } else if 0.0 < (p - 0.5) { 1.0 } else { 0.0 };
                Ok(sign * 2.0 * (q - 1.0).sqrt())
            },
            v if v < 30.0 => {
                // Cornish-Fisher extension
                let dist: NormalDistribution = NormalDistribution::build(0.0, 1.0)?;
                let z: f64 = dist.inverse_cdf(p)?;
                let result: f64 = z + z*(z.powi(2) + 1.0)/(4.0 * self.v) + z*(5.0 * z.powi(4) + 16.0 * z.powi(2) + 3.0)/(96.0 * self.v.powi(2))
                    + z*(3.0 * z.powi(7) + 19.0 * z.powi(5) + 17.0 * z.powi(3) - 16.0 * z)/(384.0 * self.v.powi(3))
                    + z*(79.0 * z.powi(9) + 776.0 * z.powi(7) + 1482.0 * z.powi(5) - 1920.0 * z.powi(3) - 945.0 * z)/(96160.0 * self.v.powi(4));
                Ok(result)
            },
            _ => {
                // Approximation via Normal distribution for high number of degrees of freedom
                NormalDistribution::build(0.0, 1.0)?.inverse_cdf(p)
            },
        }
    }

    /// Calculates the Moment Generating Function (MGF) for a Student's t-distribution.
    /// Note: Since MGF is not defined this function will always return undefined values.
    /// 
    /// # Input
    /// - `t`: Input value for the MGF
    /// 
    /// # Output
    /// - MGF value at the given `t`
    fn mgf(&self, _t: usize) -> f64 {
        f64::NAN
    }
}

impl RiskMeasure for StudentsTDistribution {
    fn expected_shortfall(&self, alpha: f64) -> Result<f64, DigiFiError> {
        self.validate_alpha(alpha)?;
        let inv_cdf_alpha: f64 = self.inverse_cdf(alpha)?;
        Ok((self.v + inv_cdf_alpha.powi(2)) * self.pdf(inv_cdf_alpha)? / ((self.v - 1.0) * (1.0 - alpha)))
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Methods and properties of Pareto distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Pareto_distribution>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, ParetoDistribution};
/// 
/// // Standard scale (scale = 1)
/// let dist: ParetoDistribution = ParetoDistribution::build(1.0, 3.0).unwrap();
/// let x: f64 = 1.5;
/// 
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
/// assert!((pdf_v - 0.5925925925925926).abs() < TEST_ACCURACY);
/// 
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 0.7037037037037037).abs() < TEST_ACCURACY);
/// 
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(0.7037037037037037).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// 
/// 
/// // Non-standard scale (scale = 2)
/// let dist: ParetoDistribution = ParetoDistribution::build(2.0, 3.0).unwrap();
/// let x: f64 = 2.5;
/// 
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
/// assert!((pdf_v - 0.6144000000000001).abs() < TEST_ACCURACY);
/// 
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 0.488).abs() < TEST_ACCURACY);
/// 
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(0.488).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// ```
pub struct ParetoDistribution {
    /// Scale parameter (i.e., minimum possible value of the distribution)
    scale: f64,
    /// Shape parameter
    a: f64,
}

impl ParetoDistribution {

    /// Creates a new `ParetoDistribution` instance.
    ///
    /// # Input
    /// - `scale`: Scale parameter (i.e., minimum possible value of the distribution)
    /// - `a`: Shape parameter
    /// 
    /// # Errors
    /// - Returns an error if `scale` is not positive.
    /// - Returns and error if `a` is not positive.
    pub fn build(scale: f64, a: f64) -> Result<Self, DigiFiError> {
        if scale <= 0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `scale` must be positive.".to_owned(),
            });
        }
        if a <= 0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `a` must be positive.".to_owned(),
            });
        }
        Ok(ParetoDistribution { scale, a, } )
    }
}

impl ErrorTitle for ParetoDistribution {
    fn error_title() -> String {
        String::from("Pareto Distribution")
    }
}

impl ProbabilityDistribution for ParetoDistribution {
    fn distribution_type() -> ProbabilityDistributionType {
        ProbabilityDistributionType::Continuous
    }
    
    fn mean(&self) -> f64 {
        if self.a <= 1.0 { f64::INFINITY } else { self.a * self.scale / (self.a - 1.0) }
    }

    fn median(&self) -> Vec<f64> {
        vec![self.scale * 2.0_f64.powf(1.0 / self.a)]
    }

    fn mode(&self) -> Vec<f64> {
        vec![self.scale]
    }

    fn variance(&self) -> f64 {
        if self.a <= 2.0 { f64::INFINITY } else { self.scale.powi(2) * self.a / ((self.a - 1.0).powi(2) * (self.a - 2.0)) }
    }

    fn skewness(&self) -> f64 {
        if 3.0 < self.a { (2.0 * (1.0 + self.a) / (self.a - 3.0)) * ((self.a - 2.0) / self.a).sqrt() } else { f64::NAN }
    }

    fn excess_kurtosis(&self) -> f64 {
        if 4.0 < self.a { 6.0 * (self.a.powi(3) + self.a.powi(2) - 6.0 * self.a - 2.0) / (self.a * (self.a - 3.0) * (self.a - 4.0)) } else { f64::NAN }
    }

    fn entropy(&self) -> Result<f64, DigiFiError> {
        Ok((self.scale * (1.0 + 1.0 / self.a).exp() / self.a).ln())
    }

    /// Calculates the Probability Density Function (PDF) for a Pareto distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF value at the given `x`
    fn pdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok(self.a * self.scale.powf(self.a) / x.powf(self.a + 1.0))
    }

    /// Computes the Cumulative Distribution Function (CDF) for a Pareto distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF value at the given `x`
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok(1.0 - (self.scale / x).powf(self.a))
    }

    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Pareto distribution.
    /// 
    /// # Input
    /// - `p`: Probability value for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF value for the given probabilities
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError> {
        Ok(self.scale * (1.0 - p).powf(-1.0 / self.a))
    }

    /// Calculates the Moment Generating Function (MGF) for a Pareto distribution.
    /// Note: Since MGF is not defined this function will always return undefined values.
    /// 
    /// # Input
    /// - `t`: Input value for the MGF
    /// 
    /// # Output
    /// - MGF value at the given `t`
    fn mgf(&self, _t: usize) -> f64 {
        f64::NAN
    }
}

impl RiskMeasure for ParetoDistribution {
    fn expected_shortfall(&self, alpha: f64) -> Result<f64, DigiFiError> {
        self.validate_alpha(alpha)?;
        Ok(self.scale * self.a * ((1.0 - alpha).powf(1.0 / self.a) * (self.a - 1.0)))
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Methods and properties of Log-normal distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Log-normal_distribution>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, LogNormalDistribution};
/// 
/// // Sandard case (mu = 0, sigma = 1)
/// let dist: LogNormalDistribution = LogNormalDistribution::build(0.0, 1.0).unwrap();
/// let x: f64 = 0.6;
/// 
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
/// assert!((pdf_v - 0.583573822594504).abs() < TEST_ACCURACY);
/// 
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 0.30473658251023167).abs() < TEST_ACCURACY);
/// 
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(0.30473658251023167).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// 
/// 
/// // Non-standard case (mu = 0, sigma = 0.5)
/// let dist: LogNormalDistribution = LogNormalDistribution::build(0.0, 0.5).unwrap();
/// let x: f64 = 0.6;
/// 
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
/// assert!((pdf_v - 0.7891085687134382).abs() < TEST_ACCURACY);
/// 
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 0.15347299656473).abs() < TEST_ACCURACY);
/// 
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(0.15347299656473).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// ```
pub struct LogNormalDistribution {
    /// Logarithm of location
    mu: f64,
    /// Logarithm of scale
    sigma: f64,
}

impl LogNormalDistribution {

    /// Creates a new `LogNormalDistribution` instance.
    ///
    /// # Input
    /// - `mu`: Logarithm of location
    /// - `sigma`: Logarithm of scale
    /// 
    /// # Errors
    /// - Returns an error if `sigma` is not positive.
    pub fn build(mu: f64, sigma: f64) -> Result<Self, DigiFiError> {
        if sigma <= 0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `sigma` must be positive.".to_owned(),
            });
        }
        Ok(LogNormalDistribution { mu, sigma, })
    }
}

impl ErrorTitle for LogNormalDistribution {
    fn error_title() -> String {
        String::from("Log-Normal Distribution")
    }
}

impl ProbabilityDistribution for LogNormalDistribution {
    fn distribution_type() -> ProbabilityDistributionType {
        ProbabilityDistributionType::Continuous
    }
    
    fn mean(&self) -> f64 {
        (self.mu + self.sigma.powi(2) / 2.0).exp()
    }

    fn median(&self) -> Vec<f64> {
        vec![self.mu.exp()]
    }

    fn mode(&self) -> Vec<f64> {
        vec![(self.mu - self.sigma.powi(2)).exp()]
    }

    fn variance(&self) -> f64 {
        let sigma_sq: f64 = self.sigma.powi(2);
        (sigma_sq.exp() - 1.0) * (2.0 * self.mu + sigma_sq).exp()
    }

    fn skewness(&self) -> f64 {
        let exp_sigma_sq: f64 = self.sigma.powi(2).exp();
        (exp_sigma_sq + 2.0) * (exp_sigma_sq - 1.0).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        let sigma_sq: f64 = self.sigma.powi(2);
        (4.0 * sigma_sq).exp() + 2.0 * (3.0 * sigma_sq).exp() + 3.0 * (2.0 * sigma_sq).exp() - 6.0
    }

    fn entropy(&self) -> Result<f64, DigiFiError> {
        Ok((self.sigma * self.mu.exp() * (2.0 * std::f64::consts::E * std::f64::consts::PI).sqrt()).log2())
    }

    /// Calculates the Probability Density Function (PDF) for a log-normal distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF value at the given `x`
    fn pdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok((- (x.ln() - self.mu).powi(2) / (2.0 * self.sigma.powi(2))).exp() / (x * self.sigma * (2.0 * std::f64::consts::PI).sqrt()))
    }

    /// Computes the Cumulative Distribution Function (CDF) for a log-normal distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF value at the given `x`
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok((1.0 + erf((x.ln() - self.mu) / (self.sigma * 2.0_f64.sqrt()), None)) / 2.0)
    }

    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a log-normal distribution.
    /// 
    /// # Input
    /// - `p`: Probability value for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF value for the given probabilities
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError> {
        Ok((self.mu + (2.0 * self.sigma.powi(2)).sqrt() * erfinv(2.0 * p - 1.0, None)).exp())
    }

    /// Calculates the Moment Generating Function (MGF) for a log-normal distribution.
    /// Note: Since MGF is not defined this function will always return undefined values.
    /// 
    /// # Input
    /// - `t`: Input value for the MGF
    /// 
    /// # Output
    /// - MGF value at the given `t`
    fn mgf(&self, _t: usize) -> f64 {
        f64::NAN
    }
}

impl RiskMeasure for LogNormalDistribution {
    fn expected_shortfall(&self, alpha: f64) -> Result<f64, DigiFiError> {
        self.validate_alpha(alpha)?;
        let num: f64 = 1.0 + erf(self.sigma / 2.0_f64.sqrt() - erfinv(2.0 * alpha - 1.0, None), None);
        Ok(0.5 * (self.mu + self.sigma.powi(2) / 2.0).exp() * num / (1.0 - alpha))
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Methods and properties of F-distribution.
/// 
/// # LaTeX Formula
/// - X = \\frac{U_{1}/d_{1}}{U_{2}/d_{2}}
/// where `X` is a random variable with F-distribution, `U_{1}` and `U_{2}` are independent random variables with chi-squared distributions
/// with respective degrees of freedom `d_{1}` and `d_{2}`.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/F-distribution>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, FDistribution};
/// 
/// let dist: FDistribution = FDistribution::build(30, 20).unwrap();
/// let x: f64 = 0.6;
/// 
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
/// assert!((pdf_v - 0.7230705152456879).abs() < TEST_ACCURACY);
/// 
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 0.1001670547365643).abs() < TEST_ACCURACY);
/// 
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(0.1001670547365643).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// ```
pub struct FDistribution {
    /// Degrees of freedom of random variable `U_{1}`
    dof_1: usize,
    /// Degrees of freedom of random variable `U_{2}`
    dof_2: usize,
}

impl FDistribution {

    /// Creates a new `FDistribution` instance.
    ///
    /// # Input
    /// - `dof_1`: Degrees of freedom of random variable `U_{1}`
    /// - `dof_2`: Degrees of freedom of random variable `U_{2}`
    /// 
    /// # Errors
    /// - Returns an error if either `ddof_1` or `ddof_2` is not positive.
    pub fn build(dof_1: usize, dof_2: usize) -> Result<Self, DigiFiError> {
        if dof_1 < 1 || dof_2 < 1 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "Both degrees of freedom `dof_1` and `dof_2` must be positive.".to_owned(),
            });
        }
        Ok(FDistribution { dof_1, dof_2, })
    }

    /// Returns degrees of freedom converted to `f64` in order (i.e., `dof_1`, `dof_2`).
    fn get_f64_dofs(&self) -> (f64, f64) {
        (self.dof_1 as f64, self.dof_2 as f64)
    }
}

impl ErrorTitle for FDistribution {
    fn error_title() -> String {
        String::from("F-distribution")
    }
}

impl ProbabilityDistribution for FDistribution {
    fn distribution_type() -> ProbabilityDistributionType {
        ProbabilityDistributionType::Continuous
    }
    
    fn mean(&self) -> f64 {
        let (_, dof_2) = self.get_f64_dofs();
        if 2 < self.dof_2 { dof_2 / (dof_2 - 2.0) } else { f64::NAN }
    }

    fn median(&self) -> Vec<f64> {
        vec![f64::NAN]
    }

    fn mode(&self) -> Vec<f64> {
        let (dof_1, dof_2) = self.get_f64_dofs();
        vec![if 2 < self.dof_1 { (dof_1 - 2.0) * dof_2 / (dof_1 * (dof_2 + 2.0)) } else { f64::NAN }]
    }

    fn variance(&self) -> f64 {
        let (dof_1, dof_2) = self.get_f64_dofs();
        if 4 < self.dof_2 { 2.0 * dof_2.powi(2) * (dof_1 + dof_2 - 2.0) / (dof_1 * (dof_2 - 2.0).powi(2) * (dof_2 - 4.0)) } else { f64::NAN }
    }

    fn skewness(&self) -> f64 {
        let (dof_1, dof_2) = self.get_f64_dofs();
        if 6 < self.dof_2 {
            (2.0 * dof_1 + dof_2 - 2.0) * (8.0 * (dof_2 - 4.0)).sqrt() / ((dof_2 - 6.0) * (dof_1 * (dof_1 + dof_2 - 2.0)).sqrt())
        } else {
            f64::NAN
        }
    }

    fn excess_kurtosis(&self) -> f64 {
        let (dof_1, dof_2) = self.get_f64_dofs();
        if 8 < self.dof_2 {
            let c: f64 = dof_1 + dof_2 - 2.0;
            let numerator: f64 = dof_1 * (5.0 * dof_2 - 22.0) * c + (dof_2 - 4.0) * (dof_2 - 2.0).powi(2);
            let denominator: f64 = dof_1 * (dof_2 - 6.0) * (dof_2 - 8.0) * c;
            12.0 * numerator / denominator
        } else {
            f64::NAN
        }
    }

    fn entropy(&self) -> Result<f64, DigiFiError> {
        let (dof_1, dof_2) = self.get_f64_dofs();
        let d_1_over_2: f64 = dof_1 / 2.0;
        let d_2_over_2: f64 = dof_2 / 2.0;
        let one_plus_d_2_over_2: f64 = 1.0 + d_2_over_2;
        let d_1_plus_d_2_over_2: f64 = (dof_1 + dof_2) / 2.0;
        let entropy: f64 = ln_gamma(d_1_over_2) + ln_gamma(d_2_over_2) - ln_gamma(d_1_plus_d_2_over_2)
            + (1.0 - d_1_over_2) * digamma(1.0 + d_1_over_2)? - one_plus_d_2_over_2 * digamma(one_plus_d_2_over_2)?
            + d_1_plus_d_2_over_2 * digamma(d_1_plus_d_2_over_2)? + (dof_2 / dof_1).ln();
        Ok(entropy)
    }

    /// Calculates the Probability Density Function (PDF) for an F-distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF value at the given `x`
    fn pdf(&self, x: f64) -> Result<f64, DigiFiError> {
        if (self.dof_1 == 1 && x == 0.0) || x < 0.0 { return Ok(f64::NAN); }
        let (dof_1, dof_2) = self.get_f64_dofs();
        let (dof_1_i32, dof_2_i32) = (self.dof_1 as i32, self.dof_2 as i32);
        let d_1_times_x: f64 = dof_1 * x;
        let numerator: f64 = (d_1_times_x.powi(dof_1_i32) * dof_2.powi(dof_2_i32) / (d_1_times_x + dof_2).powi(dof_1_i32 + dof_2_i32)).sqrt();
        let denominator: f64 = x * beta(dof_1 / 2.0, dof_2 / 2.0)?;
        Ok(numerator / denominator)
    }

    /// Computes the Cumulative Distribution Function (CDF) for an F-distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF value at the given `x`
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError> {
        if (self.dof_1 == 1 && x == 0.0) || x < 0.0 { return Ok(f64::NAN); }
        let (dof_1, dof_2) = self.get_f64_dofs();
        let d_1_times_x: f64 = dof_1 * x;
        let x: f64 = d_1_times_x / (d_1_times_x + dof_2);
        regularized_incomplete_beta(x, dof_1 / 2.0, dof_2 / 2.0)
    }

    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for an F-distribution.
    /// 
    /// # Input
    /// - `p`: Probability value for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF value for the given probabilities
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError> {
        let (dof_1, dof_2) = self.get_f64_dofs();
        let proxy_x: f64 = inverse_regularized_incomplete_beta(p, dof_1 / 2.0, dof_2 / 2.0)?;
        Ok(if proxy_x == 1.0 { f64::NAN } else { dof_2 * proxy_x / (dof_1 * (1.0 - proxy_x)) })
    }

    /// Calculates the Moment Generating Function (MGF) for an F-distribution.
    /// Note: Since MGF is not defined this function will always return undefined values.
    /// 
    /// # Input
    /// - `t`: Input value for the MGF
    /// 
    /// # Output
    /// - MGF value at the given `t`
    fn mgf(&self, _t: usize) -> f64 {
        f64::NAN
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{Array1, arr1};
    use crate::utilities::TEST_ACCURACY;
    use crate::statistics::ProbabilityDistribution;

    #[test]
    fn unit_test_continuous_uniform_distribution() -> () {
        use crate::statistics::continuous_distributions::ContinuousUniformDistribution;
        let dist: ContinuousUniformDistribution = ContinuousUniformDistribution::build(0.0, 1.0).unwrap();
        let x: f64 = 0.6;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 1.0).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 0.6).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(0.6).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_normal_distribution() -> () {
        use crate::statistics::continuous_distributions::NormalDistribution;
        let dist: NormalDistribution = NormalDistribution::build(0.0, 1.0).unwrap();
        let x: f64 = 0.6;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 0.33322460289179967).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 0.7257468822499265).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(0.7257468822499265).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_exponential_distribution() -> () {
        use crate::statistics::continuous_distributions::ExponentialDistribution;
        let dist: ExponentialDistribution = ExponentialDistribution::build(0.5).unwrap();
        let x: f64 = 0.6;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 0.37040911034085894).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 0.2591817793182821).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(0.2591817793182821).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_laplace_distribution() -> () {
        use crate::statistics::continuous_distributions::LaplaceDistribution;
        let dist: LaplaceDistribution = LaplaceDistribution::build(1.0, 2.0).unwrap();
        let x: f64 = 0.6;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 0.20468268826949546).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 0.4093653765389909).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(0.4093653765389909).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_gamma_distribution() -> () {
        use crate::statistics::continuous_distributions::GammaDistribution;
        let dist: GammaDistribution = GammaDistribution::build(0.5, 2.0).unwrap();
        let x: f64 = 0.6;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 0.38154528938409304).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 0.5614219739190003).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(0.5614219739190003).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_students_t_distribution() -> () {
        use crate::statistics::continuous_distributions::StudentsTDistribution;
        let dist: StudentsTDistribution = StudentsTDistribution::build(2.0).unwrap();
        let x: Array1<f64> = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        // PDF test (Values obtained using t.pdf() from SciPy)
        let pdf_v: Array1<f64> = dist.pdf_iter(x.iter()).unwrap();
        let tested_values: Array1<f64> = Array1::from(vec![0.06804138, 0.19245009, 0.35355339, 0.19245009, 0.06804138]);
        assert!((pdf_v - tested_values).fold(true, |test, i| if *i < TEST_ACCURACY && test { true } else { false } ));
        // CDF test (Values obtained using t.cdf() from SciPy)
        let cdf_v: Array1<f64> = dist.cdf_iter(x.iter()).unwrap();
        let tested_values: Array1<f64> = Array1::from_vec(vec![0.09175171, 0.21132487, 0.5, 0.78867513, 0.90824829]);
        assert!((cdf_v - &tested_values).fold(true, |test, i| if *i < TEST_ACCURACY && test { true } else { false } ));
        // Inverse CDF test
        let icdf_v: Array1<f64> = dist.inverse_cdf_iter(tested_values.iter()).unwrap();
        assert!((icdf_v - x).fold(true, |test, i| if *i < 10.0 * TEST_ACCURACY && test { true } else { false } ));

        // Moderate degrees of freedom test (Central values)
        let dist: StudentsTDistribution = StudentsTDistribution::build(6.0).unwrap();
        let x: Array1<f64> = Array1::from_vec(vec![0.75, 0.8, 0.85]);
        let tested_values: Array1<f64> = Array1::from_vec(vec![0.718, 0.906, 1.134]);
        let icdf_v: Array1<f64> = dist.inverse_cdf_iter(x.iter()).unwrap();
        assert!((icdf_v - &tested_values).fold(true, |test, i| if *i < 100_000.0 * TEST_ACCURACY && test { true } else { false } ));
        // Moderate degrees of freedom test (Tail values)
        let x: Array1<f64> = Array1::from_vec(vec![0.95, 0.975, 0.99]);
        let tested_values: Array1<f64> = Array1::from_vec(vec![1.943, 2.447, 3.143]);
        let icdf_v: Array1<f64> = dist.inverse_cdf_iter(x.iter()).unwrap();
        assert!((icdf_v - &tested_values).fold(true, |test, i| if *i < 1_000_000.0 * TEST_ACCURACY && test { true } else { false } ));

        // Large degrees of freedom (Approaching 30 degrees of freedom, df = 29)
        let dist: StudentsTDistribution = StudentsTDistribution::build(29.0).unwrap();
        let x: Array1<f64> = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        // PDF test (Values obtained using t.pdf() from SciPy)
        let pdf_v: Array1<f64> = dist.pdf_iter(x.iter()).unwrap();
        let tested_values: Array1<f64> = Array1::from(vec![0.05694135, 0.23785815, 0.39551858, 0.23785815, 0.05694135]);
        assert!((pdf_v - tested_values).fold(true, |test, i| if *i < TEST_ACCURACY && test { true } else { false } ));
        // CDF test (Values obtained using t.cdf() from SciPy)
        let cdf_v: Array1<f64> = dist.cdf_iter(x.iter()).unwrap();
        let tested_values: Array1<f64> = Array1::from_vec(vec![0.02747182, 0.16279099, 0.5, 0.83720901, 0.97252818]);
        assert!((cdf_v - &tested_values).fold(true, |test, i| if *i < TEST_ACCURACY && test { true } else { false } ));
        // Inverse CDF test
        let icdf_v: Array1<f64> = dist.inverse_cdf_iter(tested_values.iter()).unwrap();
        assert!((icdf_v - x).fold(true, |test, i| if *i < 1_000_000.0 * TEST_ACCURACY && test { true } else { false } ));

        // Very large degrees of freedom (df = 124)
        let dist: StudentsTDistribution = StudentsTDistribution::build(124.0).unwrap();
        let x: Array1<f64> = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        // PDF test (Values obtained using t.pdf() from SciPy)
        let pdf_v: Array1<f64> = dist.pdf_iter(x.iter()).unwrap();
        let tested_values: Array1<f64> = Array1::from(vec![0.0547352, 0.24099831, 0.39813878, 0.24099831, 0.0547352]);
        assert!((pdf_v - tested_values).fold(true, |test, i| if *i < TEST_ACCURACY && test { true } else { false } ));
        // CDF test (Values obtained using t.cdf() from SciPy)
        let cdf_v: Array1<f64> = dist.cdf_iter(x.iter()).unwrap();
        let tested_values: Array1<f64> = Array1::from_vec(vec![0.02384271, 0.15962897, 0.5, 0.84037103, 0.97615729]);
        assert!((cdf_v - &tested_values).fold(true, |test, i| if *i < TEST_ACCURACY && test { true } else { false } ));
        // Inverse CDF test
        let icdf_v: Array1<f64> = dist.inverse_cdf_iter(tested_values.iter()).unwrap();
        assert!((icdf_v - x).fold(true, |test, i| if *i < 10_000_000.0 * TEST_ACCURACY && test { true } else { false } ));
    }

    #[test]
    fn unit_test_pareto_distribution() -> () {
        use crate::statistics::continuous_distributions::ParetoDistribution;
        // Standard scale (scale = 1)
        let dist: ParetoDistribution = ParetoDistribution::build(1.0, 3.0).unwrap();
        let x: f64 = 1.5;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 0.5925925925925926).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 0.7037037037037037).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(0.7037037037037037).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);

        // Non-standard scale (scale = 2)
        let dist: ParetoDistribution = ParetoDistribution::build(2.0, 3.0).unwrap();
        let x: f64 = 2.5;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 0.6144000000000001).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 0.488).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(0.488).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_log_normal_distribution() -> () {
        use crate::statistics::continuous_distributions::LogNormalDistribution;
        // Sandard case (mu = 0, sigma = 1)
        let dist: LogNormalDistribution = LogNormalDistribution::build(0.0, 1.0).unwrap();
        let x: f64 = 0.6;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 0.583573822594504).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 0.30473658251023167).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(0.30473658251023167).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);

        // Non-standard case (mu = 0, sigma = 0.5)
        let dist: LogNormalDistribution = LogNormalDistribution::build(0.0, 0.5).unwrap();
        let x: f64 = 0.6;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 0.7891085687134382).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 0.15347299656473).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(0.15347299656473).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_f_distribution() -> () {
        use crate::statistics::continuous_distributions::FDistribution;
        let dist: FDistribution = FDistribution::build(30, 20).unwrap();
        let x: f64 = 0.6;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 0.7230705152456879).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 0.1001670547365643).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(0.1001670547365643).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);
    }
}