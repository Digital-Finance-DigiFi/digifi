use ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;
use crate::utilities::{maths_utils::{FunctionEvalMethod, erf, erfinv, derivative}, numerical_engines::nelder_mead};
use crate::statistics::{ProbabilityDistribution, ProbabilityDistributionType, gamma_function, lower_incomplete_gamma_function};


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Methods and properties of continuous uniform distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Continuous_uniform_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, ContinuousUniformDistribution};
///
/// let dist: ContinuousUniformDistribution = ContinuousUniformDistribution::new(0.0, 1.0).unwrap();
/// let x: Array1<f64> = arr1(&[0.6]);
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
/// assert!((pdf_v - 1.0).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
/// assert!((cdf_v - 0.6).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.6])).unwrap()[0];
/// assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
/// ```
pub struct ContinuousUniformDistribution {
    /// Lower bound of the distribution
    a: f64,
    /// Upper bound of the distribution
    b: f64,
    /// Type of distribution
    _distribution_type: ProbabilityDistributionType,
}

impl ContinuousUniformDistribution {
    /// # Description
    /// Creates a new `ContinuousUniformDistribution` instance.
    /// 
    /// # Input
    /// - `a`: Lower bound of the distribution
    /// - `b`: Upper bound of the distribution
    /// 
    /// # Errors
    /// - Returns an error if the value of `a` is larger or equal to `b`.
    pub fn new(a: f64, b: f64) -> Result<Self, DigiFiError> {
        if b <= a {
            return Err(DigiFiError::ParameterConstraint { title: "Continuous Uniform Distribution".to_owned(), constraint: "The argument `a` must be smaller or equal to the argument `b`.".to_owned(), });
        }
        Ok(ContinuousUniformDistribution { a, b, _distribution_type: ProbabilityDistributionType::Continuous })
    }
}

impl ProbabilityDistribution for ContinuousUniformDistribution {
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

    fn entropy(&self) -> f64 {
        (self.b - self.a).ln()
    }

    /// # Description
    /// Calculates the Probability Density Function (PDF) for a continuous uniform distribution.
    /// 
    /// # Input
    /// - `x`: Values at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF values at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Continuous_uniform_distribution#Probability_density_function>
    /// - Original Source: N/A
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        Ok(x.map(|x_| if (self.a <= *x_) && ( *x_ <= self.b) {1.0 / (self.b - self.a) } else { 0.0 } ))
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a continuous uniform distribution.
    /// 
    /// # Input
    /// - `x`: Values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Continuous_uniform_distribution#Cumulative_distribution_function>
    /// - Original Sourcew: N/A
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        Ok(x.map(|x_| if self.a <= *x_ { ((x_ - self.a) / (self.b - self.a)).min(1.0) } else { 0.0 } ))
    }

    /// # Description
    /// Calculates the Inverse Cumulative Distribution Function (Inverse CDF) for a continuous uniform distribution.
    /// 
    /// # Input
    /// - `p`: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_} );
        Ok(self.a + p * (self.b - self.a))
    }

    /// # Description
    /// Computes the Moment Generating Function (MGF) for a continuous uniform distribution.
    /// 
    /// # Input
    /// - `t`: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given `t`
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| if *t_ != 0 {
            let x: f64 = *t_ as f64;
            ((x*self.b).exp() - (x*self.a).exp()) / (x*(self.b - self.a))
        } else { 1.0 } )
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Methods and properties of normal distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Normal_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, NormalDistribution};
///
/// let dist: NormalDistribution = NormalDistribution::new(0.0, 1.0).unwrap();
/// let x: Array1<f64> = arr1(&[0.6]);
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
/// assert!((pdf_v - 0.33322460289179967).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
/// assert!((cdf_v - 0.7257468822499265).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.7257468822499265])).unwrap()[0];
/// assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
/// ```
pub struct NormalDistribution {
    /// Mean of the distribution
    mu: f64,
    /// Standard deviation of the distribution
    sigma: f64,
    /// Type of distribution
    _distribution_type: ProbabilityDistributionType
}

impl NormalDistribution {
    /// # Description
    /// Creates a new `NormalDistribution` instance.
    /// 
    /// # Input
    /// - `mu`: Mean of the distribution
    /// - `sigma`: Standard deviation of the distribution
    /// 
    /// # Errors
    /// - Returns an error if `sigma` is negative.
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DigiFiError> {
        if sigma < 0.0 {
            return Err(DigiFiError::ParameterConstraint { title: "Normal Distribution".to_owned(), constraint: "The argument `sigma` must be non-negative.".to_owned(), });
        }
        Ok(NormalDistribution { mu, sigma, _distribution_type: ProbabilityDistributionType::Continuous })
    }
}

impl ProbabilityDistribution for NormalDistribution {
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

    fn entropy(&self) -> f64 {
        (2.0 * std::f64::consts::PI * std::f64::consts::E * self.sigma.powi(2)).ln() / 2.0
    }

    /// # Description
    /// Calculates the Probability Density Function (PDF) of a normal distribution.
    /// 
    /// # Input
    /// - `x`: Values at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF values at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Normal_distribution#Probability_density_function>
    /// - Original Source: N/A
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        Ok(x.map(|x_| {
            (((x_ - self.mu) / self.sigma).powi(2) / -2.0).exp() / (self.sigma * (2.0*std::f64::consts::PI).sqrt())
        } ))
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a normal distribution.
    /// 
    /// # Input
    /// - `x`: Values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function>
    /// - Original Source: N/A
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        Ok((1.0 + ((x - self.mu) / (self.sigma * 2.0_f64.sqrt())).map(|x_| erf(*x_, FunctionEvalMethod::Approximation { n_terms: 20 }) )) / 2.0)
    }

    /// # Description
    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a normal distribution.
    /// 
    /// # Input
    /// - `p`: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_} );
        Ok(self.mu + self.sigma * 2.0_f64.sqrt() * p.map(|p_| erfinv(2.0 * *p_ - 1.0, 30) ))
    }

    /// # Description
    /// Calculates the Moment Generating Function (MGF) for a normal distribution.
    /// 
    /// # Input
    /// - `t`: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given `t`
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| { 
            let x: f64 = *t_ as f64;
            (self.mu * x + 0.5 * self.sigma.powi(2) * x.powi(2)).exp()
        } )
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Methods and properties of exponential distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Exponential_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, ExponentialDistribution};
///
/// let dist: ExponentialDistribution = ExponentialDistribution::new(0.5).unwrap();
/// let x: Array1<f64> = arr1(&[0.6]);
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
/// assert!((pdf_v - 0.37040911034085894).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
/// assert!((cdf_v - 0.2591817793182821).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.2591817793182821])).unwrap()[0];
/// assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
/// ```
pub struct ExponentialDistribution {
    /// Rate parameter
    lambda: f64,
    /// Type of distribution
    _distribution_type: ProbabilityDistributionType,
}

impl ExponentialDistribution {
    /// # Description
    /// Creates a new `ExponentialDistribution` instance.
    /// 
    /// # Input
    /// - `lambda`: Rate parameter
    /// 
    /// # Errors
    /// - Returns an error if `lambda` is not positive.
    pub fn new(lambda: f64) -> Result<Self, DigiFiError> {
        if lambda <= 0.0 {
            return Err(DigiFiError::ParameterConstraint { title: "Exponential Distribution".to_owned(), constraint: "The argument `lambda` must be positive.".to_owned(), });
        }
        Ok(ExponentialDistribution { lambda, _distribution_type: ProbabilityDistributionType::Continuous })
    }
}

impl ProbabilityDistribution for ExponentialDistribution {
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

    fn entropy(&self) -> f64 {
        1.0 - self.lambda.ln()
    }

    /// # Description
    /// Calculates the Probability Density Function (PDF) for an exponential distribution.
    /// 
    /// # Input
    /// - `x`: Values at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF values at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Exponential_distribution#Probability_density_function>
    /// - Original Source: N/A
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        Ok(x.map(|x_| self.lambda * (-self.lambda * x_).exp() ))
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for an exponential distribution.
    /// 
    /// # Input
    /// - `x`: Values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Exponential_distribution#Cumulative_distribution_function>
    /// - Original Source: N/A
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        Ok(x.map(|x_| 1.0 - (-self.lambda * x_).exp() ))
    }

    /// # Description
    /// Calculates the Inverse Cumulative Distribution Function (Inverse CDF) for an exponential distribution.
    /// 
    /// # Input
    /// - `p`: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_} );
        Ok(p.map(|p_| (1.0 - p_).ln() / -self.lambda ))
    }

    /// # Description
    /// Computes the Moment Generating Function (MGF) for an exponential distribution.
    /// 
    /// # Input
    /// - `t`: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given `t`
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| if (*t_ as f64) < self.lambda { self.lambda / (self.lambda - (*t_ as f64)) } else { f64::NAN } )
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Methods and properties of Laplace distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Laplace_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, LaplaceDistribution};
///
/// let dist: LaplaceDistribution = LaplaceDistribution::new(1.0, 2.0).unwrap();
/// let x: Array1<f64> = arr1(&[0.6]);
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
// assert!((pdf_v - 0.20468268826949546).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
/// assert!((cdf_v - 0.4093653765389909).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.4093653765389909])).unwrap()[0];
/// assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
/// ```
pub struct LaplaceDistribution {
    /// Location parameter, which is the peak of the distribution
    mu: f64,
    /// Scale parameter, which controls the spread of the distribution
    b: f64,
    /// Type of distribution
    _distribution_type: ProbabilityDistributionType,
}

impl LaplaceDistribution {
    /// # Description
    /// Creates a new `LaplaceDistribution` instance.
    ///
    /// # Input
    /// - `mu`: Location parameter, which is the peak of the distribution
    /// - `b`: Scale parameter, which controls the spread of the distribution
    pub fn new(mu: f64, b: f64) -> Result<Self, DigiFiError> {
        Ok(LaplaceDistribution { mu, b, _distribution_type: ProbabilityDistributionType::Continuous })
    }
}

impl ProbabilityDistribution for LaplaceDistribution {
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

    fn entropy(&self) -> f64 {
        (2.0 * self.b * std::f64::consts::E).ln()
    }

    /// # Description
    /// Calculates the Probability Density Function (PDF) for a Laplace distribution.
    /// 
    /// # Input
    /// - `x`: Values at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF values at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Laplace_distribution#Probability_density_function>
    /// - Original Source: N/A
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        Ok(x.map(|x_| (-(x_ - self.mu).abs() / self.b).exp() / (2.0 * self.b) ))
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a Laplace distribution.
    /// 
    /// # Input
    /// - `x`: Values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given `x`
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Laplace_distribution#Cumulative_distribution_function>
    /// - Original Source: N/A
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        Ok(x.map(|x_| if *x_ <= self.b { 0.5 * ((x_ - self.mu) / self.b).exp() } else { 1.0 - 0.5 * (-(x_ - self.mu) / self.b).exp() } ))
    }

    /// # Description
    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Laplace distribution.
    /// 
    /// # Input
    /// - `p`: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_} );
        Ok(p.map(|p_| self.mu - self.b * (p_ - 0.5).signum() * (1.0 - 2.0*(p_ - 0.5).abs()).ln() ))
    }

    /// # Description
    /// Calculates the Moment Generating Function (MGF) for a Laplace distribution.
    /// 
    /// # Input
    /// - `t`: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given `t`
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| if (*t_ as f64).abs() < 1.0 / self.b {
            let x: f64 = *t_ as f64;
            (self.mu * x).exp() / (1.0 - self.b.powi(2) * x.powi(2))
        } else { f64::NAN } )
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Methods and properties of Gamma distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Gamma_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, GammaDistribution};
///
/// let dist: GammaDistribution = GammaDistribution::new(0.5, 2.0).unwrap();
/// let x: Array1<f64> = arr1(&[0.6]);
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
/// assert!((pdf_v - 0.38154528938409304).abs() < 10_000_000.0 * TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
/// assert!((cdf_v - 0.5614219739190003).abs() < 10_000_000.0 * TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.5614219739190003])).unwrap()[0];
/// assert!((icdf_v - x[0]).abs() < 10_000_000.0 * TEST_ACCURACY);
/// ```
pub struct GammaDistribution {
    /// Shape parameter, which controls the shape of the distribution
    k: f64,
    /// Scale parameter, which controls the spread of the distribution
    theta: f64,
    /// Type of distribution
    _distribution_type: ProbabilityDistributionType,
}

impl GammaDistribution {
    /// # Description
    /// Creates a new `GammaDistribution` instance.
    ///
    /// # Input
    /// - `k`: Shape parameter, which controls the shape of the distribution
    /// - `theta`: Scale parameter, which controls the spread of the distribution
    pub fn new(k: f64, theta: f64) -> Result<Self, DigiFiError> {
        let error_title: String = String::from("Gamma Distribution");
        if k <= 0.0 {
            return Err(DigiFiError::ParameterConstraint { title: error_title.clone(), constraint: "The argument `k` must be positive.".to_owned(), });
        }
        if theta <= 0.0 {
            return Err(DigiFiError::ParameterConstraint { title: error_title.clone(), constraint: "The argument `theta` must be positive.".to_owned(), });
        }
        Ok(GammaDistribution { k, theta, _distribution_type: ProbabilityDistributionType::Continuous })
    }
}

impl ProbabilityDistribution for GammaDistribution {
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

    fn entropy(&self) -> f64 {
        let method: FunctionEvalMethod = FunctionEvalMethod::Approximation { n_terms: 30 };
        let f = |z: f64| { gamma_function(z.ln(), &method) };
        self.k + self.theta.ln() + gamma_function(self.k, &method).ln() + (1.0 - self.k) * derivative(f, self.k, 0.00000001)
    }

    /// # Description
    /// Calculates the Probability Density Function (PDF) for a Gamma distribution.
    /// 
    /// # Input
    /// - `x`: Values at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF values at the given `x`
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let x_: Array1<f64> = x.map(|v| if *v < 0.0{ f64::NAN } else {*v});
        Ok(x_.map(|v| v.powf(self.k - 1.0) * (-v / self.theta).exp() / (gamma_function(self.k, &FunctionEvalMethod::Approximation { n_terms: 0 }) * self.theta.powf(self.k)) ))
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a Gamma distribution.
    /// 
    /// # Input
    /// - `x`: Values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given `x`
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let x_: Array1<f64> = x.map(|v| if *v < 0.0{ f64::NAN } else {*v});
        let mut y: Vec<f64> = Vec::<f64>::new();
        for v in x_ {
            y.push(lower_incomplete_gamma_function(self.k, v/self.theta, &FunctionEvalMethod::Approximation { n_terms: 50 })? / gamma_function(self.k, &FunctionEvalMethod::Approximation { n_terms: 30 }));
        }
        Ok(Array1::from_vec(y))
    }

    /// # Description
    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Gamma distribution.
    /// 
    /// # Input
    /// - `p`: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_} );
        let mut y: Vec<f64> = Vec::<f64>::new();
        for p_ in p {
            let f = |x: &[f64]| { (self.cdf(&Array1::from_vec(x.to_vec())).unwrap()[0] - p_).powi(2) };
            let result: f64 = nelder_mead(f, vec![0.0], Some(1_000), Some(10_000), None, None, None)?[0];
            y.push(result);
        }
        Ok(Array1::from_vec(y))
    }

    /// # Description
    /// Calculates the Moment Generating Function (MGF) for a Gamma distribution.
    /// 
    /// # Input
    /// - `t`: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given `t`
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| if (*t_ as f64).abs() < 1.0 / self.theta {
            (1.0 - self.theta * (*t_ as f64)).powf(-self.k)
        } else { f64::NAN } )
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
        let dist: ContinuousUniformDistribution = ContinuousUniformDistribution::new(0.0, 1.0).unwrap();
        let x: Array1<f64> = arr1(&[0.6]);
        // PDF test
        let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
        assert!((pdf_v - 1.0).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
        assert!((cdf_v - 0.6).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.6])).unwrap()[0];
        assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_normal_distribution() -> () {
        use crate::statistics::continuous_distributions::NormalDistribution;
        let dist: NormalDistribution = NormalDistribution::new(0.0, 1.0).unwrap();
        let x: Array1<f64> = arr1(&[0.6]);
        // PDF test
        let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
        assert!((pdf_v - 0.33322460289179967).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
        assert!((cdf_v - 0.7257468822499265).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.7257468822499265])).unwrap()[0];
        assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_exponential_distribution() -> () {
        use crate::statistics::continuous_distributions::ExponentialDistribution;
        let dist: ExponentialDistribution = ExponentialDistribution::new(0.5).unwrap();
        let x: Array1<f64> = arr1(&[0.6]);
        // PDF test
        let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
        assert!((pdf_v - 0.37040911034085894).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
        assert!((cdf_v - 0.2591817793182821).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.2591817793182821])).unwrap()[0];
        assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_laplace_distribution() -> () {
        use crate::statistics::continuous_distributions::LaplaceDistribution;
        let dist: LaplaceDistribution = LaplaceDistribution::new(1.0, 2.0).unwrap();
        let x: Array1<f64> = arr1(&[0.6]);
        // PDF test
        let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
        assert!((pdf_v - 0.20468268826949546).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
        assert!((cdf_v - 0.4093653765389909).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.4093653765389909])).unwrap()[0];
        assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_gamma_distribution() -> () {
        use crate::statistics::continuous_distributions::GammaDistribution;
        let dist: GammaDistribution = GammaDistribution::new(0.5, 2.0).unwrap();
        let x: Array1<f64> = arr1(&[0.6]);
        // PDF test
        let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
        assert!((pdf_v - 0.38154528938409304).abs() < 10_000_000.0 * TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
        assert!((cdf_v - 0.5614219739190003).abs() < 10_000_000.0 * TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.5614219739190003])).unwrap()[0];
        assert!((icdf_v - x[0]).abs() < 10_000_000.0 * TEST_ACCURACY);
    }
}