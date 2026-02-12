#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::maths_utils::factorial;
use crate::statistics::{
    ProbabilityDistribution, ProbabilityDistributionType, n_choose_r,
    beta::regularized_incomplete_beta,
    continuous_distributions::NormalDistribution,
};


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Methods and properties of Bernoulli distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Bernoulli_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, BernoulliDistribution};
///
/// let dist: BernoulliDistribution = BernoulliDistribution::build(0.4).unwrap();
/// let x: f64 = 0.0;
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
/// assert!((pdf_v - 0.6).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 0.6).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(0.6).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// ```
pub struct BernoulliDistribution {
    /// Probability of successful outcome
    p: f64,
}

impl BernoulliDistribution {
    /// Creates a new `BernoulliDistribution` instance.
    /// 
    /// # Input
    /// - `p`: Probability of successful outcome
    /// 
    /// # Errors
    /// - Returns an error if `p` is not in the \[0,1\] range
    pub fn build(p: f64) -> Result<Self, DigiFiError> {
        if (p < 0.0) || (1.0 < p) {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `p` must be in the range `[0, 1]`.".to_owned(),
            });
        }
        Ok(Self { p, })
    }
}

impl ErrorTitle for BernoulliDistribution {
    fn error_title() -> String {
        String::from("Bernoulli Distribution")
    }
}

impl ProbabilityDistribution for BernoulliDistribution {
    fn distribution_type() -> ProbabilityDistributionType {
        ProbabilityDistributionType::Discrete
    }

    fn mean(&self) -> f64 {
        self.p
    }

    fn median(&self) -> Vec<f64> {
        if self.p < 0.5 { vec![0.0] } else if 0.5 < self.p { vec![1.0] } else { vec![0.0, 1.0] }
    }

    fn mode(&self) -> Vec<f64> {
        if self.p < 0.5 { vec![0.0] } else if 0.5 < self.p { vec![1.0] } else { vec![0.0, 1.0] }
    }

    fn variance(&self) -> f64 {
        self.p * (1.0 - self.p)
    }

    fn skewness(&self) -> f64 {
        ((1.0 - self.p) - self.p) / ((1.0 - self.p) * self.p).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        (1.0 - 6.0 * (1.0 - self.p) * self.p) / ((1.0 - self.p) * self.p)
    }

    fn entropy(&self) -> Result<f64, DigiFiError> {
        Ok(-(1.0 - self.p) * (1.0 - self.p).ln() - self.p * self.p.ln())
    }

    /// Calculates the Probability Mass Function (PMF) for a Bernoulli distribution.
    /// 
    /// # Input
    /// - `x`: Value (`0` or `1`) at which to calculate the PMF
    /// 
    /// # Output
    /// - PMF value at the given `x`
    ///
    /// # Errors
    /// - Returns an error if the value `x` is not in the set {0, 1}
    fn pdf(&self, x: f64) -> Result<f64, DigiFiError> {
        match x {
            0.0 => Ok(1.0 - self.p),
            1.0 => Ok(self.p),
            _ => Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `x` must be in the `{{0, 1}}` set.".to_owned(),
            }),
        }
    }

    /// Computes the Cumulative Distribution Function (CDF) for a Bernoulli distribution.
    /// 
    /// # Input
    /// - `x`: Value (`0` or `1`) at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF value at the given `x`
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok(if x < 0.0 { 0.0 } else if 1.0 <= x { 1.0 } else { 1.0 - self.p })
    }

    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Bernoulli distribution.
    /// 
    /// # Input
    /// - `p`: Probability value for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF value for the given probabilities
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError> {
        if (p < 0.0) || (1.0 < p) { return Ok(f64::NAN) }
        if p == 1.0 { Ok(1.0) } else { Ok(0.0) }
    }

    /// Calculates the Moment Generating Function (MGF) for a Bernoulli distribution.
    /// 
    /// # Input
    /// - `t`: Input value for the MGF
    /// 
    /// # Output
    /// - MGF value at the given `t`
    fn mgf(&self, t: usize) -> f64 {
        (1.0 - self.p) + self.p * (t as f64).exp()
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Methods and properties of binomial distribution.
/// 
///  Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Binomial_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, BinomialDistribution};
///
/// let dist: BinomialDistribution = BinomialDistribution::build(4, 0.6).unwrap();
/// let x: f64 = 2.0;
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
/// assert!((pdf_v - 0.3456).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 0.5247999999999999).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(0.5248).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// ```
pub struct BinomialDistribution {
    /// Number of trials
    n: usize,
    /// Probability of successful outcome
    p: f64,
}

impl BinomialDistribution {
    /// Creates a new `BinomialDistribution` instance.
    /// 
    /// # Input
    /// - `n`: Number of trials
    /// - `p`: Probability of successful outcome
    /// 
    /// # Errors
    /// - Returns an error if `p` is not in the \[0,1\] range
    pub fn build(n: usize, p: f64) -> Result<Self, DigiFiError> {
        if (p < 0.0) || (1.0 < p) {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `p` must be in the range `[0, 1]`.".to_owned(),
            });
        }
        Ok(Self { n, p, })
    }
}

impl ErrorTitle for BinomialDistribution {
    fn error_title() -> String {
        String::from("Binomial Distribution")
    }
}

impl ProbabilityDistribution for BinomialDistribution {
    fn distribution_type() -> ProbabilityDistributionType {
        ProbabilityDistributionType::Discrete
    }

    fn mean(&self) -> f64 {
        (self.n as f64) * self.p
    }

    fn median(&self) -> Vec<f64> {
        vec![((self.n as f64) * self.p).floor(), ((self.n as f64) * self.p).ceil()]
    }

    fn mode(&self) -> Vec<f64> {
        vec![((self.n as f64 + 1.0) * self.p).floor(), ((self.n as f64 + 1.0) * self.p).ceil() - 1.0]
    }

    fn variance(&self) -> f64 {
        (self.n as f64) * self.p * (1.0 - self.p)
    }

    fn skewness(&self) -> f64 {
        ((1.0 - self.p) - self.p) / ((self.n as f64) * self.p * (1.0 - self.p)).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        (1.0 - 6.0 * (1.0 - self.p) * self.p) / ((self.n as f64) * (1.0 - self.p) * self.p)
    }

    fn entropy(&self) -> Result<f64, DigiFiError> {
        Ok(0.5 * (2.0 * std::f64::consts::PI  * std::f64::consts::E * (self.n as f64) * self.p * (1.0 - self.p)).ln())
    }


    /// Calculates the Probability Mass Function (PMF) for a binomial distribution.
    /// 
    /// # Input
    /// - `x`: Non-negative integer value at which to calculate the PMF
    /// 
    /// # Output
    /// - PMF value at the given `x`
    fn pdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok((n_choose_r(self.n as u128, x as u128)? as f64) * self.p.powi(x as i32) * (1.0 - self.p).powi(self.n as i32 - x as i32))
    }


    /// Computes the Cumulative Distribution Function (CDF) for a binomial distribution.
    /// 
    /// # Input
    /// - `x`: Non-negative integer value at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF value at the given `x`
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError> {
        let x_floor: f64 = x.floor();
        let a: f64 = self.n as f64 - x_floor;
        let b: f64 = 1.0 + x_floor;
        let upper_integral_limit: f64 = 1.0 - self.p;
        regularized_incomplete_beta(upper_integral_limit, a, b)

    }

    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a binomial distribution.
    /// 
    /// # Input
    /// - `p`: Probability value for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF value for the given probabilities
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError> {
        if (p < 0.0) || (1.0 < p) { return Ok(f64::NAN) }
        let mut x: f64 = f64::NAN;
        // Iteratively solve CDF for p
        for i in 0..(self.n + 1) {
            let proxy: f64 = self.cdf(i as f64)?;
            if (proxy - p).abs() < 1.0 / self.n as f64 {
                x = i as f64;
                break;
            }
        }
        Ok(x)
    }

    /// Calculates the Moment Generating Function (MGF) for a binomial distribution.
    /// 
    /// # Input
    /// - `t`: Input value for the MGF
    /// 
    /// # Output
    /// - MGF value at the given `t`
    fn mgf(&self, t: usize) -> f64 {
        ((1.0 - self.p) + self.p * (t as f64).exp()).powi(self.n as i32)
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Methods and properties of discrete uniform distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Discrete_uniform_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, DiscreteUniformDistribution};
///
/// let dist: DiscreteUniformDistribution = DiscreteUniformDistribution::build(1, 3).unwrap();
/// let x: f64 = 1.0;
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
/// assert!((pdf_v - 1.0 / 3.0).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 1.0 / 3.0).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(1.0 / 3.0).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// ```
pub struct DiscreteUniformDistribution {
    /// Lower bound of the distribution
    a: i32,
    /// Upper bound of the distribution
    b: i32,
    /// `n = b - a + 1`
    n: f64,
}

impl DiscreteUniformDistribution {

    /// Creates a new `DiscreteUniformDistribution` instance.
    /// 
    /// # Input
    /// - `a`: Lower bound of the distribution
    /// - `b`: Upper bound of the distribution
    /// 
    /// # Errors
    /// - Returns an error if `b` is smaller than `a`
    pub fn build(a: i32, b: i32) -> Result<Self, DigiFiError> {
        if b < a {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `a` must be smaller or equal to the argument `b`.".to_owned(),
            });
        }
        Ok(Self { a, b, n: ((b - a + 1) as f64), })
    }
}

impl ErrorTitle for DiscreteUniformDistribution {
    fn error_title() -> String {
        String::from("Discrete Uniform Distribution")
    }
}

impl ProbabilityDistribution for DiscreteUniformDistribution {
    fn distribution_type() -> ProbabilityDistributionType {
        ProbabilityDistributionType::Discrete
    }

    fn mean(&self) -> f64 {
        (self.a + self.b) as f64 / 2.0
    }

    fn median(&self) -> Vec<f64> {
        vec![(self.a + self.b) as f64 / 2.0]
    }

    fn mode(&self) -> Vec<f64> {
        vec![f64::NAN]
    }

    fn variance(&self) -> f64 {
        (self.n.powi(2) - 1.0) / 12.0
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn excess_kurtosis(&self) -> f64 {
        -6.0 * (self.n.powi(2) + 1.0) / (5.0 * (self.n.powi(2) - 1.0))
    }

    fn entropy(&self) -> Result<f64, DigiFiError> {
        Ok(self.n.ln())
    }

    /// Calculates the Probability Mass Function (PMF) for a discrete uniform distribution.
    /// 
    /// # Input
    /// - `x`: Values at which to calculate the PMF
    /// 
    /// # Output
    /// - PMF value for the discrete uniform distribution
    fn pdf(&self, _x: f64) -> Result<f64, DigiFiError> {
        Ok(1.0 / self.n)
    }

    /// Computes the Cumulative Distribution Function (CDF) for a discrete uniform distribution.
    /// 
    /// # Input
    /// - `x`: Value at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF value at the given `x`
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError> {
        Ok(if ((self.a as f64) <= x) || (x <= (self.b as f64)) { (x.floor() - (self.a as f64) + 1.0) / self.n } else { f64::NAN })
    }

    /// Calculates the Inverse Cumulative Distribution Function (Inverse CDF) for a discrete uniform distribution.
    /// 
    /// # Input
    /// - `p`: Probability value for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF value for the given probabilities
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError> {
        if (p < 0.0) || (1.0 < p) { return Ok(f64::NAN) }
        Ok(p * self.n - 1.0 + (self.a as f64))
    }

    /// Computes the Moment Generating Function (MGF) for a discrete uniform distribution.
    /// 
    /// # Input
    /// - `t`: Input value for the MGF
    /// 
    /// # Output
    /// - MGF value at the given `t`
    fn mgf(&self, t: usize) -> f64 {
        let x: f64 = t as f64;
        ((self.a as f64 * x).exp() - ((self.b as f64 + 1.0) * x).exp()) / (self.n * (1.0 - x.exp()))
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Methods and properties of Poisson distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Poisson_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, PoissonDistribution};
///
/// let dist: PoissonDistribution = PoissonDistribution::build(1.5).unwrap();
/// let x: f64 = 3.0;
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(x).unwrap();
/// assert!((pdf_v - 0.12551071508349182).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(x).unwrap();
/// assert!((cdf_v - 0.9343575456215499).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(0.9343575456215499).unwrap();
/// assert!((icdf_v - x).abs() < TEST_ACCURACY);
/// ```
pub struct PoissonDistribution {
    /// Expected number of events in a given interval
    lambda: f64,
}

impl PoissonDistribution {
    /// Creates a new `PoissonDistribution` instance.
    /// 
    /// # Input
    /// - `lambda`: Expected number of events in a given interval
    ///
    /// # Errors
    /// - Returns an error if `lambda` is not positive
    pub fn build(lambda: f64) -> Result<Self, DigiFiError> {
        if lambda <= 0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `lambda` must be positive.".to_owned(),
            });
        }
        Ok(Self { lambda, })
    }
}

impl ErrorTitle for PoissonDistribution {
    fn error_title() -> String {
        String::from("Poisson Distribution")
    }
}

impl ProbabilityDistribution for PoissonDistribution {
    fn distribution_type() -> ProbabilityDistributionType {
        ProbabilityDistributionType::Discrete
    }

    fn mean(&self) -> f64 {
        self.lambda
    }

    fn median(&self) -> Vec<f64> {
        vec![(self.lambda + 1.0/3.0 - 1.0 / (50.0 * self.lambda)).floor()]
    }

    fn mode(&self) -> Vec<f64> {
        vec![self.lambda.ceil() - 1.0, self.lambda.floor()]
    }

    fn variance(&self) -> f64 {
        self.lambda
    }

    fn skewness(&self) -> f64 {
        1.0 / self.lambda.sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        1.0 / self.lambda
    }

    fn entropy(&self) -> Result<f64, DigiFiError> {
        Ok(0.5*(2.0 * std::f64::consts::PI * std::f64::consts::E * self.lambda).ln() - 1.0/(12.0 * self.lambda) - 1.0/(24.0 * self.lambda.powi(2)) - 1.0/(360.0*self.lambda.powi(3)))
    }

    /// Calculates the Probability Mass Function (PMF) for a Poisson distribution.
    /// 
    /// # Input
    /// - `x`: Non-negative integer value at which to calculate the PMF
    /// 
    /// # Output
    /// - PMF value at the given `x`
    ///
    /// # Errors
    /// - Returns an error if he value`x` is not positive
    fn pdf(&self, x: f64) -> Result<f64, DigiFiError> {
        if x < 0.0 {
            Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `x` must be positive.".to_owned(),
            })
        } else {
            Ok((self.lambda.powi(x as i32) * (-self.lambda).exp()) / factorial(x as u128) as f64)
        }
    }

    /// Computes the Cumulative Distribution Function (CDF) for a Poisson distribution.
    /// 
    /// # Input
    /// - `x`: Non-negative integer value at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF value at the given `x`
    ///
    /// # Errors
    /// - Returns an error if the value `x` is not positive
    fn cdf(&self, x: f64) -> Result<f64, DigiFiError> {
        if x < 0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `x` must be positive.".to_owned(),
            });
        }
        let mut proxy: f64 = 0.0;
        for j in 0..(1 + x.floor() as i32) {
            proxy += self.lambda.powi(j) / factorial(j as u128) as f64;
        }
        Ok(proxy * (-self.lambda).exp())
    }

    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Poisson distribution.
    /// 
    /// # Input
    /// - `p`: Probability value for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF value for the givenprobabilities
    fn inverse_cdf(&self, p: f64) -> Result<f64, DigiFiError> {
        if (p < 0.0) || (1.0 < p) { return Ok(f64::NAN) }
        let normal_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0)?;
        // Helper functions
        let q_n2 = |w: f64| { self.lambda + self.lambda.sqrt()*w + (1.0/3.0 + w.powi(2)/6.0) + (-w/36.0 - w.powi(3)/72.0)/self.lambda.sqrt() };
        let f = |r: f64| { (2.0 * (1.0 - r + r*r.log(10.0))).sqrt() };
        let f_1 = |v: f64| { 1.0 + v + v.powi(2)/6.0 - v.powi(3)/72.0 };
        let c_0 = |r: f64| { (f(r)* r.sqrt() / (r - 1.0)).log(10.0) / r.log(10.0) };
        // Inverse CDF algorithm
        if 4.0 < self.lambda {
            let w: f64 = normal_dist.inverse_cdf(p).unwrap();
            let mut x: f64;
            let delta: f64;
            if w.abs() < 3.0 {
                // Normal asymptotic approximation
                x = q_n2(w);
                delta = (1.0/40.0 + w.powi(2)/80.0 + w.powi(4)/160.0) / self.lambda;
            } else {
                // Temme asymptotic approximation
                let r: f64 = f_1(w / self.lambda.sqrt());
                x = self.lambda*r + c_0(r);
                x = x - 0.0218 / (x + 0.065*self.lambda);
                delta = 0.01 / self.lambda;
            }
            Ok((x + delta).floor())
        } else {
            // Bottom-up summation
            let mut delta: f64 = 0.00000000000001;
            let mut s: f64 = 1.0 - self.lambda.exp() * (p - delta);
            delta = self.lambda.exp() * delta;
            let mut n: f64 = 0.0;
            while s < 0.0 {
                n += 1.0;
                s = s * n / self.lambda + 1.0;
                delta = delta * n / self.lambda;
            }
            Ok(n)
        }
    }

    /// Calculates the Moment Generating Function (MGF) for a Poisson distribution.
    /// 
    /// # Input
    /// - `t`: Input value for the MGF
    /// 
    /// # Output
    /// - MGF value at the given `t`
    fn mgf(&self, t: usize) -> f64 {
        (self.lambda * ((t as f64).exp() - 1.0)).exp()
    }
}


#[cfg(test)]
mod tests {
    use crate::utilities::TEST_ACCURACY;
    use crate::statistics::ProbabilityDistribution;

    #[test]
    fn unit_test_bernoulli_distribution() -> () {
        use crate::statistics::discrete_distributions::BernoulliDistribution;
        let dist: BernoulliDistribution = BernoulliDistribution::build(0.4).unwrap();
        let x: f64 = 0.0;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 0.6).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 0.6).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(0.6).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_binomial_distribution() -> () {
        use crate::statistics::discrete_distributions::BinomialDistribution;
        let dist: BinomialDistribution = BinomialDistribution::build(4, 0.6).unwrap();
        let x: f64 = 2.0;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 0.3456).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 0.5247999999999999).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(0.5248).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_discrete_uniform_distribution() -> () {
        use crate::statistics::discrete_distributions::DiscreteUniformDistribution;
        let dist: DiscreteUniformDistribution = DiscreteUniformDistribution::build(1, 3).unwrap();
        let x: f64 = 1.0;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 1.0 / 3.0).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 1.0 / 3.0).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(1.0 / 3.0).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_poisson_distribution() -> () {
        use crate::statistics::discrete_distributions::PoissonDistribution;
        let dist: PoissonDistribution = PoissonDistribution::build(1.5).unwrap();
        let x: f64 = 3.0;
        // PDF test
        let pdf_v: f64 = dist.pdf(x).unwrap();
        assert!((pdf_v - 0.12551071508349182).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(x).unwrap();
        assert!((cdf_v - 0.9343575456215499).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(0.9343575456215499).unwrap();
        assert!((icdf_v - x).abs() < TEST_ACCURACY);
    }
}