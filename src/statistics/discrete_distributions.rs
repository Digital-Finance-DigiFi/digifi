use ndarray::{Array1, arr1};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;
use crate::utilities::maths_utils::factorial;
use crate::statistics::{
    ProbabilityDistribution, ProbabilityDistributionType, n_choose_r,
    beta::regularized_incomplete_beta,
    continuous_distributions::NormalDistribution,
};


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Methods and properties of Bernoulli distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Bernoulli_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, BernoulliDistribution};
///
/// let dist: BernoulliDistribution = BernoulliDistribution::new(0.4).unwrap();
/// let x: Array1<f64> = arr1(&[0.0]);
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
/// assert!((pdf_v - 0.6).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
/// assert!((cdf_v - 0.6).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.6])).unwrap()[0];
/// assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
/// ```
pub struct BernoulliDistribution {
    /// Probability of successful outcome
    p: f64,
    /// Type of distribution
    _distribution_type: ProbabilityDistributionType,
}

impl BernoulliDistribution {
    /// # Description
    /// Creates a new `BernoulliDistribution` instance.
    /// 
    /// # Input
    /// - `p`: Probability of successful outcome
    /// 
    /// # Errors
    /// - Returns an error if `p` is not in the \[0,1\] range
    pub fn new(p: f64) -> Result<Self, DigiFiError> {
        if (p < 0.0) || (1.0 < p) {
            return Err(DigiFiError::ParameterConstraint { title: "Bernoulli Distribution".to_owned(), constraint: "The argument `p` must be in the range `[0, 1]`.".to_owned(), });
        }
        Ok(BernoulliDistribution { p, _distribution_type: ProbabilityDistributionType::Discrete })
    }
}

impl ProbabilityDistribution for BernoulliDistribution {
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

    /// # Description
    /// Calculates the Probability Mass Function (PMF) for a Bernoulli distribution.
    /// 
    /// # Input
    /// - `x`: Array of values (`0` or `1`) at which to calculate the PMF
    /// 
    /// # Output
    /// - PMF values at the given `x`
    ///
    /// # Errors
    /// - Returns an error if any value inside array `x` is not in the set {0, 1}
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let mut result: Vec<f64> = Vec::<f64>::new();
        for i in x {
            let x_: f64;
            if *i == 0.0 {
                x_ = 1.0 - self.p
            } else if *i == 1.0 {
                x_ = self.p
            } else {
                return Err(DigiFiError::ParameterConstraint { title: "Bernoulli Distribution".to_owned(), constraint: "The argument `x` must be in the `{{0, 1}}` set.".to_owned(), });
            }
            result.push(x_)
        }
        Ok(Array1::from_vec(result))
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a Bernoulli distribution.
    /// 
    /// # Input
    /// - `x`: Array of values (`0` or `1`) at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given `x`
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        Ok(x.map(|x_| if *x_ < 0.0 { 0.0 } else if 1.0 <= *x_ { 1.0 } else { 1.0 - self.p } ))
    }

    /// # Description
    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Bernoulli distribution.
    /// 
    /// # Input
    /// - `p`: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_});
        Ok(p.map(|p_| if *p_ == 1.0 { 1.0 } else { 0.0 } ))
    }

    /// # Description
    /// Calculates the Moment Generating Function (MGF) for a Bernoulli distribution.
    /// 
    /// # Input
    /// - `t`: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given `t`
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| (1.0 - self.p) + self.p * (*t_ as f64).exp() )
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Methods and properties of binomial distribution.
/// 
///  Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Binomial_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, BinomialDistribution};
///
/// let dist: BinomialDistribution = BinomialDistribution::new(4, 0.6).unwrap();
/// let x: Array1<f64> = arr1(&[2.0]);
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
/// assert!((pdf_v - 0.3456).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
/// assert!((cdf_v - 0.5247999999999999).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.5248])).unwrap()[0];
/// assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
/// ```
pub struct BinomialDistribution {
    /// Number of trials
    n: usize,
    /// Probability of successful outcome
    p: f64,
    /// Type of distribution
    _distribution_type: ProbabilityDistributionType,
}

impl BinomialDistribution {
    /// # Description
    /// Creates a new `BinomialDistribution` instance.
    /// 
    /// # Input
    /// - `n`: Number of trials
    /// - `p`: Probability of successful outcome
    /// 
    /// # Errors
    /// - Returns an error if `p` is not in the \[0,1\] range
    pub fn new(n: usize, p: f64) -> Result<Self, DigiFiError> {
        if (p < 0.0) || (1.0 < p) {
            return Err(DigiFiError::ParameterConstraint { title: "Binomial Distribution".to_owned(), constraint: "The argument `p` must be in the range `[0, 1]`.".to_owned(), });
        }
        Ok(BinomialDistribution { n, p, _distribution_type: ProbabilityDistributionType::Discrete })
    }
}

impl ProbabilityDistribution for BinomialDistribution {
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


    /// # Description
    /// Calculates the Probability Mass Function (PMF) for a binomial distribution.
    /// 
    /// # Input
    /// - `x`: Array of non-negative integer values at which to calculate the PMF
    /// 
    /// # Output
    /// - PMF values at the given `x`
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let mut result: Vec<f64> = Vec::<f64>::new();
        for i in x {
            let x_: f64 = (n_choose_r(self.n as u128, *i as u128)? as f64) * self.p.powi(*i as i32) * (1.0 - self.p).powi(self.n as i32 - *i as i32);
            result.push(x_);
        }
        Ok(Array1::from_vec(result))
    }


    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a binomial distribution.
    /// 
    /// # Input
    /// - `x`: Array of non-negative integer values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given `x`
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let x_floor: Array1<f64> = x.map(|x_| x_.floor() );
        let a : Array1<f64> = self.n as f64 - &x_floor;
        let b: Array1<f64> = 1.0 + &x_floor;
        let upper_integral_limit: Array1<f64> = (1.0 - self.p) * Array1::from_vec(vec![1.0; x_floor.len()]);
        let mut cdf_: Vec<f64> = Vec::<f64>::new();
        for i in 0..x.len() {
            cdf_.push(regularized_incomplete_beta(upper_integral_limit[i], a[i], b[i])?);
        }
        Ok(Array1::from_vec(cdf_))
    }

    /// # Description
    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a binomial distribution.
    /// 
    /// # Input
    /// - `p`: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_});
        let mut result: Vec<f64> = Vec::<f64>::new();
        for p_ in p {
            if p_.is_nan() {
                result.push(p_);
                continue
            }
            let mut x: f64 = f64::NAN;
            // Iteratively solve CDF for p
            for i in 0..(self.n + 1) {
                let proxy: f64 = self.cdf(&arr1(&[i as f64]))?[0];
                if (proxy - p_).abs() < 1.0 / self.n as f64 {
                    x = i as f64;
                    break;
                }
            }
            result.push(x);
        }
        Ok(Array1::from_vec(result))
    }

    /// # Description
    /// Calculates the Moment Generating Function (MGF) for a binomial distribution.
    /// 
    /// # Input
    /// - `t`: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given `t`
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| ((1.0 - self.p) + self.p * (*t_ as f64).exp()).powi(self.n as i32) )
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Methods and properties of discrete uniform distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Discrete_uniform_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, DiscreteUniformDistribution};
///
/// let dist: DiscreteUniformDistribution = DiscreteUniformDistribution::new(1, 3).unwrap();
/// let x: Array1<f64> = arr1(&[1.0]);
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
/// assert!((pdf_v - 1.0 / 3.0).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
/// assert!((cdf_v - 1.0 / 3.0).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(&arr1(&[1.0 / 3.0])).unwrap()[0];
/// assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
/// ```
pub struct DiscreteUniformDistribution {
    /// Lower bound of the distribution
    a: i32,
    /// Upper bound of the distribution
    b: i32,
    /// `n = b - a + 1`
    n: f64,
    /// Type of distribution
    _distribution_type: ProbabilityDistributionType,
}

impl DiscreteUniformDistribution {
    /// # Discription
    /// Creates a new `DiscreteUniformDistribution` instance.
    /// 
    /// # Input
    /// - `a`: Lower bound of the distribution
    /// - `b`: Upper bound of the distribution
    /// 
    /// # Errors
    /// - Returns an error if `b` is smaller than `a`
    pub fn new(a: i32, b: i32) -> Result<Self, DigiFiError> {
        if b < a {
            return Err(DigiFiError::ParameterConstraint { title: "Discrete Uniform Distribution".to_owned(), constraint: "The argument `a` must be smaller or equal to the argument `b`.".to_owned(), });
        }
        Ok(DiscreteUniformDistribution { a, b, n: ((b - a + 1) as f64), _distribution_type: ProbabilityDistributionType::Discrete })
    }
}

impl ProbabilityDistribution for DiscreteUniformDistribution {
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

    /// # Description
    /// Calculates the Probability Mass Function (PMF) for a discrete uniform distribution.
    /// 
    /// # Input
    /// - `x`: Array of values at which to calculate the PMF
    /// 
    /// # Output
    /// - PMF values for the discrete uniform distribution
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        Ok(Array1::from_vec(vec![1.0; x.len()]) / self.n)
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a discrete uniform distribution.
    /// 
    /// # Input
    /// - `x`: Array of values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given `x`
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        Ok(x.map(|x_| if ((self.a as f64) <= *x_) || (*x_ <= (self.b as f64)) { (x_.floor() - (self.a as f64) + 1.0) / self.n } else { f64::NAN } ))
    }

    /// # Description
    /// Calculates the Inverse Cumulative Distribution Function (Inverse CDF) for a discrete uniform distribution.
    /// 
    /// # Input
    /// - `p`: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_});
        Ok(p * self.n - 1.0 + (self.a as f64))
    }

    /// # Description
    /// Computes the Moment Generating Function (MGF) for a discrete uniform distribution.
    /// 
    /// # Input
    /// - `t`: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given `t`
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| {
            let x: f64 = *t_ as f64;
            ((self.a as f64 * x).exp() - ((self.b as f64 + 1.0) * x).exp()) / (self.n * (1.0 - x.exp()))
        } )
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Methods and properties of Poisson distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Poisson_distribution>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, PoissonDistribution};
///
/// let dist: PoissonDistribution = PoissonDistribution::new(1.5).unwrap();
/// let x: Array1<f64> = arr1(&[3.0]);
///
/// // PDF test
/// let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
/// assert!((pdf_v - 0.12551071508349182).abs() < TEST_ACCURACY);
///
/// // CDF test
/// let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
/// assert!((cdf_v - 0.9343575456215499).abs() < TEST_ACCURACY);
///
/// // Inverse CDF test
/// let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.9343575456215499])).unwrap()[0];
/// assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
/// ```
pub struct PoissonDistribution {
    /// Expected number of events in a given interval
    lambda: f64,
    /// Type of distribution
    _distribution_type: ProbabilityDistributionType,
}

impl PoissonDistribution {
    /// # Description
    /// Creates a new `PoissonDistribution` instance.
    /// 
    /// # Input
    /// - `lambda`: Expected number of events in a given interval
    ///
    /// # Errors
    /// - Returns an error if `lambda` is not positive
    pub fn new(lambda: f64) -> Result<Self, DigiFiError> {
        if lambda <= 0.0 {
            return Err(DigiFiError::ParameterConstraint { title: "Poisson Distribution".to_owned(), constraint: "The argument `lambda` must be positive.".to_owned(), });
        }
        Ok(PoissonDistribution { lambda, _distribution_type: ProbabilityDistributionType::Discrete })
    }
}

impl ProbabilityDistribution for PoissonDistribution {
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

    /// # Description
    /// Calculates the Probability Mass Function (PMF) for a Poisson distribution.
    /// 
    /// # Input
    /// - `x`: Array of non-negative integer values at which to calculate the PMF
    /// 
    /// # Output
    /// - PMF values at the given `x`
    ///
    /// # Errors
    /// - Returns an error if any value inside array `x` is not positive
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let mut result: Vec<f64> = Vec::<f64>::new();
        for i in x {
            if *i < 0.0 {
                return Err(DigiFiError::ParameterConstraint { title: "Poisson Distribution".to_owned(), constraint: "The argument `x` must be positive.".to_owned(), });
            } else {
                result.push((self.lambda.powi(*i as i32) * (-self.lambda).exp()) / factorial(*i as u128) as f64)
            }
        }
        Ok(Array1::from_vec(result))
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a Poisson distribution.
    /// 
    /// # Input
    /// - `x`: Array of non-negative integer values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given `x`
    ///
    /// # Errors
    /// - Returns an error if any value inside array `x` is not positive
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let mut result: Vec<f64> = Vec::<f64>::new();
        for i in x {
            if *i < 0.0 {
                return Err(DigiFiError::ParameterConstraint { title: "Poisson Distribution".to_owned(), constraint: "The argument `x` must be positive.".to_owned(), });
            }
            let mut proxy: f64 = 0.0;
            for j in 0..(1 + i.floor() as i32) {
                proxy += self.lambda.powi(j) / factorial(j as u128) as f64;
            }
            result.push(proxy * (-self.lambda).exp());
        }
        Ok(Array1::from_vec(result))
    }

    /// # Description
    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Poisson distribution.
    /// 
    /// # Input
    /// - `p`: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the givenprobabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_});
        let normal_dist: NormalDistribution = NormalDistribution::new(0.0, 1.0)?;
        // Helper functions
        let q_n2 = |w: f64| { self.lambda + self.lambda.sqrt()*w + (1.0/3.0 + w.powi(2)/6.0) + (-w/36.0 - w.powi(3)/72.0)/self.lambda.sqrt() };
        let f = |r: f64| { (2.0 * (1.0 - r + r*r.log(10.0))).sqrt() };
        let f_1 = |v: f64| { 1.0 + v + v.powi(2)/6.0 - v.powi(3)/72.0 };
        let c_0 = |r: f64| { (f(r)* r.sqrt() / (r - 1.0)).log(10.0) / r.log(10.0) };
        // Inverse CDF algorithm
        let result: Array1<f64> = p.map(|p_| {
            if 4.0 < self.lambda {
                let w: f64 = normal_dist.inverse_cdf(&Array1::from_vec(vec![*p_])).unwrap()[0];
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
                (x + delta).floor()
            } else {
                // Bottom-up summation
                let mut delta: f64 = 0.00000000000001;
                let mut s: f64 = 1.0 - self.lambda.exp() * (p_-delta);
                delta = self.lambda.exp() * delta;
                let mut n: f64 = 0.0;
                while s < 0.0 {
                    n += 1.0;
                    s = s * n / self.lambda + 1.0;
                    delta = delta * n / self.lambda;
                }
                n
            }
        } );
        Ok(result)
    }

    /// # Description
    /// Calculates the Moment Generating Function (MGF) for a Poisson distribution.
    /// 
    /// # Input
    /// - `t`: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given `t`
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| (self.lambda * ((*t_ as f64).exp() - 1.0)).exp() )
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{Array1, arr1};
    use crate::utilities::TEST_ACCURACY;
    use crate::statistics::ProbabilityDistribution;

    #[test]
    fn unit_test_bernoulli_distribution() -> () {
        use crate::statistics::discrete_distributions::BernoulliDistribution;
        let dist: BernoulliDistribution = BernoulliDistribution::new(0.4).unwrap();
        let x: Array1<f64> = arr1(&[0.0]);
        // PDF test
        let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
        assert!((pdf_v - 0.6).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
        assert!((cdf_v - 0.6).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.6])).unwrap()[0];
        assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_binomial_distribution() -> () {
        use crate::statistics::discrete_distributions::BinomialDistribution;
        let dist: BinomialDistribution = BinomialDistribution::new(4, 0.6).unwrap();
        let x: Array1<f64> = arr1(&[2.0]);
        // PDF test
        let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
        assert!((pdf_v - 0.3456).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
        assert!((cdf_v - 0.5247999999999999).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.5248])).unwrap()[0];
        assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_discrete_uniform_distribution() -> () {
        use crate::statistics::discrete_distributions::DiscreteUniformDistribution;
        let dist: DiscreteUniformDistribution = DiscreteUniformDistribution::new(1, 3).unwrap();
        let x: Array1<f64> = arr1(&[1.0]);
        // PDF test
        let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
        assert!((pdf_v - 1.0 / 3.0).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
        assert!((cdf_v - 1.0 / 3.0).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(&arr1(&[1.0 / 3.0])).unwrap()[0];
        assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_poisson_distribution() -> () {
        use crate::statistics::discrete_distributions::PoissonDistribution;
        let dist: PoissonDistribution = PoissonDistribution::new(1.5).unwrap();
        let x: Array1<f64> = arr1(&[3.0]);
        // PDF test
        let pdf_v: f64 = dist.pdf(&x).unwrap()[0];
        assert!((pdf_v - 0.12551071508349182).abs() < TEST_ACCURACY);
        // CDF test
        let cdf_v: f64 = dist.cdf(&x).unwrap()[0];
        assert!((cdf_v - 0.9343575456215499).abs() < TEST_ACCURACY);
        // Inverse CDF test
        let icdf_v: f64 = dist.inverse_cdf(&arr1(&[0.9343575456215499])).unwrap()[0];
        assert!((icdf_v - x[0]).abs() < TEST_ACCURACY);
    }
}