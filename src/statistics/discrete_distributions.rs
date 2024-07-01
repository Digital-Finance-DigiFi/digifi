use ndarray::Array1;
use num::complex::Complex;
use crate::utilities::maths_utils::factorial;
use crate::statistics::{ProbabilityDistribution, ProbabilityDistributionType, n_choose_r};
use crate::statistics::continuous_distributions::NormalDistribution;


/// # Description
/// Methods and properties of Bernoulli distribution.
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Bernoulli_distribution
/// - Original Source: N/A
pub struct BernoulliDistribution {
    /// Probability of successful outcome
    p: f64,
    /// Type of distribution
    _distribution_type: ProbabilityDistributionType,
}

impl BernoulliDistribution {
    /// # Description
    /// Creates a new BernoulliDistribution instance.
    /// 
    /// # Input
    /// - p: Probability of successful outcome
    /// 
    /// # Panics
    /// - Panics if p is not in the [0,1] range
    pub fn new(p: f64) -> Self {
        if (p < 0.0) || (1.0 < p) {
            panic!("The argument p must be in the [0, 1] range.");
        }
        BernoulliDistribution { p, _distribution_type: ProbabilityDistributionType::Discrete }
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

    fn entropy(&self) -> f64 {
        -(1.0 - self.p) * (1.0 - self.p).ln() - self.p * self.p.ln()
    }

    /// # Description
    /// Calculates the Probability Mass Function (PMF) for a Bernoulli distribution.
    /// 
    /// # Input
    /// - x: Array of values (0 or 1) at which to calculate the PMF
    /// 
    /// # Output
    /// - PMF values at the given x
    fn pdf(&self, x: &Array1<f64>) -> Array1<f64> {
        x.map(|x_| if *x_ == 0.0 { 1.0 - self.p } else if *x_ == 1.0 { self.p } else { panic!("The argument x must be in the {{0, 1}} set.") } )
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a Bernoulli distribution.
    /// 
    /// # Input
    /// - x: Array of values (0 or 1) at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given x
    fn cdf(&self, x: &Array1<f64>) -> Array1<f64> {
        x.map(|x_| if *x_ < 0.0 { 0.0 } else if 1.0 <= *x_ { 1.0 } else { 1.0 - self.p } )
    }

    /// # Description
    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Bernoulli distribution.
    /// 
    /// # Input
    /// - p: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Array1<f64> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_});
        p.map(|p_| if *p_ == 1.0 { 1.0 } else { 0.0 } )
    }

    /// # Description
    /// Calculates the Moment Generating Function (MGF) for a Bernoulli distribution.
    /// 
    /// # Input
    /// - t: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given t
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| (1.0 - self.p) + self.p * (*t_ as f64).exp() )
    }

    /// # Description
    /// Computes the Characteristic Function (CF) for a Bernoulli distribution.
    /// 
    /// # Input
    /// - t: Input values for the CF
    /// 
    /// # Output
    /// - CF values at the given t
    fn cf(&self, t: &Array1<usize>) -> Array1<Complex<f64>> {
        t.map(|t_| (1.0 - self.p) + self.p * Complex::new(0.0, *t_ as f64).exp() )
    }
}


/// # Description
/// Methods and properties of binomial distribution.
/// 
///  Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Binomial_distribution
/// - Original Source: N/A
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
    /// Creates a new BinomialDistribution instance.
    /// 
    /// # Input
    /// - n: Number of trials
    /// - p: Probability of successful outcome
    /// 
    /// # Panics
    /// - Panics if p is not in the [0,1] range
    pub fn new(n: usize, p: f64) -> Self {
        if (p < 0.0) || (1.0 < p) {
            panic!("The argument p must be in the [0, 1] range.");
        }
        BinomialDistribution { n, p, _distribution_type: ProbabilityDistributionType::Discrete }
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

    fn entropy(&self) -> f64 {
        0.5 * (2.0 * std::f64::consts::PI  * std::f64::consts::E * (self.n as f64) * self.p * (1.0 - self.p)).ln()
    }


    /// # Description
    /// Calculates the Probability Mass Function (PMF) for a binomial distribution.
    /// 
    /// # Input
    /// - x: Array of non-negative integer values at which to calculate the PMF
    /// 
    /// # Output
    /// - PMF values at the given x
    fn pdf(&self, x: &Array1<f64>) -> Array1<f64> {
        x.map(|x_| (n_choose_r(self.n as u128, *x_ as u128) as f64) * self.p.powi(*x_ as i32) * (1.0 - self.p).powi(self.n as i32 - *x_ as i32) )
    }


    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a binomial distribution.
    /// 
    /// # Input
    /// - x: Array of non-negative integer values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given x
    fn cdf(&self, x: &Array1<f64>) -> Array1<f64> {
        let x_floor: Array1<f64> = x.map(|x_| x_.floor() );
        let a : Array1<f64> = self.n as f64 - &x_floor;
        let b: Array1<f64> = 1.0 - &x_floor;
        let upper_integral_limit: Array1<f64> = (1.0 - self.p) * Array1::from_vec(vec![1.0; x_floor.len()]);
        // TODO: Implement incomplete beta function
        // betainc(a, b, upper_integral_limit)
        a
    }

    /// # Description
    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a binomial distribution.
    /// 
    /// # Input
    /// - p: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Array1<f64> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_});
        let mut result: Vec<f64> = Vec::<f64>::new();
        for p_ in p {
            if p_ == f64::NAN {
                result.push(p_);
                continue
            }
            let quantile = |rv: f64| { self.cdf(&Array1::from_vec(vec![rv])).map(|cdf_| if *cdf_ == f64::NAN { 0.0 } else { *cdf_ } )[0] - p_ };
            let x_0: f64 = 0.5;
            // TODO: Implement numerical solver
            // result.push(fsolve(quantile, x_0));
        }
        Array1::from_vec(result)
    }

    /// # Description
    /// Calculates the Moment Generating Function (MGF) for a binomial distribution.
    /// 
    /// # Input
    /// - t: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given t
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| ((1.0 - self.p) + self.p * (*t_ as f64).exp()).powi(self.n as i32) )
    }

    /// # Description
    fn cf(&self, t: &Array1<usize>) -> Array1<Complex<f64>> {
        t.map(|t_| ((1.0 - self.p) + self.p * Complex::new(0.0, *t_ as f64).exp()).powi(self.n as i32) )
    }
}


/// # Description
/// Methods and properties of discrete uniform distribution.
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Discrete_uniform_distribution
/// - Original Source: N/A
pub struct DiscreteUniformDistribution {
    /// Lower bound of the distribution
    a: i32,
    /// Upper bound of the distribution
    b: i32,
    /// n = b - a + 1
    n: f64,
    /// Type of distribution
    _distribution_type: ProbabilityDistributionType,
}

impl DiscreteUniformDistribution {
    /// # Discription
    /// Creates a new DiscreteUniformDistribution instance.
    /// 
    /// # Input
    /// - a: Lower bound of the distribution
    /// - b: Upper bound of the distribution
    /// 
    /// # Panics
    /// - Panics if b is smaller than a
    pub fn new(a: i32, b: i32) -> Self {
        if b < a {
            panic!("The argument a must be smaller or equal to the argument b.");
        }
        DiscreteUniformDistribution { a, b, n: ((b - a + 1) as f64), _distribution_type: ProbabilityDistributionType::Discrete }
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

    fn entropy(&self) -> f64 {
        self.n.ln()
    }

    /// # Description
    /// Calculates the Probability Mass Function (PMF) for a discrete uniform distribution.
    /// 
    /// # Input
    /// - x: Array of values at which to calculate the PMF
    /// 
    /// # Output
    /// - PMF values for the discrete uniform distribution
    fn pdf(&self, x: &Array1<f64>) -> Array1<f64> {
        Array1::from_vec(vec![1.0; x.len()]) / self.n
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a discrete uniform distribution.
    /// 
    /// # Input
    /// - x: Array of values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given x
    fn cdf(&self, x: &Array1<f64>) -> Array1<f64> {
        x.map(|x_| if ((self.a as f64) <= *x_) || (*x_ <= (self.b as f64)) { (x_.floor() - (self.a as f64) + 1.0) / self.n } else { f64::NAN } )
    }

    /// # Description
    /// Calculates the Inverse Cumulative Distribution Function (Inverse CDF) for a discrete uniform distribution.
    /// 
    /// # Input
    /// - p: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Array1<f64> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_});
        p * self.n - 1.0 + (self.a as f64)
    }

    /// # Description
    /// Computes the Moment Generating Function (MGF) for a discrete uniform distribution.
    /// 
    /// # Input
    /// - t: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given t
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| {
            let x: f64 = *t_ as f64;
            ((self.a as f64 * x).exp() - ((self.b as f64 + 1.0) * x).exp()) / (self.n * (1.0 - x.exp()))
        } )
    }

    /// # Description
    /// Calculates the Characteristic Function (CF) for a discrete uniform distribution.
    /// 
    /// # Input
    /// - t: Input values for the CF
    /// 
    /// # Output
    /// - CF values at the given t
    fn cf(&self, t: &Array1<usize>) -> Array1<Complex<f64>> {
        t.map(|t_| {
            let x: f64 = *t_ as f64;
            (Complex::new(0.0, self.a as f64 * x).exp() - Complex::new(0.0, (self.b as f64 + 1.0) * x).exp()) / (self.n * (1.0 - Complex::new(0.0, x).exp()))
        } )
    }
}


/// # Description
/// Methods and properties of Poisson distribution.
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Poisson_distribution
/// - Original SOurce: N/A
pub struct PoissonDistribution {
    /// Expected number of events in a given interval
    lambda: f64,
    /// Type of distribution
    _distribution_type: ProbabilityDistributionType,
}

impl PoissonDistribution {
    /// # Description
    /// Creates a new PoissonDistribution instance.
    /// 
    /// # Input
    /// - lambda: Expected number of events in a given interval
    pub fn new(lambda: f64) -> Self {
        if lambda <= 0.0 {
            panic!("The argument lambda must be positive.");
        }
        PoissonDistribution { lambda, _distribution_type: ProbabilityDistributionType::Discrete }
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

    fn entropy(&self) -> f64 {
        0.5*(2.0 * std::f64::consts::PI * std::f64::consts::E * self.lambda).ln() - 1.0/(12.0 * self.lambda) - 1.0/(24.0 * self.lambda.powi(2)) - 1.0/(360.0*self.lambda.powi(3))
    }

    /// # Description
    /// Calculates the Probability Mass Function (PMF) for a Poisson distribution.
    /// 
    /// # Input
    /// - x: Array of non-negative integer values at which to calculate the PMF
    /// 
    /// # Output
    /// - PMF values at the given x
    fn pdf(&self, x: &Array1<f64>) -> Array1<f64> {
        x.map(|x_| if *x_ < 0.0 { panic!("The argument x must be positive."); } else { (self.lambda.powi(*x_ as i32) * (-self.lambda).exp()) / factorial(*x_ as u128) as f64 } )
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a Poisson distribution.
    /// 
    /// # Input
    /// - x: Array of non-negative integer values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given x
    fn cdf(&self, x: &Array1<f64>) -> Array1<f64> {
        x.map(|x_| {
            if *x_ < 0.0 {
                panic!("The argument x must be positive.");
            }
            let mut result: f64 = 0.0;
            for i in 0..(x_.floor() as i32) {
                result += self.lambda.powi(i) / factorial(i as u128) as f64;
            }
            result * (-self.lambda).exp()
        } )
    }

    /// # Description
    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Poisson distribution.
    /// 
    /// # Input
    /// - p: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the givenprobabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Array1<f64> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_});
        let w: Array1<f64> = NormalDistribution::new(0.0, 1.0).inverse_cdf(&p);
        let w_2: Array1<f64> = w.map(|w_| w_.powi(2) );
        let w_3: Array1<f64> = w.map(|w_| w_.powi(3) );
        let w_4: Array1<f64> = w.map(|w_| w_.powi(4) );
        self.lambda + self.lambda.sqrt()*&w + (1.0/3.0 + &w_2/6.0) + (-&w/36.0 - w_3/72.0)/self.lambda.sqrt() + (-8.0/405.0 + 7.0*&w_2/810.0 + w_4/270.0)/self.lambda
    }

    /// # Description
    /// Calculates the Moment Generating Function (MGF) for a Poisson distribution.
    /// 
    /// # Input
    /// - t: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given t
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| (self.lambda * ((*t_ as f64).exp() - 1.0)).exp() )
    }

    /// # Description
    /// Computes the Characteristic Function (CF) for a Poisson distribution.
    /// 
    /// # Input
    /// - t: Input values for the CF
    /// 
    /// # Output
    /// - CF values at the given t
    fn cf(&self, t: &Array1<usize>) -> Array1<Complex<f64>> {
        t.map(|t_| (self.lambda * (Complex::new(0.0, *t_ as f64).exp() - 1.0)).exp() )
    }
}


#[cfg(test)]
mod tests {
    // TODO: Add discrete distribution tests
}