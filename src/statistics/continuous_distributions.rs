use std::io::Error;
use ndarray::Array1;
use num::complex::Complex;
use crate::utilities::{input_error, maths_utils::{erf, erfinv}};
use crate::statistics::{ProbabilityDistribution, ProbabilityDistributionType};


/// # Description
/// Methods and properties of continuous uniform distribution.
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Continuous_uniform_distribution
/// - Original Source: N/A
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
    /// Creates a new ContinuousUniformDistribution instance.
    /// 
    /// # Input
    /// - a: Lower bound of the distribution
    /// - b: Upper bound of the distribution
    /// 
    /// # Panics
    /// - Panics if the value of a is larger or equal to b
    pub fn new(a: f64, b: f64) -> Result<Self, Error> {
        if b <= a {
            return Err(input_error("The argument a must be smaller or equal to the argument b."));
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
    /// - x: Values at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF values at the given x
    /// 
    /// # Links
    /// - Wikipedia: https://en.wikipedia.org/wiki/Continuous_uniform_distribution#Probability_density_function
    /// - Original Source: N/A
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, Error> {
        Ok(x.map(|x_| if (self.a <= *x_) && ( *x_ <= self.b) {1.0 / (self.b - self.a) } else { 0.0 } ))
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a continuous uniform distribution.
    /// 
    /// # Input
    /// - x: Values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given x
    /// 
    /// # Links
    /// - Wikipedia: https://en.wikipedia.org/wiki/Continuous_uniform_distribution#Cumulative_distribution_function
    /// - Original Sourcew: N/A
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, Error> {
        Ok(x.map(|x_| if self.a <= *x_ { ((x_ - self.a) / (self.b - self.a)).min(1.0) } else { 0.0 } ))
    }

    /// # Description
    /// Calculates the Inverse Cumulative Distribution Function (Inverse CDF) for a continuous uniform distribution.
    /// 
    /// # Input
    /// - p: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, Error> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_} );
        Ok(self.a + p * (self.b - self.a))
    }

    /// # Description
    /// Computes the Moment Generating Function (MGF) for a continuous uniform distribution.
    /// 
    /// # Input
    /// - t: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given t
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| if *t_ != 0 {
            let x: f64 = *t_ as f64;
            ((x*self.b).exp() - (x*self.a).exp()) / (x*(self.b - self.a))
        } else { 1.0 } )
    }

    /// # Description
    /// Calculates the Characteristic Function (CF) for a continuous uniform distribution.
    /// 
    /// # Input
    /// - t: Input values for the CF
    /// 
    /// # Output
    /// - CF values at the given t
    fn cf(&self, t: &Array1<usize>) -> Array1<Complex<f64>> {
        t.map(|t_| if *t_ != 0 {
            let x: f64 = *t_ as f64;
            (Complex::new(0.0, x*self.b).exp() - Complex::new(0.0, x*self.a).exp()) / (x*(self.b - self.a))
        } else { Complex::new(1.0, 0.0) } )
    }
}


/// # Description
/// Methods and properties of normal distribution.
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Normal_distribution
/// - Original Source: N/A
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
    /// Creates a new NormalDistribution instance.
    /// 
    /// # Input
    /// - mu: Mean of the distribution
    /// - sigma: Standard deviation of the distribution
    /// 
    /// # Panics
    /// - Panics if sigma is negative
    pub fn new(mu: f64, sigma: f64) -> Result<Self, Error> {
        if sigma < 0.0 {
            return Err(input_error("The argument sigma must be non-negative."));
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
    /// - x: Values at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF values at the given x
    /// 
    /// # Links
    /// - Wikipedia: https://en.wikipedia.org/wiki/Normal_distribution#Probability_density_function
    /// - Original Source: N/A
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, Error> {
        Ok(x.map(|x_| {
            (((x_ - self.mu) / self.sigma).powi(2) / -2.0).exp() / (self.sigma * (2.0*std::f64::consts::PI).sqrt())
        } ))
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a normal distribution.
    /// 
    /// # Input
    /// - x: Values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given x
    /// 
    /// # Links
    /// - Wikipedia: https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
    /// - Original Source: N/A
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, Error> {
        Ok((1.0 + ((x - self.mu) / (self.sigma * 2.0_f64.sqrt())).map(|x_| erf(*x_, 15) )) / 2.0)
    }

    /// # Description
    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a normal distribution.
    /// 
    /// # Input
    /// - p: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, Error> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_} );
        Ok(self.mu + self.sigma * 2.0_f64.sqrt() * p.map(|p_| erfinv(2.0 * *p_ - 1.0, 15) ))
    }

    /// # Description
    /// Calculates the Moment Generating Function (MGF) for a normal distribution.
    /// 
    /// # Input
    /// - t: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given t
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| { 
            let x: f64 = *t_ as f64;
            (self.mu * x + 0.5 * self.sigma.powi(2) * x.powi(2)).exp()
        } )
    }

    /// # Description
    /// Computes the Characteristic Function (CF) for a normal distribution.
    /// 
    /// # Input
    /// - t: Input values for the CF
    /// 
    /// # Output
    /// - CF values at the given t
    fn cf(&self, t: &Array1<usize>) -> Array1<Complex<f64>> {
        t.map(|t_| { 
            let x: f64 = *t_ as f64;
            (Complex::new(0.0, self.mu * x) + 0.5 * self.sigma.powi(2) * x.powi(2)).exp()
        } )
    }
}


/// # Description
/// Methods and properties of exponential distribution.
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Exponential_distribution
/// - Original Source: N/A
pub struct ExponentialDistribution {
    /// Rate parameter
    lambda: f64,
    /// Type of distribution
    _distribution_type: ProbabilityDistributionType,
}

impl ExponentialDistribution {
    /// # Description
    /// Creates a new ExponentialDistribution instance.
    /// 
    /// # Input
    /// - lambda: Rate parameter
    /// 
    /// # Panics
    /// - Panics if lambda is not positive
    pub fn new(lambda: f64) -> Result<Self, Error> {
        if lambda <= 0.0 {
            return Err(input_error("The argument lambda must be positive."));
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
    /// - x: Values at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF values at the given x
    /// 
    /// # Links
    /// - Wikipedia: https://en.wikipedia.org/wiki/Exponential_distribution#Probability_density_function
    /// - Original Source: N/A
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, Error> {
        Ok(x.map(|x_| self.lambda * (-self.lambda * x_).exp() ))
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for an exponential distribution.
    /// 
    /// # Input
    /// - x: Values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given x
    /// 
    /// # Links
    /// - Wikipedia: https://en.wikipedia.org/wiki/Exponential_distribution#Cumulative_distribution_function
    /// - Original Source: N/A
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, Error> {
        Ok(x.map(|x_| 1.0 - (-self.lambda * x_).exp() ))
    }

    /// # Description
    /// Calculates the Inverse Cumulative Distribution Function (Inverse CDF) for an exponential distribution.
    /// 
    /// # Input
    /// - p: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, Error> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_} );
        Ok(p.map(|p_| (1.0 - p_).ln() / -self.lambda ))
    }

    /// # Description
    /// Computes the Moment Generating Function (MGF) for an exponential distribution.
    /// 
    /// # Input
    /// - t: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given t
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| if (*t_ as f64) < self.lambda { self.lambda / (self.lambda - (*t_ as f64)) } else { f64::NAN } )
    }

    /// # Description
    /// Calculates the Characteristic Function (CF) for an exponential distribution.
    /// 
    /// # Input
    /// - t: Input values for the CF
    /// 
    /// # Output
    /// - CF values at the given t
    fn cf(&self, t: &Array1<usize>) -> Array1<Complex<f64>> {
        t.map(|t_| self.lambda / (self.lambda - Complex::new(0.0, *t_ as f64)) )
    }
}


/// # Description
/// Methods and properties of Laplace distribution.
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Laplace_distribution
/// - Original Source: N/A
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
    /// Creates a new LaplaceDistribution instance.
    pub fn new(mu: f64, b: f64) -> Result<Self, Error> {
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
    /// - x: Values at which to calculate the PDF
    /// 
    /// # Output
    /// - PDF values at the given x
    /// 
    /// # Links
    /// - Wikipedia: https://en.wikipedia.org/wiki/Laplace_distribution#Probability_density_function
    /// - Original Source: N/A
    fn pdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, Error> {
        Ok(x.map(|x_| (-(x_ - self.mu).abs() / self.b).exp() / (2.0 * self.b) ))
    }

    /// # Description
    /// Computes the Cumulative Distribution Function (CDF) for a Laplace distribution.
    /// 
    /// # Input
    /// - x: Values at which to calculate the CDF
    /// 
    /// # Output
    /// - CDF values at the given x
    /// 
    /// # Links
    /// - Wikipedia: https://en.wikipedia.org/wiki/Laplace_distribution#Cumulative_distribution_function
    /// - Original Source: N/A
    fn cdf(&self, x: &Array1<f64>) -> Result<Array1<f64>, Error> {
        Ok(x.map(|x_| if *x_ <= self.b { 0.5 * ((x_ - self.mu) / self.b).exp() } else { 1.0 - 0.5 * (-(x_ - self.mu) / self.b).exp() } ))
    }

    /// # Description
    /// Computes the Inverse Cumulative Distribution Function (Inverse CDF) for a Laplace distribution.
    /// 
    /// # Input
    /// - p: Probability values for which to calculate the inverse CDF
    /// 
    /// # Output
    /// - Inverse CDF values for the given probabilities
    fn inverse_cdf(&self, p: &Array1<f64>) -> Result<Array1<f64>, Error> {
        let p: Array1<f64> = p.map(|p_| if (*p_ < 0.0) || (1.0 < *p_) { f64::NAN } else {*p_} );
        Ok(p.map(|p_| self.mu - self.b * (p_ - 0.5).signum() * (1.0 - 2.0*(p_ - 0.5).abs()).ln() ))
    }

    /// # Description
    /// Calculates the Moment Generating Function (MGF) for a Laplace distribution.
    /// 
    /// # Input
    /// - t: Input values for the MGF
    /// 
    /// # Output
    /// - MGF values at the given t
    fn mgf(&self, t: &Array1<usize>) -> Array1<f64> {
        t.map(|t_| if (*t_ as f64).abs() < 1.0 / self.b {
            let x: f64 = *t_ as f64;
            (self.mu * x).exp() / (1.0 - self.b.powi(2) * x.powi(2))
        } else { f64::NAN } )
    }

    /// # Description
    /// Computes the Characteristic Function (CF) for a Laplace distribution.
    /// 
    /// # Input
    /// - t: Input values for the CF
    /// 
    /// # Output
    /// - CF values at the given t
    fn cf(&self, t: &Array1<usize>) -> Array1<Complex<f64>> {
        t.map(|t_| if (*t_ as f64).abs() < 1.0 / self.b {
            let x: f64 = *t_ as f64;
            Complex::new(self.mu * x, 0.0).exp() / (1.0 - self.b.powi(2) * x.powi(2))
        } else { Complex::new(f64::NAN, 0.0) } )
    }
}


#[cfg(test)]
mod tests {
    // TODO: Add continuous distribution tests
}