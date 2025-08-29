use ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;
use crate::random_generators::RandomGenerator;
use crate::stochastic_processes::StochasticProcess;
use crate::random_generators::{standard_normal_generators::StandardNormalInverseTransform, generator_algorithms::inverse_transform};
use crate::statistics::continuous_distributions::GammaDistribution;


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Model used to reproduce the volatility smile effect.
/// 
/// # LaTeX Formula
/// - dS_{t} = \\mu*S_{t}*dt + \\sigma*S^{beta+1}_{t}*dW_{t}
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Constant_elasticity_of_variance_model>
/// - Original Source: <https://doi.org/10.3905/jpm.1996.015>
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::stochastic_processes::{StochasticProcess, ConstantElasticityOfVariance};
///
/// let n_paths: usize = 100;
/// let n_steps: usize = 200;
///
/// let cev: ConstantElasticityOfVariance = ConstantElasticityOfVariance::build(1.0, 0.4, 0.5, n_paths, n_steps, 1.0, 100.0).unwrap();
/// let paths: Vec<Array1<f64>> = cev.get_paths().unwrap();
///
/// assert_eq!(paths.len(), n_paths);
/// assert_eq!(paths[0].len(), n_steps+1);
/// let mut final_steps: Vec<f64> = Vec::<f64>::new();
/// for i in 0..n_paths {
///     final_steps.push(paths[i][n_steps]);
/// }
/// let final_steps: Array1<f64> = Array1::from_vec(final_steps);
/// let expected_path: Array1<f64> = cev.get_expectations();
/// assert!((final_steps.mean().unwrap() - expected_path.last().unwrap()).abs() < 200_000_000.0 * TEST_ACCURACY);
/// ```
pub struct ConstantElasticityOfVariance {
    /// Mean of the process
    mu: f64,
    /// Standard deviation of the process
    sigma: f64,
    /// Parameter controlling the relationship between volatility and price
    gamma: f64,
    /// Number of paths to generate
    n_paths: usize,
    /// Number of steps
    n_steps: usize,
    /// Final time step
    t_f: f64,
    /// Initial value of the stochastic process
    s_0: f64,
    /// Time difference between two consequtive time steps
    dt: f64,
    /// Array of time steps
    t: Array1<f64>,
}

impl ConstantElasticityOfVariance {
    /// Creates a new `ConstantElasticityOfVariance` instance.
    /// 
    /// # Input
    /// - `mu`: Mean of the process
    /// - `sigma`: Standard deviation of the process
    /// - `gamma`: Parameter controlling the relationship between volatility and price
    /// - `n_paths`: Number of paths to generate
    /// - `n_steps`: Number of steps
    /// - `t_f`: Final time step
    /// - `s_0`: Initial value of the stochastic process
    pub fn build(mu: f64, sigma: f64, gamma: f64, n_paths: usize, n_steps: usize, t_f: f64, s_0: f64) -> Result<Self, DigiFiError> {
        if gamma < 0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: "Constant Elasticity of Variance".to_owned(),
                constraint: "Tha value of argument `gamma` cannot be smaller than `0`.".to_owned(),
            });
        }
        let dt: f64 = t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, t_f + dt, dt);
        Ok(ConstantElasticityOfVariance { mu, sigma, gamma, n_paths, n_steps, t_f, s_0, dt, t })
    }

    /// Calculates the expected path of the CEV process.
    ///
    /// # Output
    /// - An array of expected values of the stock price at each time step
    ///
    /// # LaTeX Formula
    /// - E[S_{t}] = S_{0} e^{\\mu t}
    pub fn get_expectations(&self) -> Array1<f64> {
        self.s_0 * (self.mu * &self.t).map(|v| { v.exp() } )
    }
}

impl StochasticProcess for ConstantElasticityOfVariance {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths;
    }

    fn get_n_steps(&self) -> usize {
        self.n_steps
    }

    fn get_t_f(&self) -> f64 {
        self.t_f
    }

    /// Generates simulation paths for the Constant Elasticity of Variance (CEV) process.
    ///
    /// # Output
    /// - An array of simulated paths following the CEV process
    ///
    /// # LaTeX Formula
    /// - dS_{t} = \\mu*S_{t}*dt + \\sigma*S^{beta+1}_{t}*dW_{t}
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, DigiFiError> {
        let mut paths: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
        for _ in 0..self.n_paths {
            let dw: Array1<f64> = self.dt.sqrt() * StandardNormalInverseTransform::new_shuffle(self.t.len())?.generate()?;
            let mut s: Array1<f64> = Array1::from_vec(vec![self.s_0; self.t.len()]);
            for i in 1..self.t.len() {
                s[i] = s[i-1] + self.mu * s[i-1] * self.dt + self.sigma * s[i-1].powf(self.gamma) * dw[i];
            }
            paths.push(s);
        }
        Ok(paths)
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Model describes the evolution of stock price and its volatility.
/// 
/// # LaTeX Formula
/// - dS_{t} = \\mu*S_{t}*dt + \\sqrt{v_{t}}*S_{t}*dW^{S}_{t}
/// - dv_{t} = k*(\\theta-v)*dt + \\epsilon*\\sqrt{v}dW^{v}_{t}
/// - Corr(W^{S}_{t}, W^{v}_{t}) = \\rho
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Heston_model>
/// - Original Source: <https://doi.org/10.1093%2Frfs%2F6.2.327>
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::stochastic_processes::{StochasticProcess, HestonStochasticVolatility};
///
/// let n_paths: usize = 100;
/// let n_steps: usize = 200;
///
/// let hsv: HestonStochasticVolatility = HestonStochasticVolatility::new(0.1, 5.0, 0.07, 0.2, 0.0, n_paths, n_steps, 1.0, 100.0, 0.03);
/// let paths: Vec<Array1<f64>> = hsv.get_paths().unwrap();
/// 
/// assert_eq!(paths.len(), n_paths);
/// assert_eq!(paths[0].len(), n_steps+1);
/// let mut final_steps: Vec<f64> = Vec::<f64>::new();
/// for i in 0..n_paths {
///     final_steps.push(paths[i][n_steps]);
/// }
/// let final_steps: Array1<f64> = Array1::from_vec(final_steps);
/// let expected_path: Array1<f64> = hsv.get_expectations();
/// assert!((final_steps.mean().unwrap() - expected_path.last().unwrap()).abs() < 1_000_000.0 * TEST_ACCURACY);
/// ```
pub struct HestonStochasticVolatility {
    /// Mean of the process
    mu: f64,
    /// Scaling of volatility drift
    k: f64,
    /// Volatility trend
    theta: f64,
    /// Standard deviation of volatility
    epsilon: f64,
    /// Correlation between stock price and volatility
    rho: f64,
    /// Number of paths to generate
    n_paths: usize,
    /// Number of steps
    n_steps: usize,
    /// Final time step
    t_f: f64,
    /// Initial value of the stochastic process
    s_0: f64,
    /// Initial value of the volatility process
    v_0: f64,
    /// Time difference between two consequtive time steps
    dt: f64,
    /// Array of time steps
    t: Array1<f64>,
}

impl HestonStochasticVolatility {
    /// Creates a new `HestonStochasticVolatility` instance.
    /// 
    /// # Input
    /// - `mu`: Mean of the process
    /// - `k`: Scaling of volatility drift
    /// - `theta`: Volatility trend
    /// - `epsilon`: Standard deviation of volatility
    /// - `rho`: Correlation between stock price and volatility
    /// - `n_paths`: Number of paths to generate
    /// - `n_steps`: Number of steps
    /// - `t_f`: Final time step
    /// - `s_0`: Initial value of the stochastic process
    /// - `v_0`: Initial value of the volatility process
    pub fn new(mu: f64, k: f64, theta: f64, epsilon: f64, rho: f64, n_paths: usize, n_steps: usize, t_f: f64, s_0: f64, v_0: f64) -> Self {
        let dt: f64 = t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, t_f + dt, dt);
        HestonStochasticVolatility { mu, k, theta, epsilon, rho, n_paths, n_steps, t_f, s_0, v_0, dt, t }
    }

    /// Calculates the expected path of the Heston Stochastic Volatility process.
    ///
    /// # Output
    /// - An array of expected values of the stock price at each time step
    ///
    /// # LaTeX Formula
    /// - E[S_{t}] = S_{0} + (\\mu - \\frac{1}{2}\\theta) t + \\frac{\\theta - v_{0}}{2k} (1 - e^{-kt})
    pub fn get_expectations(&self) -> Array1<f64> {
        self.s_0 + (self.mu - 0.5 * self.theta)*&self.t + (self.theta - self.v_0)*(1.0 - (-self.k * &self.t).map(|v| { v.exp() } ))/(2.0 * self.k)
    }
}

impl StochasticProcess for HestonStochasticVolatility {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths;
    }

    fn get_n_steps(&self) -> usize {
        self.n_steps
    }

    fn get_t_f(&self) -> f64 {
        self.t_f
    }

    /// Generates simulation paths for the Heston Stochastic Volatility process.
    ///
    /// # Output
    /// - A tuple of arrays representing the simulated paths of stock prices and volatilities
    ///
    /// # LaTeX Formula
    /// - dS_{t} = \\mu*S_{t}*dt + \\sqrt{v_{t}}*S_{t}*dW^{S}_{t}
    /// - dv_{t} = k*(\\theta-v)*dt + \\epsilon*\\sqrt{v}dW^{v}_{t}
    /// - Corr(W^{S}_{t}, W^{v}_{t}) = \\rho
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, DigiFiError> {
        let mut paths: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
        let a: f64 = self.epsilon.powi(2) / self.k * ((-self.k * self.dt).exp() - (-2.0 * self.k * self.dt).exp());
        let b: f64 = self.theta * self.epsilon.powi(2) / (2.0 * self.k) * (1.0 - (-self.k * self.dt).exp()).powi(2);
        for _ in 0..self.n_paths {
            let nv: Array1<f64> = StandardNormalInverseTransform::new_shuffle(self.t.len())?.generate()?;
            let n: Array1<f64> = StandardNormalInverseTransform::new_shuffle(self.t.len())?.generate()?;
            let ns: Array1<f64> = self.rho * &nv + (1.0 - self.rho.powi(2)).sqrt() * &n;
            let mut v: Array1<f64> = Array1::from_vec(vec![self.v_0; self.t.len()]);
            let mut s: Array1<f64> = Array1::from_vec(vec![self.s_0; self.t.len()]);
            for i in 1..self.t.len() {
                v[i] = self.theta + (v[i-1] - self.theta) * (-self.k * self.dt).exp() + (a * v[i-1] + b).sqrt() * nv[i-1];
                v[i] = v[i].max(0.0);
                s[i] = s[i-1] + (self.mu - 0.5 * v[i-1])* self.dt + self.epsilon * (v[i-1] * self.dt).sqrt() * ns[i-1];
                s[i] = s[i].max(0.0)
            }
            paths.push(s);
        }
        Ok(paths)
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Model used in option pricing.
/// 
/// # LaTeX Formula
/// - dS_{t} = \\mu*dG(t) + \\sigma*\\sqrt{dG(t)}\\mathcal{N}(0,1)
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Variance_gamma_process>
/// - Original Source: <https://doi.org/10.1086%2F296519>
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::stochastic_processes::{StochasticProcess, VarianceGammaProcess};
///
/// let n_paths: usize = 2;
/// let n_steps: usize = 200;
///
/// let vg: VarianceGammaProcess = VarianceGammaProcess::new(0.2, 0.3, 20.0, n_paths, n_steps, 1.0, 0.03);
/// let paths: Vec<Array1<f64>> = vg.get_paths().unwrap();
///
/// assert_eq!(paths.len(), n_paths);
/// assert_eq!(paths[0].len(), n_steps+1);
/// ```
pub struct VarianceGammaProcess {
    /// Mean of the process
    mu: f64,
    /// Standard deviation of the process
    sigma: f64,
    /// Rate parameter of the Gamma distribution
    kappa: f64,
    /// Number of paths to generate
    n_paths: usize,
    /// Number of steps
    n_steps: usize,
    /// Final time step
    t_f: f64,
    /// Initial value of the stochastic process
    s_0: f64,
    /// Time difference between two consequtive time steps
    dt: f64,
    /// Array of time steps
    t: Array1<f64>,
}

impl VarianceGammaProcess {
    /// Creates a new `VarianceGammaProcess` instance.
    /// 
    /// # Input
    /// - `mu`: Mean of the process
    /// - `sigma`: Standard deviation of the process
    /// - `kappa`: Rate parameter of the Gamma distribution
    /// - `n_paths`: Number of paths to generate
    /// - `n_steps`: Number of steps
    /// - `t_f`: Final time step
    pub fn new(mu: f64, sigma: f64, kappa: f64, n_paths: usize, n_steps: usize, t_f: f64, s_0: f64) -> Self {
        let dt: f64 = t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, t_f + dt, dt);
        VarianceGammaProcess { mu, sigma, kappa, n_paths, n_steps, t_f, s_0, dt, t }
    }

    /// Calculates the expected path of the Variance Gamma process.
    ///
    /// # Output
    /// - An array of expected values of the stock price at each time step
    ///
    /// # LaTeX Formula
    /// - E[S_{t}] = S_{0} + \\mu t
    pub fn get_expectations(&self) -> Array1<f64> {
        self.s_0 + self.mu * &self.t
    }

    /// Calculates the variance of the Variance Gamma process.
    ///
    /// # Output
    /// - An array of variances of the stock price at each time step
    ///
    /// # LaTeX Formula
    /// - Var[S_{t}] = (\\sigma^{2} + \\frac{\\mu^{2}}{\\textit{rate}}) t
    pub fn get_variance(&self) -> Array1<f64> {
        (self.sigma.powi(2) + self.mu.powi(2) / self.kappa) * &self.t
    }
}

impl StochasticProcess for VarianceGammaProcess {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths;
    }

    fn get_n_steps(&self) -> usize {
        self.n_steps
    }

    fn get_t_f(&self) -> f64 {
        self.t_f
    }

    /// Generates simulation paths for the Variance Gamma process.
    ///
    /// # Output
    /// - An array of simulated paths following the Variance Gamma process
    ///
    /// # LaTeX Formula
    /// - dS_{t} = \\mu*dG(t) + \\sigma*\\sqrt{dG(t)}\\mathcal{N}(0,1)
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, DigiFiError> {
        let mut paths: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
        for _ in 0..self.n_paths {
            let gamma_dist: GammaDistribution = GammaDistribution::build(self.dt*self.kappa, 1.0/self.kappa)?;
            let dg: Array1<f64> = inverse_transform(&gamma_dist, self.t.len())?;
            let dg: Array1<f64> = dg.map(|v| { if v.is_nan() { 0.0 } else { *v } } );
            let n: Array1<f64> = StandardNormalInverseTransform::new_shuffle(self.t.len())?.generate()?;
            let mut s: Array1<f64> = Array1::from_vec(vec![self.s_0; self.t.len()]);
            for i in 1..self.t.len() {
                s[i] = s[i-1] + self.mu*dg[i-1] + dg[i-1].sqrt()*self.sigma*n[i-1];
            }
            paths.push(s);
        }
        Ok(paths)
    }
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::utilities::TEST_ACCURACY;
    use crate::stochastic_processes::StochasticProcess;

    #[test]
    fn unit_test_constant_elasticity_of_variance() -> () {
        use crate::stochastic_processes::stochastic_volatility_models::ConstantElasticityOfVariance;
        let n_paths: usize = 100;
        let n_steps: usize = 200;
        let cev: ConstantElasticityOfVariance = ConstantElasticityOfVariance::build(
            1.0, 0.4, 0.5, n_paths, n_steps, 1.0, 100.0
        ).unwrap();
        let paths: Vec<Array1<f64>> = cev.get_paths().unwrap();
        assert_eq!(paths.len(), n_paths);
        assert_eq!(paths[0].len(), n_steps+1);
        let mut final_steps: Vec<f64> = Vec::<f64>::new();
        for i in 0..n_paths {
            final_steps.push(paths[i][n_steps]);
        }
        let final_steps: Array1<f64> = Array1::from_vec(final_steps);
        let expected_path: Array1<f64> = cev.get_expectations();
        assert!((final_steps.mean().unwrap() - expected_path.last().unwrap()).abs() < 200_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_heston_stochastic_volatility() -> () {
        use crate::stochastic_processes::stochastic_volatility_models::HestonStochasticVolatility;
        let n_paths: usize = 100;
        let n_steps: usize = 200;
        let hsv: HestonStochasticVolatility = HestonStochasticVolatility::new(
            0.1, 5.0, 0.07, 0.2, 0.0, n_paths, n_steps, 1.0, 100.0, 0.03
        );
        let paths: Vec<Array1<f64>> = hsv.get_paths().unwrap();
        assert_eq!(paths.len(), n_paths);
        assert_eq!(paths[0].len(), n_steps+1);
        let mut final_steps: Vec<f64> = Vec::<f64>::new();
        for i in 0..n_paths {
            final_steps.push(paths[i][n_steps]);
        }
        let final_steps: Array1<f64> = Array1::from_vec(final_steps);
        let expected_path: Array1<f64> = hsv.get_expectations();
        assert!((final_steps.mean().unwrap() - expected_path.last().unwrap()).abs() < 1_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_variance_gamma() -> () {
        use crate::stochastic_processes::stochastic_volatility_models::VarianceGammaProcess;
        let n_paths: usize = 2;
        let n_steps: usize = 200;
        let vg: VarianceGammaProcess = VarianceGammaProcess::new(0.2, 0.3, 20.0, n_paths, n_steps, 1.0, 0.03);
        let paths: Vec<Array1<f64>> = vg.get_paths().unwrap();
        assert_eq!(paths.len(), n_paths);
        assert_eq!(paths[0].len(), n_steps+1);
    }
}