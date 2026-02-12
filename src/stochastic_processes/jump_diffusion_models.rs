use ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::random_generators::RandomGenerator;
use crate::stochastic_processes::StochasticProcess;
use crate::random_generators::{uniform_generators::FibonacciGenerator, standard_normal_generators::StandardNormalBoxMuller};
use crate::statistics::discrete_distributions::PoissonDistribution;
use crate::random_generators::generator_algorithms::inverse_transform;


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Model describes stock price with continuous movement that have rare large jumps.
/// 
/// # LaTeX Formula
/// - S_{t} = (\\mu-0.5*\\sigma^2)*t + \\sigma*W_{t} + sum_{i=1}^{N(t)} Z_{i}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Jump_diffusion#:~:text=a%20restricted%20volume-,In%20economics%20and%20finance,-%5Bedit%5D>
/// - Original Source: <https://doi.org/10.1016%2F0304-405X%2876%2990022-2>
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::stochastic_processes::{StochasticProcessResult, StochasticProcess, MertonJumpDiffusionProcess};
///
/// let n_paths: usize = 1_000;
/// let n_steps: usize = 200;
///
/// let mjd: MertonJumpDiffusionProcess = MertonJumpDiffusionProcess::new(0.03, 0.2, -0.03, 0.1, 1.5, n_paths, n_steps, 1.0, 100.0);
/// let sp_result: StochasticProcessResult = mjd.simulate().unwrap();
///
/// assert_eq!(sp_result.paths.len(), n_paths);
/// assert_eq!(sp_result.paths[0].len(), n_steps + 1);
/// assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
/// assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
/// assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 10_000_000.0 * TEST_ACCURACY);
/// ```
pub struct MertonJumpDiffusionProcess {
    /// Mean of the base stochastic process (i.e., process without jumps)
    mu_s: f64,
    /// Standard deviation of the base process (i.e., process without jumps)
    sigma_s: f64,
    /// Mean of the jumps
    mu_j: f64,
    /// Standard deviation of the jumps
    sigma_j: f64,
    /// Rate of jumps
    lambda_j: f64,
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

impl MertonJumpDiffusionProcess {
    /// Creates a new `MertonJumpDiffusionProcess` instance.
    /// 
    /// # Input
    /// - `mu_s`: Mean of the base stochastic process (i.e., process without jumps)
    /// - `sigma_s`: Standard deviation of the base process (i.e., process without jumps)
    /// - `mu_j`: Mean of the jumps
    /// - `sigma_j`: Standard deviation of the jumps
    /// - `lambda_j`: Rate of jumps
    /// - `n_paths`: Number of paths to generate
    /// - `n_steps`: Number of steps
    /// - `t_f`: Final time step
    /// - `s_0`: Initial value of the stochastic process
    pub fn new(mu_s: f64, sigma_s: f64, mu_j: f64, sigma_j: f64, lambda_j: f64, n_paths: usize, n_steps: usize, t_f: f64, s_0: f64) -> Self {
        let dt: f64 = t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, t_f + dt, dt);
        Self { mu_s, sigma_s, mu_j, sigma_j, lambda_j, n_paths, n_steps, t_f, s_0, dt, t }
    }
}

impl StochasticProcess for MertonJumpDiffusionProcess {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths;
    }

    fn get_n_steps(&self) -> usize {
        self.n_steps
    }

    fn get_t_f(&self) -> f64 {
        self.t_f
    }

    /// Calculates the expected path of the Merton Jump-Diffusion process.
    /// 
    /// # Output
    /// - An array of expected values of the stock price at each time step
    /// 
    /// # LaTeX Formula
    /// - E\[S_t\] = S_{0} + t(\\mu_{s} + \\lambda_{j}\\mu_{j})
    fn get_expectations(&self) -> Option<Array1<f64>> {
        Some(self.s_0 + &self.t*(self.mu_s + self.lambda_j * self.mu_j))
    }

    /// Calculates the variance of the Merton Jump-Diffusion process at each time step.
    /// 
    /// # Output
    /// - An array of variances of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - Var[S_{t}] = t(\\mu^{2}_{s} + \\lambda_{j}(\\mu^{2}_{j} + \\sigma^{2}_{j}))
    fn get_variances(&self) -> Option<Array1<f64>> {
        Some(&self.t * (self.mu_s.powi(2) + self.lambda_j*(self.mu_j.powi(2) + self.sigma_j.powi(2))))
    }

    /// Generates simulation paths for the Merton Jump-Diffusion process.
    /// 
    /// # Output
    /// - An array of simulated paths following the Merton Jump-Diffusion process
    /// 
    /// # LaTeX Formula
    /// - S_{t} = (\\mu-0.5*\\sigma^2)*t + \\sigma*W_{t} + sum_{i=1}^{N(t)} Z_{i}
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, DigiFiError> {
        let mut paths: Vec<Array1<f64>> = Vec::with_capacity(self.n_paths);
        let len: usize = self.t.len();
        let drift_coef: f64 = (self.mu_s - 0.5 * self.sigma_s.powi(2)) * self.dt;
        let diffusion_coef:  f64 = self.sigma_s * self.dt.sqrt();
        let poisson_dist: PoissonDistribution = PoissonDistribution::build(self.lambda_j * self.dt)?;
        for _ in 0..self.n_paths {
            let n_j: Array1<f64> = StandardNormalBoxMuller::new_shuffle(len)?.generate()?;
            let dp: Array1<f64> = inverse_transform(&poisson_dist, len)?;
            let n: Array1<f64> = StandardNormalBoxMuller::new_shuffle(len)?.generate()?;
            let (s, _) = (0..len).into_iter().fold((Vec::with_capacity(len), self.s_0), |(mut s, prev), i| {
                s.push(prev);
                // Jump process
                let dj: f64 = self.mu_j * dp[i] + self.sigma_j * dp[i].sqrt() * n_j[i];
                // Stochastic process
                (s, prev + (drift_coef + diffusion_coef * n[i]) + dj)
            } );
            paths.push(Array1::from_vec(s));
        }
        Ok(paths)
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Model describes stock price with continuous movement that have rare large jumps, with the jump sizes following a double exponential distribution.
/// 
/// # LaTeX Formula
/// - dS_{t} = \\mu*dt + \\sigma*dW_{t} + d(sum_{i=1}^{N(t)}(V_{i}-1))\n
/// where V_{i} is i.i.d. non-negative random variables such that Y = log(V) is the assymetric double exponential distribution with density:\n
/// - f_{Y}(y) = p*\\eta_{1}*e^{-\\eta_{1}y}\mathbb{1}_{0\\leq y} + (1-p)*\\eta_{2}*e^{\\eta_{2}y}\mathbb{1}_{y<0}
/// 
/// # Links
/// - Wikipedia: N/A
/// - Original Source: <https://dx.doi.org/10.2139/ssrn.242367>
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::stochastic_processes::{StochasticProcessResult, StochasticProcess, KouJumpDiffusionProcess};
///
/// let n_paths: usize = 1_000;
/// let n_steps: usize = 200;
///
/// let kjd: KouJumpDiffusionProcess = KouJumpDiffusionProcess::build(0.2, 0.3, 0.5, 9.0, 5.0, 0.5, n_paths, n_steps, 1.0, 100.0).unwrap();
/// let sp_result: StochasticProcessResult = kjd.simulate().unwrap();
///
/// assert_eq!(sp_result.paths.len(), n_paths);
/// assert_eq!(sp_result.paths[0].len(), n_steps + 1);
/// assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
/// assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
/// assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 20_000_000.0 * TEST_ACCURACY);
/// ```
pub struct KouJumpDiffusionProcess {
    /// Mean of the base stochastic process (i.e., process without jumps)
    mu: f64,
    /// Standard deviation of the base process (i.e., process without jumps)
    sigma: f64,
    /// Rate of jumps
    lambda_n: f64,
    /// Rate parameter of the positive jumps
    eta_1: f64,
    /// Rate parameter of the negative jumps
    eta_2: f64,
    /// Probability of a jump up
    p: f64,
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

impl KouJumpDiffusionProcess {
    /// Creates a new `KouJumpDiffusionProcess` instance.
    /// 
    /// # Input
    /// - `mu`: Mean of base stochastic process (i.e., process without jumps)
    /// - `sigma`: Standard deviation of base process (i.e., process without jumps)
    /// - `lambda_n`: Rate of jumps
    /// - `eta_1`: Rate parameter of the positive jumps
    /// - `eta_2`: Rate parameter of the negative jumps
    /// - `p`: Probability of a jump up
    /// - `n_paths`: Number of paths to generate
    /// - `n_steps`: Number of steps
    /// - `t_f`: Final time step
    /// - `s_0`: Initial value of the stochastic process
    ///
    /// # Errors
    /// - Returns an error if the argument `p` is not in the range \[0,1\].
    pub fn build(mu: f64, sigma: f64, lambda_n: f64, eta_1: f64, eta_2: f64, p: f64, n_paths: usize, n_steps: usize, t_f: f64, s_0: f64) -> Result<Self, DigiFiError> {
        if (p < 0.0) || (1.0 < p) {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `p` must be in the range `[0,1]`.".to_owned(),
            });
        }
        let dt: f64 = t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, t_f + dt, dt);
        Ok(Self { mu, sigma, lambda_n, eta_1, eta_2, p, n_paths, n_steps, t_f, s_0, dt, t })
    }
}

impl ErrorTitle for KouJumpDiffusionProcess {
    fn error_title() -> String {
        String::from("Kou Jump-Diffusion Process")
    }
}

impl StochasticProcess for KouJumpDiffusionProcess {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths;
    }

    fn get_n_steps(&self) -> usize {
        self.n_steps
    }

    fn get_t_f(&self) -> f64 {
        self.t_f
    }

    /// Calculates the expected path of the Kou Jump-Diffusion process
    /// 
    /// # Output
    /// - An array of expected values of the stock price at each time step
    /// 
    /// # LaTeX Formula
    /// - E\[S_t\] = S_{0} + t(\\mu + \\lambda_{n}(\\frac{p}{\\eta_{1}} - \\frac{1-p}{\\eta_{2}}))
    fn get_expectations(&self) -> Option<Array1<f64>> {
        Some(self.s_0 + &self.t * (self.mu + self.lambda_n * (self.p / self.eta_1 - (1.0 - self.p) / self.eta_2)))
    }

    /// Calculates the variance of the Kou Jump-Diffusion process.
    /// 
    /// # Output
    /// - An array of variances of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - Var[S_{t}] = t(\\sigma^{2} + 2\\lambda_{n}(\\frac{p}{\\eta^{2}_{1}} + \\frac{1-p}{\\eta^{2}_{2}}))
    fn get_variances(&self) -> Option<Array1<f64>> {
        Some(&self.t * (self.sigma.powi(2) + 2.0 * self.lambda_n * (self.p / self.eta_1.powi(2) + (1.0 - self.p) / self.eta_2.powi(2))))
    }

    /// Generates simulation paths for the Kou Jump-Diffusion process.
    /// 
    /// # Output
    /// - Array of simulated paths following the Kou Jump-Diffusion process
    ///
    /// # LaTeX Formula
    /// - dS_{t} = \\mu*dt + \\sigma*dW_{t} + d(sum_{i=1}^{N(t)}(V_{i}-1))\n
    /// where V_{i} is i.i.d. non-negative random variables such that Y = log(V) is the assymetric double exponential distribution with density:\n
    /// - f_{Y}(y) = p*\\eta_{1}*e^{-\\eta_{1}y}\mathbb{1}_{0\\leq y} + (1-p)*\\eta_{2}*e^{\\eta_{2}y}\mathbb{1}_{y<0}
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, DigiFiError> {
        let mut paths: Vec<Array1<f64>> = Vec::with_capacity(self.n_paths);
        let len: usize = self.t.len();
        let neg_one_over_eta_1: f64 = -1.0 / self.eta_1;
        let one_over_eta_2: f64 = 1.0 / self.eta_2;
        let drift_coef: f64 = (self.mu - 0.5 * self.sigma.powi(2)) * self.dt;
        let diffusion_coef:  f64 = self.sigma * self.dt.sqrt();
        let poisson_dist: PoissonDistribution = PoissonDistribution::build(self.lambda_n * self.dt)?;
        for _ in 0..self.n_paths {
            let dp: Array1<f64> = inverse_transform(&poisson_dist, len)?;
            let u: Array1<f64> = FibonacciGenerator::new_shuffle(len)?.generate()?;
            let n: Array1<f64> = StandardNormalBoxMuller::new_shuffle(len)?.generate()?;
            let (s, _) = (0..len).into_iter().fold((Vec::with_capacity(len), self.s_0), |(mut s, prev), i| {
                s.push(prev);
                // Assymetric double exponential random variable
                let dj: f64 = if self.p <= u[i] {
                    ((neg_one_over_eta_1 * ((1.0 - u[i]) / self.p).ln()).exp() - 1.0) * dp[i]
                } else if u[i] < self.p {
                    ((one_over_eta_2 * (u[i] / (1.0 - self.p)).ln()).exp() - 1.0) * dp[i]
                } else {
                    0.0
                };
                // Stochastic process
                (s, prev + (drift_coef + diffusion_coef * n[i]) + dj)
            } );
            paths.push(Array1::from_vec(s));
        }
        Ok(paths)
    }
}


#[cfg(test)]
mod tests {
    use crate::utilities::TEST_ACCURACY;
    use crate::stochastic_processes::{StochasticProcessResult, StochasticProcess};

    #[test]
    fn unit_test_merton_jump_diffusion() -> () {
        use crate::stochastic_processes::jump_diffusion_models::MertonJumpDiffusionProcess;
        let n_paths: usize = 1_000;
        let n_steps: usize = 200;
        let mjd: MertonJumpDiffusionProcess = MertonJumpDiffusionProcess::new(
            0.03, 0.2, -0.03, 0.1, 1.5, n_paths, n_steps, 1.0, 100.0
        );
        let sp_result: StochasticProcessResult = mjd.simulate().unwrap();
        assert_eq!(sp_result.paths.len(), n_paths);
        assert_eq!(sp_result.paths[0].len(), n_steps + 1);
        assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
        assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
        assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 10_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_kou_jump_diffusion() -> () {
        use crate::stochastic_processes::jump_diffusion_models::KouJumpDiffusionProcess;
        let n_paths: usize = 1_000;
        let n_steps: usize = 200;
        let kjd: KouJumpDiffusionProcess = KouJumpDiffusionProcess::build(
            0.2, 0.3, 0.5, 9.0, 5.0, 0.5, n_paths, n_steps, 1.0, 100.0
        ).unwrap();
        let sp_result: StochasticProcessResult = kjd.simulate().unwrap();
        assert_eq!(sp_result.paths.len(), n_paths);
        assert_eq!(sp_result.paths[0].len(), n_steps + 1);
        assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
        assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
        assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 20_000_000.0 * TEST_ACCURACY);
    }
}