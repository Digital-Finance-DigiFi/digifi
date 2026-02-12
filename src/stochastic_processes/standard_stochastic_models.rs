use ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::random_generators::RandomGenerator;
use crate::stochastic_processes::StochasticProcess;
use crate::random_generators::standard_normal_generators::StandardNormalBoxMuller;


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Different methods of simulating the Feller Square-Root Process.
pub enum FSRSimulationMethod {
    EulerMaruyama,
    AnalyticEulerMaruyama,
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Arithmetic Brownian motion.
/// 
/// # LaTeX Formula
/// - dS_{t} = \\mu*dt + \\sigma*dW_{t}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Geometric_Brownian_motion#:~:text=solution%20claimed%20above.-,Arithmetic%20Brownian%20Motion,-%5Bedit%5D>
/// - Original Source: <https://doi.org/10.24033/asens.476>
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::stochastic_processes::{StochasticProcessResult, StochasticProcess, ArithmeticBrownianMotion};
///
/// let n_paths: usize = 1_000;
/// let n_steps: usize = 200;
///
/// let abm: ArithmeticBrownianMotion = ArithmeticBrownianMotion::new(1.0, 0.4, n_paths, n_steps, 1.0, 100.0);
/// let sp_result: StochasticProcessResult = abm.simulate().unwrap();
/// 
/// assert_eq!(sp_result.paths.len(), n_paths);
/// assert_eq!(sp_result.paths[0].len(), n_steps + 1);
/// assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
/// assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
/// assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 10_000_000.0 * TEST_ACCURACY);
/// ```
pub struct ArithmeticBrownianMotion {
    /// Drift constant of the process
    mu: f64,
    /// Diffusion constant of the process
    sigma: f64,
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

impl ArithmeticBrownianMotion {
    /// Creates a new `ArithmeticBrownianMotion` instance.
    /// 
    /// # Input
    /// - `mu`: Drift constant of the process
    /// - `sigma`: Diffusion constant of the process
    /// - `n_paths`: Number of paths to generate
    /// - `n_steps`: Number of steps
    /// - `t_f`: Final time step
    /// - `s_0`: Initial value of the stochastic process
    pub fn new(mu: f64, sigma: f64, n_paths: usize, n_steps: usize, t_f: f64, s_0: f64) -> Self {
        let dt: f64 = t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, t_f + dt, dt);
        Self { mu, sigma, n_paths, n_steps, t_f, s_0, dt, t }
    }

    /// Calculates the auto-covariance of the Arithmetic Brownian Motion between two time points.
    /// 
    /// # Input
    /// - `index_t1`: Index of the first time point
    /// - `index_t2`: Index of the second time point
    /// 
    /// # Output
    /// - Auto-covariance of the process between times t1 and t2
    /// 
    /// # Errors
    /// - Returns an error if the index provided is out of bounds for the time array.
    /// 
    /// # LaTeX Formula
    /// - \\textit{Cov}(S_{t_{1}}, S_{t_{2}}) = \\sigma^{2} \\min(S_{t_{1}}, S_{t_{2}})
    pub fn get_auto_cov(&self, index_t1: usize, index_t2: usize) -> Result<f64, DigiFiError> {
        let t_len: usize = self.t.len();
        if t_len < index_t1 {
            return Err(DigiFiError::IndexOutOfRange { title: Self::error_title(), index: "index_t1".to_owned(), array: "time".to_owned(), });
        }
        if t_len < index_t2 {
            return Err(DigiFiError::IndexOutOfRange { title: Self::error_title(), index: "index_t2".to_owned(), array: "time".to_owned(), });
        }
        Ok(self.sigma.powi(2) * self.t[index_t1].min(self.t[index_t2]))
    }
}

impl ErrorTitle for ArithmeticBrownianMotion {
    fn error_title() -> String {
        String::from("Arithmetic Brownian Motion")
    }
}

impl StochasticProcess for ArithmeticBrownianMotion {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths;
    }

    fn get_n_steps(&self) -> usize {
        self.n_steps
    }

    fn get_t_f(&self) -> f64 {
        self.t_f
    }

    /// Calculates the expected path of the Arithmetic Brownian Motion.
    /// 
    /// # Output
    /// - An array of expected values of the stock price at each time step
    /// 
    /// # LaTeX Formula
    /// - E\[S_t\] = \\mu t + S_{0}
    fn get_expectations(&self) -> Option<Array1<f64>> {
        Some(self.mu * &self.t + self.s_0)
    }

    /// Calculates the variance of the Arithmetic Brownian Motion at each time step.
    /// 
    /// # Output
    /// - An array of variances of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - Var[S_{t}] = \\sigma^{2} t
    fn get_variances(&self) -> Option<Array1<f64>> {
        Some(&self.t * self.sigma.powi(2))
    }

    /// Generates simulation paths for the Arithmetic Brownian Motion using the Euler-Maruyama method.
    /// 
    /// # Output
    /// - An array of simulated paths following the Arithmetic Brownian Motion
    /// 
    /// # LaTeX Formula
    /// - dS_{t} = \\mu dt + \\sigma dW_{t}
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, DigiFiError> {
        let mut paths: Vec<Array1<f64>> = Vec::with_capacity(self.n_paths);
        let len: usize = self.t.len();
        let drift_coef: f64 = self.mu * self.dt;
        let diffusion_coef: f64 = self.sigma * self.dt.sqrt();
        for _ in 0..self.n_paths {
            let n: Array1<f64> = StandardNormalBoxMuller::new_shuffle(len)?.generate()?;
            let (ds, _) = (0..len).into_iter().fold((Vec::with_capacity(len), self.s_0), |(mut ds, prev), i| {
                ds.push(prev);
                (ds, prev + drift_coef + diffusion_coef * n[i])
            } );
            paths.push(Array1::from_vec(ds));
        }
        Ok(paths)
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Model describing the evolution of stock prices.
/// 
/// # LaTeX Formula
/// - dS_{t} = \\mu*S_{t}*dt + \\sigma*S_{t}*dW_{t}\n
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Geometric_Brownian_motion>
/// - Original Source: <http://dx.doi.org/10.1086/260062>
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::stochastic_processes::{StochasticProcessResult, StochasticProcess, GeometricBrownianMotion};
///
/// let n_paths: usize = 1_000;
/// let n_steps: usize = 200;
/// let gbm: GeometricBrownianMotion = GeometricBrownianMotion::new(0.0, 0.2, n_paths, n_steps, 1.0, 100.0);
/// let sp_result: StochasticProcessResult = gbm.simulate().unwrap();
/// 
/// assert_eq!(sp_result.paths.len(), n_paths);
/// assert_eq!(sp_result.paths[0].len(), n_steps + 1);
/// assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
/// assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
/// assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 100_000_000.0 * TEST_ACCURACY);
/// ```
pub struct GeometricBrownianMotion {
    /// Drift constant of the process
    mu: f64,
    /// Diffusion constant of the process
    sigma: f64,
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

impl GeometricBrownianMotion {
    /// Creates a new `GoemetricBrownianMotion` instance.
    /// 
    /// # Input
    /// - `mu`: Drift constant of the process
    /// - `sigma`: Diffusion constant of the process
    /// - `n_paths`: Number of paths to generate
    /// - `n_steps`: Number of steps
    /// - `t_f`: Final time step
    /// - `s_0`: Initial value of the stochastic process
    pub fn new(mu: f64, sigma: f64, n_paths: usize, n_steps: usize, t_f: f64, s_0: f64) -> Self {
        let dt: f64 = t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, t_f + dt, dt);
        Self { mu, sigma, n_paths, n_steps, t_f, s_0, dt, t }
    }
}

impl StochasticProcess for GeometricBrownianMotion {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths;
    }

    fn get_n_steps(&self) -> usize {
        self.n_steps
    }

    fn get_t_f(&self) -> f64 {
        self.t_f
    }

    /// Calculates the expected path of the Geometric Brownian Motion. This represents the mean trajectory of the stock price over time.
    /// 
    /// # Output
    /// - An array of expected values of the stock price at each time step, representing the mean trajectory
    /// 
    /// # LaTeX Formula:
    /// - E\[S_t\] = S_{0} e^{\\mu t}
    fn get_expectations(&self) -> Option<Array1<f64>> {
        Some(self.t.map(|t| { self.s_0 * (self.mu * t).exp() } ))
    }

    /// Computes the variance of the stock price at each time step under the Geometric Brownian Motion model.
    /// This provides an indication of the variability or risk associated with the stock price.
    /// 
    /// 
    /// # Output
    /// - An array of variances of the stock price at each time step
    /// 
    /// # LaTeX Formula
    /// - \\textit{Var}[S_{t}] = (S^{2}_{0}) e^{2\\mu t} (e^{\\sigma^{2} t} - 1)
    fn get_variances(&self) -> Option<Array1<f64>> {
        let s_0_sq: f64 = self.s_0.powi(2);
        let two_mu: f64 = 2.0 * self.mu;
        let sigma_sq: f64 = self.sigma.powi(2);
        Some(self.t.map(|t| { s_0_sq * (two_mu * t).exp() * ((t * sigma_sq).exp() - 1.0) } ))
    }

    /// Simulates paths of the Geometric Brownian Motion using the Euler-Maruyama method.
    /// This method provides an approximation of the continuous-time process.
    /// 
    /// # Output
    /// - An array of simulated stock prices following the Geometric Brownian Motion for each path and time step
    /// 
    /// # LaTeX Formula
    /// - dS_{t} = \\mu S_{t} dt + \\sigma S_{t} dW_{t}
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, DigiFiError> {
        let mut paths: Vec<Array1<f64>> = Vec::with_capacity(self.n_paths);
        let len: usize = self.t.len();
        let drift_coef: f64 = self.mu * self.dt;
        let diffusion_coef: f64 = self.sigma * self.dt.sqrt();
        for _ in 0..self.n_paths {
            let n: Array1<f64> = StandardNormalBoxMuller::new_shuffle(len)?.generate()?;
            let (s, _) = (0..len).into_iter().fold((Vec::with_capacity(len), self.s_0), |(mut s, prev), i| {
                s.push(prev);
                (s, prev + drift_coef * prev + diffusion_coef * prev * n[i])
            } );
            paths.push(Array1::from_vec(s));
        }
        Ok(paths)
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Model describes the evolution of interest rates.
/// 
/// # LaTeX Formula
/// - dS_{t} = \\alpha*(\\mu-S_{t})*dt + \\sigma*dW_{t}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process>
/// - Original Source: <https://doi.org/10.1103%2FPhysRev.36.823>
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::stochastic_processes::{StochasticProcessResult, StochasticProcess, OrnsteinUhlenbeckProcess};
///
/// let n_paths: usize = 1_000;
/// let n_steps: usize = 200;
///
/// let oup: OrnsteinUhlenbeckProcess = OrnsteinUhlenbeckProcess::new(0.07, 0.1, 10.0, n_paths, n_steps, 1.0, 0.05, true);
/// let sp_result: StochasticProcessResult = oup.simulate().unwrap();
/// 
/// assert_eq!(sp_result.paths.len(), n_paths);
/// assert_eq!(sp_result.paths[0].len(), n_steps + 1);
/// assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
/// assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
/// assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 1_000_000.0 * TEST_ACCURACY);
/// ```
pub struct OrnsteinUhlenbeckProcess {
    /// Drift constant of the process
    mu: f64,
    /// Diffusion constant of the process
    sigma: f64,
    /// Drift scaling parameter
    alpha: f64,
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
    /// If True, uses the analytic moments for Euler-Maruyama; otherwise, uses plain Euler-Maruyama simulation
    analytic_em: bool,
}

impl OrnsteinUhlenbeckProcess {
    /// Creates a new `OrnsteinUhlenbeckProcess` instance.
    /// 
    /// # Input
    /// - `mu`: Drift constant of the process
    /// - `sigma`: Diffusion constant of the process
    /// - `alpha`: Drift scaling parameter
    /// - `n_paths`: Number of paths to generate
    /// - `n_steps`: Number of steps
    /// - `t_f`: Final time step
    /// - `s_0`: Initial value of the stochastic process
    /// - `analytic_ecm`: If true, uses the analytic moments for Euler-Maruyama; otherwise, uses plain Euler-Maruyama simulation
    pub fn new(mu: f64, sigma: f64, alpha: f64, n_paths: usize, n_steps: usize, t_f: f64, s_0: f64, analytic_em: bool) -> Self {
        let dt: f64 = t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, t_f + dt, dt);
        Self { mu, sigma, alpha, n_paths, n_steps, t_f, s_0, dt, t, analytic_em }
    }
}

impl StochasticProcess for OrnsteinUhlenbeckProcess {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths
    }

    fn get_n_steps(&self) -> usize {
        self.n_steps
    }

    fn get_t_f(&self) -> f64 {
        self.t_f
    }

    /// Calculates the expected path of the Ornstein-Uhlenbeck Process, showing the mean-reverting nature of the process over time.
    /// 
    /// # Output
    /// - An array (np.ndarray) of expected values of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - E\[S_t\] = \\mu + (S_{0} - \\mu) e^{-\\alpha t}
    fn get_expectations(&self) -> Option<Array1<f64>> {
        let drift_coef: f64 = self.s_0 - self.mu;
        Some(self.t.map(|t| { self.mu + drift_coef * (-self.alpha * t).exp() } ))
    }

    /// Computes the variance of the Ornstein-Uhlenbeck Process at each time step, providing insights into the variability around the mean.
    /// 
    /// # Output
    /// - An array of variances of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - \\textit{Var}[S_{t}] = \\frac{\\sigma^{2}}{2\\alpha} (1 - e^{-2\\alpha t})
    fn get_variances(&self) -> Option<Array1<f64>> {
        let two_alpha: f64 = 2.0 * self.alpha;
        let sigma_sq: f64 = self.sigma.powi(2);
        Some(self.t.map(|t| { (1.0 - (-two_alpha * t).exp()) * sigma_sq / two_alpha } ))
    }

    /// Simulates paths of the Ornstein-Uhlenbeck Process using the Euler-Maruyama method.
    /// This method can be adjusted to use either the standard numerical simulation or an analytic adjustment for Euler-Maruyama.
    /// 
    /// # Output
    /// - An array representing simulated paths of the process for each path and time step
    /// 
    /// # LaTeX Formula
    /// - dS_{t} = \\alpha(\\mu - S_{t}) dt + \\sigma dW_{t}
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, DigiFiError> {
        let mut paths: Vec<Array1<f64>> = Vec::with_capacity(self.n_paths);
        let len: usize = self.t.len();
        let alpha_dt: f64 = self.alpha * self.dt;
        let (drift_coef, diffusion_coef) = if self.analytic_em {
            let drift_coef: f64 = (-alpha_dt).exp();
            let diffusion_coef: f64 = self.sigma * ((1.0 - (-2.0 * alpha_dt).exp()) / (2.0 * self.alpha)).sqrt();
            (drift_coef, diffusion_coef)
        } else {
            let drift_coef: f64 = alpha_dt;
            let diffusion_coef: f64 = self.sigma * self.dt.sqrt();
            (drift_coef, diffusion_coef)
        };
        for _ in 0..self.n_paths {
            let mut s: Array1<f64> = Array1::from_vec(vec![self.s_0; len]);
            let r: Array1<f64> = StandardNormalBoxMuller::new_shuffle(len)?.generate()?;
            if self.analytic_em {
                // Analytic Euler-Maruyama method
                for i in 1..len {
                    s[i] = self.mu + drift_coef * (s[i-1] - self.mu) + diffusion_coef * r[i-1]
                }
            } else {
                // Plain Euler-Maruyama method
                for i in 1..len {
                    s[i] = s[i-1] + drift_coef * (self.mu - s[i]) + diffusion_coef * r[i];
                }
            }
            paths.push(s);
        }
        Ok(paths)
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Model can support useful variance reduction techniques for pricing derivative contracts using Monte-Carlo simulation, 
/// such as sampling. Also used in scenario generation.
/// 
/// # LaTeX Formula
/// - dS_{t} = ((b-a)/(T-t))*dt + \\sigma*dW_{t}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Brownian_bridge>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::stochastic_processes::{StochasticProcessResult, StochasticProcess, BrownianBridge};
///
/// let n_paths: usize = 1_000;
/// let n_steps: usize = 200;
/// let bb: BrownianBridge = BrownianBridge::new(1.0, 2.0, 0.5, n_paths, n_steps, 1.0);
/// let sp_result: StochasticProcessResult = bb.simulate().unwrap();
/// 
/// assert_eq!(sp_result.paths.len(), n_paths);
/// assert_eq!(sp_result.paths[0].len(), n_steps + 1);
/// assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
/// assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
/// assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 1_000_000.0 * TEST_ACCURACY);
/// ```
pub struct BrownianBridge {
    /// Initial value of the process
    alpha: f64,
    /// Final value of the process
    beta: f64,
    /// Standard deviation of the process
    sigma: f64,
    /// Number of paths to generate
    n_paths: usize,
    /// Number of steps
    n_steps: usize,
    /// Final time step
    t_f: f64,
    /// Initial value of the stochastic process
    dt: f64,
    /// Array of time steps
    t: Array1<f64>,
}

impl BrownianBridge {
    /// Creates a new `BrownianBridge` instance.
    /// 
    /// # Input
    /// - `alpha`: Initial value of the process
    /// - `beta`: Final value of the process
    /// - `sigma`: Standard deviation of the process
    /// - `n_paths`: Number of paths to generate
    /// - `n_steps`: Number of steps
    /// - `t_f`: Final time step
    pub fn new(alpha: f64, beta: f64, sigma: f64, n_paths: usize, n_steps: usize, t_f: f64) -> Self {
        let dt: f64 = t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, t_f + dt, dt);
        Self { alpha, beta, sigma, n_paths, n_steps, t_f, dt, t }
    }
}

impl StochasticProcess for BrownianBridge {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths
    }

    fn get_n_steps(&self) -> usize {
        self.n_steps
    }

    fn get_t_f(&self) -> f64 {
        self.t_f
    }

    /// Calculates the expected path of the Brownian Bridge. It represents the expected value of the process at each time step,
    /// starting at 'alpha' and trending towards 'beta'.
    /// 
    /// # Output
    /// - An array of expected values of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - E[S_{t}] = \\alpha + (\\beta - \\alpha) \\frac{t}{T}
    fn get_expectations(&self) -> Option<Array1<f64>> {
        let c: f64 = (self.beta - self.alpha) / self.t_f;
        Some(self.t.map(|t| self.alpha + c * t ))
    }

    /// Computes the variance of the Brownian Bridge at each time step.
    /// This illustrates how the variability of the process decreases as it approaches the endpoint 'beta' at time T.
    /// 
    /// # Output
    /// - An array of variances of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - \\text{Var}[S_{t}] = \\frac{t(T-t)}{T} \\sigma^{2}
    fn get_variances(&self) -> Option<Array1<f64>> {
        Some(&self.t * (self.t_f - &self.t) / self.t_f)
    }

    /// Generates simulation paths for the Brownian Bridge using the Euler-Maruyama method.
    /// 
    /// This method approximates the continuous-time process and ensures that the path starts at 'alpha' and ends at 'beta' at time t_f.
    /// 
    /// # Output
    /// - An array of simulated paths following the Brownian Bridge for each path and time step
    /// 
    /// # LaTeX Formula
    /// - dS_{t} = ((\\beta - \\alpha)/(T - t)) dt + \\sigma dW_{t}
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, DigiFiError> {
        let mut paths: Vec<Array1<f64>> = Vec::with_capacity(self.n_paths);
        let len: usize = self.t.len();
        let diffusion_coef: f64 = self.sigma * self.dt.sqrt();
        for _ in 0..self.n_paths {
            let n: Array1<f64> = StandardNormalBoxMuller::new_shuffle(len)?.generate()?;
            let mut s: Array1<f64> = Array1::from_vec(vec![self.alpha; len]);
            s[len - 1] = self.beta;
            for i in 0..(len - 2) {
                s[i+1] = s[i] + (self.beta - s[i]) / (len - i) as f64 + diffusion_coef * n[i];
            }
            paths.push(s)
        }
        Ok(paths)
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Model describes the evolution of interest rates.
/// 
/// # LaTeX Formula
/// - dS_{t} = \\alpha*(\\mu-S_{t})*dt + \\sigma\\sqrt(S_{t})*dW_{t}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model>
/// - Original Source: <https://doi.org/10.2307/1911242>
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::stochastic_processes::{StochasticProcessResult, StochasticProcess, FellerSquareRootProcess, FSRSimulationMethod};
///
/// let n_paths: usize = 1_000;
/// let n_steps: usize = 200;
///
/// let fsrp: FellerSquareRootProcess = FellerSquareRootProcess::new(0.05, 0.265, 5.0, n_paths, n_steps, 1.0, 0.03, FSRSimulationMethod::EulerMaruyama);
/// let sp_result: StochasticProcessResult = fsrp.simulate().unwrap();
/// 
/// assert_eq!(sp_result.paths.len(), n_paths);
/// assert_eq!(sp_result.paths[0].len(), n_steps + 1);
/// assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
/// assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
/// assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 1_000_000.0 * TEST_ACCURACY);
/// ```
pub struct FellerSquareRootProcess {
    /// Drift constant of the process
    mu: f64,
    /// Diffusion constant of the process
    sigma: f64,
    /// Drift scaling parameter
    alpha: f64,
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
    /// Method for computing Feller-Square root process
    method: FSRSimulationMethod,
}

impl FellerSquareRootProcess {
    /// Creates a new `FellerSquareRootProcess` instance.
    /// 
    /// # Input
    /// - `mu`: Mean of the process
    /// - `sigma`: Standard deviation of the process
    /// - `alpha`: Drift scaling parameter
    /// - `n_paths`: Number of paths to generate
    /// - `n_steps`: Number of steps
    /// - `T`: Final time step
    /// - `s_0`: Initial value of the stochastic process
    /// - `method`: Method for computing Feller-Square root process
    pub fn new(mu: f64, sigma: f64, alpha: f64, n_paths: usize, n_steps: usize, t_f: f64, s_0: f64, method: FSRSimulationMethod) -> Self {
        let dt: f64 = t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, t_f + dt, dt);
        Self { mu, sigma, alpha, n_paths, n_steps, t_f, s_0, dt, t, method }
    }
}

impl StochasticProcess for FellerSquareRootProcess {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths
    }

    fn get_n_steps(&self) -> usize {
        self.n_steps
    }

    fn get_t_f(&self) -> f64 {
        self.t_f
    }

    /// Calculates the expected path of the Feller Square-Root Process, showing the mean-reverting nature over time towards the long-term mean \\mu.
    /// 
    /// # Output
    /// - An array of expected values of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - E[S_{t}] = \\mu + (S_{0} - \\mu) e^{-\\alpha t}
    fn get_expectations(&self) -> Option<Array1<f64>> {
        let drift_delta: f64 = self.s_0 - self.mu;
        Some(self.t.map(|t| { self.mu + drift_delta * (-self.alpha * t).exp() } ))
    }

    /// Computes the variance of the Feller Square-Root Process at each time step, providing insights into the variability around the mean.
    /// 
    /// # Output
    /// - An array of variances of the process at each time step
    fn get_variances(&self) -> Option<Array1<f64>> {
        let neg_two_alpha: f64 = -2.0 * self.alpha;
        let mult_1: f64 = self.s_0 / self.alpha;
        let mult_2: f64 = self.mu / (2.0 * self.alpha);
        let sigma_sq: f64 = self.sigma.powi(2);
        Some(self.t.map(|t| {
            let v_1: f64 = ((-self.alpha * t).exp() - (neg_two_alpha * t).exp()) * mult_1;
            let v_2: f64 = (neg_two_alpha * t).exp() * ((self.alpha * t).exp() - 1.0).powi(2) * mult_2;
            sigma_sq * (v_1 + v_2)
        } ))
    }

    /// Simulates paths of the Feller Square-Root Process using different methods: Euler-Maruyama, Analytic Euler-Maruyama,
    /// or Exact method, depending on the specified method in the process setup.
    /// 
    /// # Output
    /// - An array of simulated paths of the process for each path and time step, following the chosen simulation method
    ///
    /// # LaTeX Formula
    /// - dS_{t} = \\alpha*(\\mu-S_{t})*dt + \\sigma\\sqrt(S_{t})*dW_{t}
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, DigiFiError> {
        let mut paths: Vec<Array1<f64>> = Vec::with_capacity(self.n_paths);
        let len: usize = self.t.len();
        let alpha_dt: f64 = self.alpha * self.dt;
        let (exp_neg_alpha_dt, a, b) = match self.method {
            FSRSimulationMethod::EulerMaruyama => (f64::NAN, f64::NAN, f64::NAN),
            FSRSimulationMethod::AnalyticEulerMaruyama => {
                let exp_neg_alpha_dt: f64 = (-alpha_dt).exp();
                let sigma_sq: f64 = self.sigma.powi(2);
                let a: f64 = sigma_sq / self.alpha * (exp_neg_alpha_dt - (-2.0 * alpha_dt).exp());
                let b: f64 = self.mu * sigma_sq / (2.0 * self.alpha) * (1.0 - exp_neg_alpha_dt).powi(2);
                (exp_neg_alpha_dt, a, b)
            }
        };
        for _ in 0..self.n_paths {
            let mut s: Array1<f64> = Array1::from_vec(vec![self.s_0; len]);
            let r: Array1<f64> = StandardNormalBoxMuller::new_shuffle(len)?.generate()?;
            match self.method {
                FSRSimulationMethod::EulerMaruyama => {
                    for i in 1..len {
                        s[i] = s[i-1] + alpha_dt * (self.mu - s[i-1]) + self.sigma * (s[i-1] * self.dt).sqrt() * r[i-1];
                        s[i] = s[i].max(0.0);
                    }
                },
                FSRSimulationMethod::AnalyticEulerMaruyama => {
                    for i in 1..len {
                        s[i] = self.mu + (s[i-1] - self.mu) * exp_neg_alpha_dt + (a * s[i-1] + b).sqrt() * r[i-1];
                        s[i] = s[i].max(0.0);
                    }
                },
            }
            paths.push(s);
        }
        Ok(paths)
    }
}


#[cfg(test)]
mod tests {
    use crate::stochastic_processes::{StochasticProcessResult, StochasticProcess};
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_arithmetic_brownian_motion() -> () {
        use crate::stochastic_processes::standard_stochastic_models::ArithmeticBrownianMotion;
        let n_paths: usize = 1_000;
        let n_steps: usize = 200;
        let abm: ArithmeticBrownianMotion = ArithmeticBrownianMotion::new(1.0, 0.4, n_paths, n_steps, 1.0, 100.0);
        let sp_result: StochasticProcessResult = abm.simulate().unwrap();
        assert_eq!(sp_result.paths.len(), n_paths);
        assert_eq!(sp_result.paths[0].len(), n_steps + 1);
        assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
        assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
        assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 10_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_geometric_brownian_motion() -> () {
        use crate::stochastic_processes::standard_stochastic_models::GeometricBrownianMotion;
        let n_paths: usize = 1_000;
        let n_steps: usize = 200;
        let gbm: GeometricBrownianMotion = GeometricBrownianMotion::new(0.0, 0.2, n_paths, n_steps, 1.0, 100.0);
        let sp_result: StochasticProcessResult = gbm.simulate().unwrap();
        assert_eq!(sp_result.paths.len(), n_paths);
        assert_eq!(sp_result.paths[0].len(), n_steps + 1);
        assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
        assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
        assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 100_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_ornstein_uhlenbeck_process() -> () {
        use crate::stochastic_processes::standard_stochastic_models::OrnsteinUhlenbeckProcess;
        let n_paths: usize = 1_000;
        let n_steps: usize = 200;
        let oup: OrnsteinUhlenbeckProcess = OrnsteinUhlenbeckProcess::new(
            0.07, 0.1, 10.0, n_paths, n_steps, 1.0, 0.05, true
        );
        let sp_result: StochasticProcessResult = oup.simulate().unwrap();
        assert_eq!(sp_result.paths.len(), n_paths);
        assert_eq!(sp_result.paths[0].len(), n_steps + 1);
        assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
        assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
        assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 1_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_brownian_bridge() -> () {
        use crate::stochastic_processes::standard_stochastic_models::BrownianBridge;
        let n_paths: usize = 1_000;
        let n_steps: usize = 200;
        let bb: BrownianBridge = BrownianBridge::new(1.0, 2.0, 0.5, n_paths, n_steps, 1.0);
        let sp_result: StochasticProcessResult = bb.simulate().unwrap();
        assert_eq!(sp_result.paths.len(), n_paths);
        assert_eq!(sp_result.paths[0].len(), n_steps + 1);
        assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
        assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
        assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 1_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_feller_square_root_process() -> () {
        use crate::stochastic_processes::standard_stochastic_models::{FellerSquareRootProcess, FSRSimulationMethod};
        let n_paths: usize = 1_000;
        let n_steps: usize = 200;
        let fsrp: FellerSquareRootProcess = FellerSquareRootProcess::new(
            0.05, 0.265, 5.0, n_paths, n_steps,1.0, 0.03, FSRSimulationMethod::EulerMaruyama,
        );
        let sp_result: StochasticProcessResult = fsrp.simulate().unwrap();
        assert_eq!(sp_result.paths.len(), n_paths);
        assert_eq!(sp_result.paths[0].len(), n_steps + 1);
        assert_eq!(sp_result.expectations_path.clone().unwrap().len(), n_steps + 1);
        assert_eq!(sp_result.variances_path.unwrap().len(), n_steps + 1);
        assert!((sp_result.mean - sp_result.expectations_path.unwrap()[n_steps]).abs() < 1_000_000.0 * TEST_ACCURACY);
    }
}