use std::io::Error;
use ndarray::{Array1, Axis};
use crate::utilities::input_error;
use crate::random_generators::RandomGenerator;
use crate::stochastic_processes::StochasticProcess;
use crate::random_generators::standard_normal_generators::StandardNormalInverseTransform;


/// # Description
/// Different methods of simulating the Feller Square-Root Process.
pub enum FSRSimulationMethod {
    EulerMaruyama,
    AnalyticEulerMaruyama,
}


/// # Description
/// Arithmetic Brownian motion.
/// 
/// # LaTeX Formula
/// - dS_{t} = \\mu*dt + \\sigma*dW_{t}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Geometric_Brownian_motion#:~:text=solution%20claimed%20above.-,Arithmetic%20Brownian%20Motion,-%5Bedit%5D
/// - Original Source: https://doi.org/10.24033/asens.476
pub struct ArithmeticBrownianMotion {
    /// Drift constant of the process
    mu: f64,
    /// Diffusion constant of the process
    sigma: f64,
    /// Number of paths to generate
    n_paths: usize,
    /// Number of steps
    _n_steps: usize,
    /// Final time step
    _t_f: f64,
    /// Initial value of the stochastic process
    s_0: f64,
    /// Time difference between two consequtive time steps
    dt: f64,
    /// Array of time steps
    t: Array1<f64>,
}

impl ArithmeticBrownianMotion {
    /// # Description
    /// Creates a new ArithmeticBrownianMotion instance.
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
        ArithmeticBrownianMotion { mu, sigma, n_paths, _n_steps: n_steps, _t_f: t_f, s_0, dt, t }
    }

    /// # Description
    /// Calculates the variance of the Arithmetic Brownian Motion at each time step.
    /// 
    /// # Output
    /// - An array of variances of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - Var[S_{t}] = \\sigma^{2} t
    pub fn get_variance(&self) -> Array1<f64> {
        &self.t * self.sigma.powi(2)
    }

    /// # Description
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
    /// - Returns an error if the index provided is out of bounds for the time array
    /// 
    /// # LaTeX Formula
    /// - \\textit{Cov}(S_{t_{1}}, S_{t_{2}}) = \\sigma^{2} \\min(S_{t_{1}}, S_{t_{2}})
    pub fn get_auto_cov(&self, index_t1: usize, index_t2: usize) -> Result<f64, Error> {
        let t_len: usize = self.t.len();
        if t_len < index_t1 {
            return Err(input_error(format!("Arithmetic Brownian Motion: The argument index_t1 is out of range for price array of length {}.", t_len)));
        }
        if t_len < index_t2 {
            return Err(input_error(format!("Arithmetic Brownian Motion: The argument index_t2 is out of range for price array of length {}.", t_len)));
        }
        Ok(self.sigma.powi(2) * self.t[index_t1].min(self.t[index_t2]))
    }
}

impl StochasticProcess for ArithmeticBrownianMotion {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths;
    }

    /// # Description
    /// Calculates the expected path of the Arithmetic Brownian Motion
    /// 
    /// # Output
    /// - An array of expected values of the stock price at each time step
    /// 
    /// # LaTeX Formula
    /// - E[S_t] = \\mu t + S_{0}
    fn get_expectations(&self) -> Array1<f64> {
        self.mu * &self.t + self.s_0
    }

    /// # Description
    /// Generates simulation paths for the Arithmetic Brownian Motion using the Euler-Maruyama method.
    /// 
    /// # Output
    /// - An array of simulated paths following the Arithmetic Brownian Motion
    /// 
    /// # LaTeX Formula
    /// - dS_{t} = \\mu dt + \\sigma dW_{t}
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, Error> {
        let mut paths: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
        for _ in 0..self.n_paths {
            let dw: Array1<f64> = self.dt.sqrt() * StandardNormalInverseTransform::new_shuffle(self.t.len())?.generate()?;
            let mut ds: Array1<f64> = self.mu * self.dt + self.sigma * dw;
            ds[0] = self.s_0;
            ds.accumulate_axis_inplace(Axis(0), |&prev, curr| { *curr += prev });
            paths.push(ds);
        }
        Ok(paths)
    }
}


/// # Description
/// Model describing the evolution of stock prices.
/// 
/// # LaTeX Formula
/// - dS_{t} = \\mu*S_{t}*dt + \\sigma*S_{t}*dW_{t}\n
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Geometric_Brownian_motion
/// - Original Source: http://dx.doi.org/10.1086/260062
pub struct GeometricBrownianMotion {
    /// Drift constant of the process
    mu: f64,
    /// Diffusion constant of the process
    sigma: f64,
    /// Number of paths to generate
    n_paths: usize,
    /// Number of steps
    _n_steps: usize,
    /// Final time step
    _t_f: f64,
    /// Initial value of the stochastic process
    s_0: f64,
    /// Time difference between two consequtive time steps
    dt: f64,
    /// Array of time steps
    t: Array1<f64>,
}

impl GeometricBrownianMotion {
    /// # Description
    /// Creates a new GoemetricBrownianMotion instance.
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
        GeometricBrownianMotion { mu, sigma, n_paths, _n_steps: n_steps, _t_f: t_f, s_0, dt, t }
    }

    /// # Description
    /// Computes the variance of the stock price at each time step under the Geometric Brownian Motion model.
    /// This provides an indication of the variability or risk associated with the stock price.
    /// 
    /// 
    /// # Output
    /// - An array of variances of the stock price at each time step
    /// 
    /// # LaTeX Formula
    /// - \\textit{Var}[S_{t}] = (S^{2}_{0}) e^{2\\mu t} (e^{\\sigma^{2} t} - 1)
    pub fn get_variance(&self) -> Array1<f64> {
        self.s_0.powi(2) * (2.0 * self.mu * &self.t).map(|i| i.exp() ) * ((&self.t * self.sigma.powi(2)).map(|i| i.exp() ) - 1.0)
    }
}

impl StochasticProcess for GeometricBrownianMotion {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths;
    }

    /// # Description
    /// Calculates the expected path of the Geometric Brownian Motion. This represents the mean trajectory of the stock price over time.
    /// 
    /// # Output
    /// - An array of expected values of the stock price at each time step, representing the mean trajectory
    /// 
    /// # LaTeX Formula:
    /// - E[S_t] = S_{0} e^{\\mu t}
    fn get_expectations(&self) -> Array1<f64> {
        self.s_0 * (self.mu * &self.t).map(|i| i.exp() )
    }

    /// # Description
    /// Simulates paths of the Geometric Brownian Motion using the Euler-Maruyama method.
    /// This method provides an approximation of the continuous-time process.
    /// 
    /// # Output
    /// - An array of simulated stock prices following the Geometric Brownian Motion for each path and time step
    /// 
    /// # LaTeX Formula
    /// - dS_{t} = \\mu S_{t} dt + \\sigma S_{t} dW_{t}
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, Error> {
        let mut paths: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
        for _ in 0..self.n_paths {
            let dw: Array1<f64> = self.dt.sqrt() * StandardNormalInverseTransform::new_shuffle(self.t.len())?.generate()?;
            let mut s: Array1<f64> = Array1::from_vec(vec![self.s_0; self.t.len()]);
            for i in 1..self.t.len() {
                s[i] = s[i-1] + self.mu * s[i-1] * self.dt + self.sigma * s[i-1] * dw[i];
            }
            paths.push(s);
        }
        Ok(paths)
    }
}


/// # Description
/// Model describes the evolution of interest rates.
/// 
/// # LaTeX Formula
/// - dS_{t} = \\alpha*(\\mu-S_{t})*dt + \\sigma*dW_{t}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
/// - Original Source: https://doi.org/10.1103%2FPhysRev.36.823
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
    _n_steps: usize,
    /// Final time step
    _t_f: f64,
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
    /// # Description
    /// Creates a new OrnsteinUhlenbeckProcess instance.
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
        OrnsteinUhlenbeckProcess { mu, sigma, alpha, n_paths, _n_steps: n_steps, _t_f: t_f, s_0, dt, t, analytic_em }
    }

    /// # Description
    /// Computes the variance of the Ornstein-Uhlenbeck Process at each time step, providing insights into the variability around the mean.
    /// 
    /// # Output
    /// - An array of variances of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - \\textit{Var}[S_{t}] = \\frac{\\sigma^{2}}{2\\alpha} (1 - e^{-2\\alpha t})
    pub fn get_variance(&self) -> Array1<f64> {
        (1.0 - (-2.0 * self.alpha * &self.t).map(|i| i.exp() )) * self.sigma.powi(2) / (2.0 * self.alpha)
    }
}

impl StochasticProcess for OrnsteinUhlenbeckProcess {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths
    }

    /// # Description
    /// Calculates the expected path of the Ornstein-Uhlenbeck Process, showing the mean-reverting nature of the process over time.
    /// 
    /// # Output
    /// - An array (np.ndarray) of expected values of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - E[S_t] = \\mu + (S_{0} - \\mu) e^{-\\alpha t}
    fn get_expectations(&self) -> Array1<f64> {
        self.mu + (self.s_0 * self.mu) * (-self.alpha * &self.t).map(|i| i.exp() )
    }

    /// # Description
    /// Simulates paths of the Ornstein-Uhlenbeck Process using the Euler-Maruyama method.
    /// This method can be adjusted to use either the standard numerical simulation or an analytic adjustment for Euler-Maruyama.
    /// 
    /// # Output
    /// - An array representing simulated paths of the process for each path and time step
    /// 
    /// # LaTeX Formula
    /// - dS_{t} = \\alpha(\\mu - S_{t}) dt + \\sigma dW_{t}
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, Error> {
        let mut paths: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
        let std: f64;
        if self.analytic_em {
            std = self.sigma * ((1.0 - (-2.0 * self.alpha * self.dt).exp()) / (2.0 * self.alpha)).sqrt();
        } else {
            std = self.sigma * self.dt.sqrt();
        }
        for _ in 0..self.n_paths {
            let mut s: Array1<f64> = Array1::from_vec(vec![self.s_0; self.t.len()]);
            let r: Array1<f64> = StandardNormalInverseTransform::new_shuffle(self.t.len())?.generate()?;
            if self.analytic_em {
                // Analytic Euler-Maruyama method
                for i in 1..self.t.len() {
                    s[i] = self.mu + (s[i-1] - self.mu) * (-self.alpha * self.dt).exp() + std * r[i-1]
                }
            } else {
                // Plain Euler-Maruyama method
                for i in 1..self.t.len() {
                    s[i] = s[i-1] + self.alpha * (self.mu - s[i]) * self.dt + std * r[i];
                }
            }

            paths.push(s);
        }
        Ok(paths)
    }
}


/// # Description
/// Model can support useful variance reduction techniques for pricing derivative contracts using Monte-Carlo simulation, 
/// such as sampling. Also used in scenario generation.
/// 
/// # LaTeX Formula
/// - dS_{t} = ((b-a)/(T-t))*dt + \\sigma*dW_{t}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Brownian_bridge
/// - Original Source: N/A
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
    _n_steps: usize,
    /// Final time step
    t_f: f64,
    /// Initial value of the stochastic process
    dt: f64,
    /// Array of time steps
    t: Array1<f64>,
}

impl BrownianBridge {
    /// # Description
    /// Creates a new OrnsteinUhlenbeckProcess instance.
    /// 
    /// # Input
    /// - `alpha`: Initial value of the process
    /// - `beta`: Final value of the process
    /// - `sigma`: Standard deviation of the process
    /// - `alpha`: Drift scaling parameter
    /// - `n_paths`: Number of paths to generate
    /// - `n_steps`: Number of steps
    /// - `t_f`: Final time step
    pub fn new(alpha: f64, beta: f64, sigma: f64, n_paths: usize, n_steps: usize, t_f: f64) -> Self {
        let dt: f64 = t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, t_f + dt, dt);
        BrownianBridge { alpha, beta, sigma, n_paths, _n_steps: n_steps, t_f, dt, t }
    }

    /// # Description
    /// Computes the variance of the Brownian Bridge at each time step.
    /// This illustrates how the variability of the process decreases as it approaches the endpoint 'beta' at time T.
    /// 
    /// # Output
    /// - An array of variances of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - \\text{Var}[S_{t}] = \\frac{t(T-t)}{T} \\sigma^{2}
    pub fn get_variance(&self) -> Array1<f64> {
        &self.t * (self.t_f - &self.t) / self.t_f
    }
}

impl StochasticProcess for BrownianBridge {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths
    }

    /// # Description
    /// Calculates the expected path of the Brownian Bridge. It represents the expected value of the process at each time step, starting at 'alpha' and trending towards 'beta'.
    /// 
    /// # Output
    /// - An array of expected values of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - E[S_{t}] = \\alpha + (\\beta - \\alpha) \\frac{t}{T}
    fn get_expectations(&self) -> Array1<f64> {
        self.alpha + (self.beta - self.alpha) / self.t_f * &self.t
    }

    /// # Description
    /// Generates simulation paths for the Brownian Bridge using the Euler-Maruyama method.
    /// 
    /// This method approximates the continuous-time process and ensures that the path starts at 'alpha' and ends at 'beta' at time t_f.
    /// 
    /// # Output
    /// - An array of simulated paths following the Brownian Bridge for each path and time step
    /// 
    /// # LaTeX Formula
    /// - dS_{t} = ((\\beta - \\alpha)/(T - t)) dt + \\sigma dW_{t}
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, Error> {
        let mut paths: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
        for _ in 0..self.n_paths {
            let dw: Array1<f64> = self.dt.sqrt() * StandardNormalInverseTransform::new_shuffle(self.t.len())?.generate()?;
            let mut s: Array1<f64> = Array1::from_vec(vec![self.alpha; self.t.len()]);
            s[self.t.len()-1] = self.beta;
            for i in 0..(self.t.len() - 2) {
                s[i+1] = s[i] + (self.beta - s[i]) / (self.t.len() - i) as f64 + self.sigma * dw[i];
            }
            paths.push(s)
        }
        Ok(paths)
    }
}


/// # Description
/// Model describes the evolution of interest rates.
/// 
/// # LaTeX Formula
/// - dS_{t} = \\alpha*(\\mu-S_{t})*dt + \\sigma\\sqrt(S_{t})*dW_{t}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model
/// - Original Source: https://doi.org/10.2307/1911242
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
    _n_steps: usize,
    /// Final time step
    _t_f: f64,
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
    /// # Description
    /// Creates a new FellerSquareRootProcess instance.
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
        FellerSquareRootProcess { mu, sigma, alpha, n_paths, _n_steps: n_steps, _t_f: t_f, s_0, dt, t, method }
    }

    /// # Description
    /// Computes the variance of the Feller Square-Root Process at each time step, providing insights into the variability around the mean.
    /// 
    /// # Output
    /// - An array of variances of the process at each time step
    pub fn get_variance(&self) -> Array1<f64> {
        let v_1: Array1<f64> = ((-self.alpha * &self.t).map(|i| i.exp() ) - (-self.alpha * 2.0 * &self.t).map(|i| i.exp() )) * self.s_0 / self.alpha;
        let v_2: Array1<f64> = (-self.alpha * 2.0 * &self.t).map(|i| i.exp() ) * ((self.alpha * &self.t).map(|i| i.exp() ) - 1.0).map(|i| i.powi(2) ) * self.mu / (2.0 * self.alpha);
        self.sigma.powi(2) * (v_1 + v_2)
    }
}

impl StochasticProcess for FellerSquareRootProcess {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths
    }

    /// # Description
    /// Calculates the expected path of the Feller Square-Root Process, showing the mean-reverting nature over time towards the long-term mean \\mu.
    /// 
    /// # Output
    /// - An array of expected values of the process at each time step
    /// 
    /// # LaTeX Formula
    /// - E[S_{t}] = \\mu + (S_{0} - \\mu) e^{-\\alpha t}
    fn get_expectations(&self) -> Array1<f64> {
        self.mu + (self.s_0 - self.mu) * (-self.alpha * &self.t).map(|i| i.exp() )
    }

    /// # Description
    /// Simulates paths of the Feller Square-Root Process using different methods: Euler-Maruyama, Analytic Euler-Maruyama,
    /// or Exact method, depending on the specified method in the process setup.
    /// 
    /// # Output
    /// - An array of simulated paths of the process for each path and time step, following the chosen simulation method
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, Error> {
        let mut paths: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
        for _ in 0..self.n_paths {
            let mut s: Array1<f64> = Array1::from_vec(vec![self.s_0; self.t.len()]);
            let r: Array1<f64> = StandardNormalInverseTransform::new_shuffle(self.t.len())?.generate()?;
            match self.method {
                FSRSimulationMethod::EulerMaruyama => {
                    for i in 1..self.t.len() {
                        s[i] = s[i-1] + self.alpha * (self.mu - s[i-1]) * self.dt + self.sigma * (s[i-1] * self.dt).sqrt() * r[i-1];
                        s[i] = s[i].max(0.0);
                    }
                },
                FSRSimulationMethod::AnalyticEulerMaruyama => {
                    let a: f64 = self.sigma.powi(2) / self.alpha * ((-self.alpha * self.dt).exp() - (-2.0 * self.alpha * self.dt).exp());
                    let b: f64 = self.mu * self.sigma.powi(2) / (2.0 * self.alpha) * (1.0 - (-self.alpha * self.dt).exp()).powi(2);
                    for i in 1..self.t.len() {
                        s[i] = self.mu + (s[i-1] - self.mu) * (-self.alpha * self.dt).exp() + (a * s[i-1] + b).sqrt() * r[i-1];
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
    use ndarray::Array1;
    use crate::stochastic_processes::StochasticProcess;
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_arithmetic_brownian_motion() -> () {
        use crate::stochastic_processes::standard_stochastic_models::ArithmeticBrownianMotion;
        let n_paths: usize = 100;
        let n_steps: usize = 200;
        let abm: ArithmeticBrownianMotion = ArithmeticBrownianMotion::new(1.0, 0.4, n_paths, n_steps, 1.0, 100.0);
        let paths: Vec<Array1<f64>> = abm.get_paths().unwrap();
        assert_eq!(paths.len(), n_paths);
        assert_eq!(paths[0].len(), n_steps+1);
        let mut final_steps: Vec<f64> = Vec::<f64>::new();
        for i in 0..n_paths {
            final_steps.push(paths[i][n_steps]);
        }
        let final_steps: Array1<f64> = Array1::from_vec(final_steps);
        let expected_path: Array1<f64> = abm.get_expectations();
        assert!((final_steps.mean().unwrap() - expected_path.last().unwrap()).abs() < 10_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_geometric_brownian_motion() -> () {
        use crate::stochastic_processes::standard_stochastic_models::GeometricBrownianMotion;
        let n_paths: usize = 1_000;
        let n_steps: usize = 200;
        let gbm: GeometricBrownianMotion = GeometricBrownianMotion::new(0.0, 0.2, n_paths, n_steps, 1.0, 100.0);
        let paths: Vec<Array1<f64>> = gbm.get_paths().unwrap();
        assert_eq!(paths.len(), n_paths);
        assert_eq!(paths[0].len(), n_steps+1);
        let mut final_steps: Vec<f64> = Vec::<f64>::new();
        for i in 0..n_paths {
            final_steps.push(paths[i][n_steps]);
        }
        let final_steps: Array1<f64> = Array1::from_vec(final_steps);
        let expected_path: Array1<f64> = gbm.get_expectations();
        assert!((final_steps.mean().unwrap() - expected_path.last().unwrap()).abs() < 100_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_ornstein_uhlenbeck_process() -> () {
        use crate::stochastic_processes::standard_stochastic_models::OrnsteinUhlenbeckProcess;
        let n_paths: usize = 100;
        let n_steps: usize = 200;
        let oup: OrnsteinUhlenbeckProcess = OrnsteinUhlenbeckProcess::new(0.07, 0.1, 10.0, n_paths, n_steps, 1.0, 0.05, true);
        let paths: Vec<Array1<f64>> = oup.get_paths().unwrap();
        assert_eq!(paths.len(), n_paths);
        assert_eq!(paths[0].len(), n_steps+1);
        let mut final_steps: Vec<f64> = Vec::<f64>::new();
        for i in 0..n_paths {
            final_steps.push(paths[i][n_steps]);
        }
        let final_steps: Array1<f64> = Array1::from_vec(final_steps);
        let expected_path: Array1<f64> = oup.get_expectations();
        assert!((final_steps.mean().unwrap() - expected_path.last().unwrap()).abs() < 1_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_brownian_bridge() -> () {
        use crate::stochastic_processes::standard_stochastic_models::BrownianBridge;
        let n_paths: usize = 1;
        let n_steps: usize = 200;
        let bb: BrownianBridge = BrownianBridge::new(1.0, 2.0, 0.5, n_paths, n_steps, 1.0);
        let paths: Vec<Array1<f64>> = bb.get_paths().unwrap();
        assert_eq!(paths.len(), n_paths);
        assert_eq!(paths[0].len(), n_steps+1);
        let mut final_steps: Vec<f64> = Vec::<f64>::new();
        for i in 0..n_paths {
           final_steps.push(paths[i][n_steps-1]);
        }
        let final_steps: Array1<f64> = Array1::from_vec(final_steps);
        let expected_path: Array1<f64> = bb.get_expectations();
        assert!((final_steps.mean().unwrap() - expected_path[expected_path.len()-2]).abs() < 10_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_feller_square_root_process() -> () {
        use crate::stochastic_processes::standard_stochastic_models::{FellerSquareRootProcess, FSRSimulationMethod};
        let n_paths: usize = 1;
        let n_steps: usize = 200;
        let fsrp: FellerSquareRootProcess = FellerSquareRootProcess::new(0.05, 0.265, 5.0, n_paths, n_steps,
                                                                         1.0, 0.03, FSRSimulationMethod::EulerMaruyama);
        let paths: Vec<Array1<f64>> = fsrp.get_paths().unwrap();
        assert_eq!(paths.len(), n_paths);
        assert_eq!(paths[0].len(), n_steps+1);
        let mut final_steps: Vec<f64> = Vec::<f64>::new();
        for i in 0..n_paths {
           final_steps.push(paths[i][n_steps]);
        }
        let final_steps: Array1<f64> = Array1::from_vec(final_steps);
        let expected_path: Array1<f64> = fsrp.get_expectations();
        assert!((final_steps.mean().unwrap() - expected_path.last().unwrap()).abs() < 10_000_000.0 * TEST_ACCURACY);
    }
}