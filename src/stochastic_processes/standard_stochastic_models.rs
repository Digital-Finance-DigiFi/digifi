use ndarray::{Array1, Axis};

use crate::random_generators::RandomGenerator;
use crate::stochastic_processes::StochasticProcess;
use crate::random_generators::standard_normal_generators::StandardNormalBoxMuller;


/// # Description
/// Different methods of simulating the Feller Square-Root Process.
pub enum FSRSimulationMethod {
    EulerMaruyama,
    AnalyticEulerMaruyama,
    Exact,
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
    n_steps: usize,
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
    /// - mu: Drift constant of the process
    /// - sigma: Diffusion constant of the process
    /// - n_paths: Number of paths to generate
    /// - n_steps: Number of steps
    /// - t_f: Final time step
    /// - s_0: Initial value of the stochastic process
    pub fn new(mu: f64, sigma: f64, n_paths: usize, n_steps: usize, _t_f: f64, s_0: f64) -> Self {
        let dt: f64 = _t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, _t_f+dt, dt);
        ArithmeticBrownianMotion { mu, sigma, n_paths, n_steps, _t_f, s_0, dt, t }
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
    /// - index_t1: Index of the first time point
    /// - index_t2: Index of the second time point
    /// 
    /// # Output
    /// - Auto-covariance of the process between times t1 and t2
    /// 
    /// # Panics
    /// - Panics if the index provided is out of bounds for the time array
    /// 
    /// # LaTeX Formula
    /// - \\textit{Cov}(S_{t_{1}}, S_{t_{2}}) = \\sigma^{2} \\min(S_{t_{1}}, S_{t_{2}})
    pub fn get_auto_cov(&self, index_t1: usize, index_t2: usize) -> f64 {
        let t_len: usize = self.t.len();
        if t_len < index_t1 {
            panic!("The argument index_t1 is out of range for price array of length {}.", t_len)
        }
        if t_len < index_t2 {
            panic!("The argument index_t2 is out of range for price array of length {}.", t_len)
        }
        self.sigma.powi(2) * self.t[index_t1].min(self.t[index_t2])
    }
}

impl StochasticProcess for ArithmeticBrownianMotion {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths
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
        self.mu * self.t.clone() + self.s_0
    }

    /// # Description
    /// Generates simulation paths for the Arithmetic Brownian Motion using the Euler-Maruyama method.
    /// 
    /// # Output
    /// - An array of simulated paths following the Arithmetic Brownian Motion
    /// 
    /// # LaTeX Formula
    /// - dS_{t} = \\mu dt + \\sigma dW_{t}
    fn get_paths(&self) -> Vec<Array1<f64>> {
        let mut paths: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
        for _ in 0..self.n_paths {
            let dw: Array1<f64> = self.dt.sqrt() * StandardNormalBoxMuller::new_shuffle(self.n_steps).generate();
            let mut ds: Array1<f64> = self.mu * self.dt + self.sigma * dw;
            ds[0] = self.s_0;
            ds.accumulate_axis_inplace(Axis(0), |&prev, curr| { *curr += prev });
            paths.push(ds);
        }
        paths
    }
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::stochastic_processes::StochasticProcess;
    use crate::statistics::covariance;

    #[test]
    fn unit_test_arithmetic_brownian_motion() -> () {
        use crate::stochastic_processes::standard_stochastic_models::ArithmeticBrownianMotion;
        let n_paths: usize = 100;
        let n_steps: usize = 200;
        let abm: ArithmeticBrownianMotion = ArithmeticBrownianMotion::new(1.0, 0.4, n_paths, n_steps, 1.0, 100.0);
        let paths: Vec<Array1<f64>> = abm.get_paths();
        assert_eq!(paths.len(), n_paths);
        assert_eq!(paths[0].len(), n_steps);
        // let mut final_steps: Vec<f64> = Vec::<f64>::new();
        // for i in 0..paths.len() {
        //     final_steps.push(paths[i][n_steps-1]);
        // }
        // let final_steps: Array1<f64> = Array1::from_vec(final_steps);
        // println!("Mean: {}", final_steps.mean().unwrap());
        // println!("Std: {}", covariance(&final_steps, &final_steps, 0));
        // let expected_path: Array1<f64> = abm.get_expectations();
        // let variances: Array1<f64> = abm.get_variance();
        // println!("Expected end point: {}", expected_path[expected_path.len()-1]);
        // println!("Variance: {}", variances[variances.len()-1]);
    }
}