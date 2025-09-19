// Re-Exports
pub use self::sde_components::{CustomSDEComponent, SDEComponent, CustomNoise, Noise, CustomJump, Jump};
pub use self::stochastic_drift_components::{StochasticDriftType, CustomTrendStationary, TrendStationary, CustomStationaryError, StationaryError, DifferenceStationary};


pub mod stochastic_drift_components;
pub mod sde_components;


use ndarray::Array1;
use crate::error::DigiFiError;
use crate::stochastic_processes::StochasticProcess;


fn transpose(v: Vec<Array1<f64>>) -> Vec<Array1<f64>> {
    let length: usize = v[0].len();
    // Transpose the vec of steps into a vec of paths
    let mut iters: Vec<_> = v.into_iter().map(|n| n.to_vec().into_iter() ).collect();
    (0..length).map(|_| {
        Array1::from_vec(iters.iter_mut().map(|n| n.next().unwrap() ).collect::<Vec<f64>>())
    }).collect()
}


/// Defines an SDE in the generalized form.
///
/// SDE = Drift + Diffusion + Jump
///
/// # LaTeX Formula
/// - dS_{t} = \\mu(S_{t}, t)dt + \\sigma(S_{t}, t)dW_{t} + dJ
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Stochastic_differential_equation>
/// - Original Source: N/A
///
/// # Examples
///
/// 1. Generating Arithmetic Brownian Motion SDE:
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::stochastic_processes::StochasticProcess;
/// use digifi::stochastic_processes::stochastic_process_generator::{SDEComponent, Noise, Jump, SDE};
///
/// // Parameter definition
/// let n_paths: usize = 100;
/// let n_steps: usize = 200;
/// let mu: f64 = 10.0;
/// let t_f: f64 = 1.0;
/// let s_0: f64 = 100.0;
/// let drift_component: SDEComponent = SDEComponent::Linear { a: mu };
/// let diffusion_component: SDEComponent = SDEComponent::Linear { a: 0.4 };
/// let noise: Noise = Noise::WeinerProcess;
/// let jump: Jump = Jump::NoJumps;
///
/// // SDE definition
/// let sde: SDE = SDE::build(t_f, s_0, n_steps, n_paths, drift_component, diffusion_component, noise, jump).unwrap();
/// let paths: Vec<Array1<f64>> = sde.get_paths().unwrap();
///
/// // Tests
/// assert_eq!(paths.len(), n_paths);
/// assert_eq!(paths[0].len(), n_steps+1);
/// let mut final_steps: Vec<f64> = Vec::with_capacity(n_paths);
/// for i in 0..n_paths {
///     final_steps.push(paths[i][n_steps]);
/// }
/// let final_steps: Array1<f64> = Array1::from_vec(final_steps);
/// let expected_value: f64 = mu * t_f + s_0;
/// assert!((final_steps.mean().unwrap() - expected_value).abs() < 50_000_000.0 * TEST_ACCURACY);
/// ```
///
/// 2. Generating Geometric Brownian Motion SDE:
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::stochastic_processes::StochasticProcess;
/// use digifi::stochastic_processes::stochastic_process_generator::{SDEComponent, Noise, Jump, SDE};
///
/// // Parameter definition
/// let n_paths: usize = 1_000;
/// let n_steps: usize = 200;
/// let mu: f64 = 0.1;
/// let sigma: f64 = 0.1;
/// let t_f: f64 = 1.0;
/// let s_0: f64 = 100.0;
/// let drift_component: SDEComponent = SDEComponent::RegressionToTrend { scale: -mu, trend: 0.0 };
/// let diffusion_component: SDEComponent = SDEComponent::RegressionToTrend { scale: -sigma, trend: 0.0 };
/// let noise: Noise = Noise::WeinerProcess;
/// let jump: Jump = Jump::NoJumps;
///
/// // SDE definition
/// let sde: SDE = SDE::build(t_f, s_0, n_steps, n_paths, drift_component, diffusion_component, noise, jump).unwrap();
/// let paths: Vec<Array1<f64>> = sde.get_paths().unwrap();
///
/// // Tests
/// assert_eq!(paths.len(), n_paths);
/// assert_eq!(paths[0].len(), n_steps+1);
/// let mut final_steps: Vec<f64> = Vec::with_capacity(n_paths);
/// for i in 0..n_paths {
///     final_steps.push(paths[i][n_steps]);
/// }
/// let final_steps: Array1<f64> = Array1::from_vec(final_steps);
/// let expected_value: f64 = (mu * t_f).exp() * s_0;
/// assert!((final_steps.mean().unwrap() - expected_value).abs() < 100_000_000.0 * TEST_ACCURACY);
/// ```
pub struct SDE {
    /// Final time step
    t_f: f64,
    /// Starting point for the process
    s_0: f64,
    /// Number of steps a process makes after the starting point
    n_steps: usize,
    /// Number of paths to return
    n_paths: usize,
    /// Type of function \\mu(S_{t}, t) to be used when constructing the drift
    drift_component: SDEComponent,
    /// Type of function \\sigma(S_{t}, t) to be used when constructing the diffusion
    diffusion_component: SDEComponent,
    /// Type of noise to be used when constructing the diffusion
    noise: Noise,
    /// Type of function dJ to be used when constructing the jump
    jump: Jump,
    /// Time difference between two consequtive time steps
    dt: f64,
    /// Array of time steps
    t: Array1<f64>,
}

impl SDE {

    /// Creates a new `SDE` instance.
    ///
    /// # Input
    /// - `t_f`: Final time step
    /// - `s_0`: Starting point for the process
    /// - `n_steps`: Number of steps a process makes after the starting point
    /// - `n_paths`: Number of paths to return
    /// - `drift_component`: Type of function \\mu(S_{t}, t) to be used when constructing the drift
    /// - `diffusion_component`: Type of function \\sigma(S_{t}, t) to be used when constructing the diffusion
    /// - `noise`: Type of noise to be used when constructing the diffusion
    /// - `jump`: Type of function dJ to be used when constructing the jump
    pub fn build(
        t_f: f64, s_0: f64, n_steps: usize, n_paths: usize, drift_component: SDEComponent, diffusion_component: SDEComponent, noise: Noise, jump: Jump
    ) -> Result<Self, DigiFiError> {
        // Input validation
        drift_component.validate(n_paths)?;
        diffusion_component.validate(n_paths)?;
        noise.validate(n_paths)?;
        jump.validate(n_paths)?;
        // Definition of SDE parameters
        let dt: f64 = t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, t_f + dt, dt);
        Ok(SDE { t_f, s_0, n_steps, n_paths, drift_component, diffusion_component, noise, jump, dt, t, })
    }
}

impl StochasticProcess for SDE {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths;
    }

    fn get_n_steps(&self) -> usize {
        self.n_steps
    }

    fn get_t_f(&self) -> f64 {
        self.t_f
    }

    /// Generate paths for the constructed SDE.
    ///
    /// # Output
    /// - An array of random paths
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, DigiFiError> {
        let mut time_steps: Vec<Array1<f64>> = vec![Array1::from_vec(vec![self.s_0; self.n_paths])];
        for i in 1..self.t.len() {
            let mut time_slice: Array1<f64> = self.drift_component.get_component_values(self.n_paths, &time_steps[time_steps.len() - 1], self.t[i], self.dt)? * self.dt
                + self.diffusion_component.get_component_values(self.n_paths, &time_steps[time_steps.len() - 1], self.t[i], self.dt)? * self.noise.get_noise(self.n_paths, self.dt)?
                + self.jump.get_jumps(self.n_paths, self.dt)?;
            time_slice += &time_steps[time_steps.len() - 1];
            time_steps.push(time_slice);
        }
        Ok(transpose(time_steps))
    }
}


/// Defines a stochastic process in the generalized where the stochastic drift can be difference-stationary (i.e., autoregressive)
/// or trend-stationary (i.e., only time dependent).
///
/// Stochastic Drift Process = Trend + Stationary Error
///
/// # LaTeX Formula
/// - S_{t} = \\mu(S_{0}, ..., S_{t-1}, t) + e_{t}
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Stochastic_drift>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::stochastic_processes::StochasticProcess;
/// use digifi::stochastic_processes::stochastic_process_generator::{StochasticDriftType, TrendStationary, StationaryError, StochasticDrift};
///
/// // Parameters definition
/// let n_paths: usize = 1_000;
/// let n_steps: usize = 200;
/// let mu: f64 = 0.1;
/// let sigma: f64 = 0.1;
/// let t_f: f64 = 1.0;
/// let s_0: f64 = 100.0;
/// let trend: TrendStationary = TrendStationary::Exponential { a: (mu - 0.5 * sigma.powi(2)), b: 1.0 };
/// let stochastic_drift_type: StochasticDriftType = StochasticDriftType::TrendStationary { trend, s_0 };
/// let error: StationaryError = StationaryError::Weiner { sigma };
///
/// // Process definition
/// let process: StochasticDrift = StochasticDrift::build(t_f, n_steps, n_paths, stochastic_drift_type, error).unwrap();
/// let paths: Vec<Array1<f64>> = process.get_paths().unwrap();
///
/// // Tests
/// assert_eq!(paths.len(), n_paths);
/// assert_eq!(paths[0].len(), n_steps+1);
/// let mut final_steps: Vec<f64> = Vec::with_capacity(n_paths);
/// for i in 0..n_paths {
///     final_steps.push(paths[i][n_steps]);
/// }
/// let final_steps: Array1<f64> = Array1::from_vec(final_steps);
/// let expected_value: f64 = (mu * t_f).exp() * s_0;
/// assert!((final_steps.mean().unwrap() - expected_value).abs() < 100_000_000.0 * TEST_ACCURACY);
/// ```
pub struct StochasticDrift {
    /// Final time step
    t_f: f64,
    /// Number of steps a process makes after the starting point
    n_steps: usize,
    /// Number of paths to return
    n_paths: usize,
    /// Type of stochastic drift
    stochastic_drift_type: StochasticDriftType,
    /// Type of error function to be used in the construction of an error term
    error: StationaryError,
    /// Time difference between two consequtive time steps
    dt: f64,
    /// Array of time steps
    t: Array1<f64>,
}

impl StochasticDrift {

    /// Creates a new `StochasticDrift` instance.
    ///
    /// # Input
    /// - `t_f`: Final time step
    /// - `n_steps`: Number of steps a process makes after the starting point
    /// - `n_paths`: Number of paths to return
    /// - `stochastic_drift_type`: Type of stochastic drift
    /// - `error`: Type of error function to be used in the construction of an error term
    pub fn build(t_f: f64, n_steps: usize, n_paths: usize, stochastic_drift_type: StochasticDriftType, error: StationaryError) -> Result<Self, DigiFiError> {
        // Input validation
        match &stochastic_drift_type {
            StochasticDriftType::TrendStationary { trend, .. } => { trend.validate(n_paths)?; },
            _ => (),
        }
        error.validate(n_paths)?;
        // Definition of SDE parameters
        let dt: f64 = t_f / (n_steps as f64);
        let t: Array1<f64> = Array1::range(0.0, t_f + dt, dt);
        Ok(StochasticDrift { t_f, n_steps, n_paths, stochastic_drift_type, error, dt, t, })
    }
}

impl StochasticProcess for StochasticDrift {
    fn update_n_paths(&mut self, n_paths: usize) -> () {
        self.n_paths = n_paths;
    }

    fn get_n_steps(&self) -> usize {
        self.n_steps
    }

    fn get_t_f(&self) -> f64 {
        self.t_f
    }

    /// Generate paths for the constructed process.
    ///
    /// # Output
    /// - An array of random paths
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, DigiFiError> {
        let mut time_steps: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
        match &self.stochastic_drift_type {
            StochasticDriftType::TrendStationary { trend, s_0 } => {
                time_steps.push(Array1::from_vec(vec![*s_0; self.n_paths]));
                for i in 1..self.t.len() {
                    let time_slice: Array1<f64>;
                    match &trend {
                        TrendStationary::Exponential { .. } => {
                            time_slice = *s_0 * trend.get_stationary_trend(self.n_paths, self.t[i])? * self.error.get_error(self.n_paths, self.dt)?.map(|v| v.exp() ) ;
                        },
                        _ => {
                            time_slice = *s_0 + trend.get_stationary_trend(self.n_paths, self.t[i])? + self.error.get_error(self.n_paths, self.dt)?;
                        },
                    }
                    time_steps.push(time_slice);
                }

            },
            StochasticDriftType::DifferenceStationary { trend } => {
                time_steps.extend(transpose(trend.strating_values()));
                let mut previous_values: Vec<Array1<f64>> = trend.strating_values();
                for _ in 1..self.t.len() {
                    let autoreg: Vec<f64> = trend.get_autoregression(&previous_values)?;
                    // Update previous values
                    for j in 0..previous_values.len() {
                        let mut values: Vec<f64> = previous_values[j].to_vec();
                        values.remove(0);
                        values.push(autoreg[j]);
                        previous_values[j] = Array1::from_vec(values);
                    }
                    time_steps.push(Array1::from_vec(autoreg) + self.error.get_error(self.n_paths, self.dt)?);
                }
            },
        }
        Ok(transpose(time_steps))
    }
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::utilities::TEST_ACCURACY;
    use crate::stochastic_processes::StochasticProcess;
    use crate::stochastic_processes::stochastic_process_generator::sde_components::{SDEComponent, Noise, Jump};
    use crate::stochastic_processes::stochastic_process_generator::stochastic_drift_components::{StochasticDriftType, TrendStationary, StationaryError};
    use crate::stochastic_processes::stochastic_process_generator::{SDE, StochasticDrift};

    #[test]
    fn unit_test_sde_abm() -> () {
        // Parameter definition
        let n_paths: usize = 100;
        let n_steps: usize = 200;
        let mu: f64 = 10.0;
        let t_f: f64 = 1.0;
        let s_0: f64 = 100.0;
        let drift_component: SDEComponent = SDEComponent::Linear { a: mu };
        let diffusion_component: SDEComponent = SDEComponent::Linear { a: 0.4 };
        let noise: Noise = Noise::WeinerProcess;
        let jump: Jump = Jump::NoJumps;
        // SDE definition
        let sde: SDE = SDE::build(t_f, s_0, n_steps, n_paths, drift_component, diffusion_component, noise, jump).unwrap();
        let paths: Vec<Array1<f64>> = sde.get_paths().unwrap();
        // Tests
        assert_eq!(paths.len(), n_paths);
        assert_eq!(paths[0].len(), n_steps+1);
        let mut final_steps: Vec<f64> = Vec::with_capacity(n_paths);
        for i in 0..n_paths {
            final_steps.push(paths[i][n_steps]);
        }
        let final_steps: Array1<f64> = Array1::from_vec(final_steps);
        let expected_value: f64 = mu * t_f + s_0;
        assert!((final_steps.mean().unwrap() - expected_value).abs() < 50_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_sde_gbm() -> () {
        // Parameter definition
        let n_paths: usize = 1_000;
        let n_steps: usize = 200;
        let mu: f64 = 0.1;
        let sigma: f64 = 0.1;
        let t_f: f64 = 1.0;
        let s_0: f64 = 100.0;
        let drift_component: SDEComponent = SDEComponent::RegressionToTrend { scale: -mu, trend: 0.0 };
        let diffusion_component: SDEComponent = SDEComponent::RegressionToTrend { scale: -sigma, trend: 0.0 };
        let noise: Noise = Noise::WeinerProcess;
        let jump: Jump = Jump::NoJumps;
        // SDE definition
        let sde: SDE = SDE::build(t_f, s_0, n_steps, n_paths, drift_component, diffusion_component, noise, jump).unwrap();
        let paths: Vec<Array1<f64>> = sde.get_paths().unwrap();
        // Tests
        assert_eq!(paths.len(), n_paths);
        assert_eq!(paths[0].len(), n_steps+1);
        let mut final_steps: Vec<f64> = Vec::with_capacity(n_paths);
        for i in 0..n_paths {
            final_steps.push(paths[i][n_steps]);
        }
        let final_steps: Array1<f64> = Array1::from_vec(final_steps);
        let expected_value: f64 = (mu * t_f).exp() * s_0;
        assert!((final_steps.mean().unwrap() - expected_value).abs() < 100_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_stochastic_drift_gbm() -> () {
        // Parameters definition
        let n_paths: usize = 1_000;
        let n_steps: usize = 200;
        let mu: f64 = 0.1;
        let sigma: f64 = 0.1;
        let t_f: f64 = 1.0;
        let s_0: f64 = 100.0;
        let trend: TrendStationary = TrendStationary::Exponential { a: (mu - 0.5 * sigma.powi(2)), b: 1.0 };
        let stochastic_drift_type: StochasticDriftType = StochasticDriftType::TrendStationary { trend, s_0 };
        let error: StationaryError = StationaryError::Weiner { sigma };
        // Process definition
        let process: StochasticDrift = StochasticDrift::build(t_f, n_steps, n_paths, stochastic_drift_type, error).unwrap();
        let paths: Vec<Array1<f64>> = process.get_paths().unwrap();
        // Tests
        assert_eq!(paths.len(), n_paths);
        assert_eq!(paths[0].len(), n_steps+1);
        let mut final_steps: Vec<f64> = Vec::with_capacity(n_paths);
        for i in 0..n_paths {
            final_steps.push(paths[i][n_steps]);
        }
        let final_steps: Array1<f64> = Array1::from_vec(final_steps);
        let expected_value: f64 = (mu * t_f).exp() * s_0;
        assert!((final_steps.mean().unwrap() - expected_value).abs() < 100_000_000.0 * TEST_ACCURACY);
    }
}