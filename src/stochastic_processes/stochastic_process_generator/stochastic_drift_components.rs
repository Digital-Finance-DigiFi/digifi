use std::io::Error;
use ndarray::Array1;
use crate::utilities::{input_error, data_error};
use crate::statistics::continuous_distributions::ContinuousUniformDistribution;
use crate::random_generators::{RandomGenerator, generator_algorithms::inverse_transform, standard_normal_generators::StandardNormalInverseTransform};


/// # Description
/// Type of drift that the stochastic process has.
pub enum StochasticDriftType {
    TrendStationary { trend: TrendStationary, s_0: f64, },
    DifferenceStationary { trend: DifferenceStationary, },
}


pub trait CustomTrendStationary {

    /// # Description
    /// Custom trend-stationary trend function.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    /// - `t`: Current time step
    fn trend_func(&self, n_paths: usize, t: f64) -> Result<Array1<f64>, Error>;

    /// # Description
    /// Validate custom trend-stationary component function object to satisfy the computational requirements.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    ///
    /// # Errors
    /// - Returns an error if the custom function does not return an array of the same length as there are number of paths.
    fn validate(&self, n_paths: usize) -> Result<(), Error> {
        let cont_uni_dist: ContinuousUniformDistribution = ContinuousUniformDistribution::new(0.0, 1.0)?;
        let t: f64 = inverse_transform(&cont_uni_dist, 1)?[0];
        if self.trend_func(n_paths, t)?.len() != n_paths {
            return Err(data_error("Custom Trend Stationary Function: Custom function does not produce an array of the same size as the defined number of paths."));
        }
        Ok(())
    }
}


/// # Description
/// The Trend Stationary enum handles the trend term for a discrete-time stochastic process.
/// This enum models trends such as linear, quadratic, or exponential, which can revert to a deterministic trend over time.
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Trend-stationary_process>
/// - Original Source: N/A
pub enum TrendStationary {
    Linear { a: f64, b: f64, },
    Quadratic { a: f64, b: f64, c: f64, },
    Exponential { a: f64, b: f64, },
    Custom { f: Box<dyn CustomTrendStationary> },
}

impl TrendStationary {

    /// # Description
    /// Validates the parameters of the trend stationary type.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    ///
    /// # Errors
    /// - Returns an error if the argument `a` of the quadratic trend type is zero.
    /// - Returns an error if the custom function does not return an array of the same length as there are number of paths.
    pub fn validate(&self, n_paths: usize) -> Result<(), Error> {
        match &self {
            TrendStationary::Linear { .. } => Ok(()),
            TrendStationary::Quadratic { a, .. } => {
                if a == &0.0 {
                    return Err(input_error("Trend Stationary Trend Type: The argument a for quadratic trend stationary trend type must be non-zero."));
                }
                Ok(())
            },
            TrendStationary::Exponential { .. } => Ok(()),
            TrendStationary::Custom { f } => { f.validate(n_paths) },
        }
    }

    /// # Description
    /// Generates an array of trend values based on the specified trend type.
    /// It accounts for different trends like linear, quadratic, exponential, or a custom-defined trend.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    /// - `t`: Current time step
    ///
    /// # Output
    /// - An array representing the trend values for each path
    pub fn get_stationary_trend(&self, n_paths: usize, t: f64) -> Result<Array1<f64>, Error> {
        let base_shape: Array1<f64> = Array1::from_vec(vec![1.0; n_paths]);
        match &self {
            TrendStationary::Linear { a, b } => { Ok((a*t + b) * base_shape) },
            TrendStationary::Quadratic { a, b, c } => { Ok((a*t.powi(2) + b*t + c) * base_shape) },
            TrendStationary::Exponential { a, b } => { Ok((b * (a * t).exp()) * base_shape) },
            TrendStationary::Custom { f } => { f.trend_func(n_paths, t) },
        }
    }
}


pub trait CustomStationaryError {

    /// # Description
    /// Custom error of the diffusion component.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    /// - `dt`: Time step increment
    fn error_func(&self, n_paths: usize, dt: f64) -> Result<Array1<f64>, Error>;

    /// # Description
    /// Validate custom error component function object to satisfy the computational requirements.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    ///
    /// # Errors
    /// - Returns an error if the custom function does not return an array of the same length as there are number of paths.
    fn validate(&self, n_paths: usize) -> Result<(), Error> {
        let cont_uni_dist: ContinuousUniformDistribution = ContinuousUniformDistribution::new(0.0, 1.0)?;
        let dt: f64 = inverse_transform(&cont_uni_dist, 1)?[0];
        if self.error_func(n_paths, dt)?.len() != n_paths {
            return Err(data_error("Custom Error Function: Custom function does not produce an array of the same size as the defined number of paths."));
        }
        Ok(())
    }
}


/// # Description
/// The StationaryError enum represents a stationary error term for a discrete-time stochastic process.
/// This error term can follow a Weiner process or be custom-defined, adding randomness to the process in a controlled manner.
pub enum StationaryError {
    Weiner {
        /// Standard deviation of the diffusion of the process
        sigma: f64,
    },
    Custom { f: Box<dyn CustomStationaryError> },
}

impl StationaryError {

    /// # Description
    /// Validates the parameters of the stationary error.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    ///
    /// # Errors
    /// - Returns an error if the custom function does not return an array of the same length as there are number of paths.
    pub fn validate(&self, n_paths: usize) -> Result<(), Error> {
        match &self {
            StationaryError::Custom { f } => { f.validate(n_paths) },
            _ => Ok(())
        }
    }

    /// # Description
    /// Generates an array of error values based on the defined error type. The method supports a Weiner process or a custom error generation mechanism.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    /// - `dt`: Time step increment
    ///
    /// # Output
    /// - An array representing the error values for each path
    pub fn get_error(&self, n_paths: usize, dt: f64) -> Result<Array1<f64>, Error> {
        match &self {
            StationaryError::Weiner { sigma } => {
                let n: Array1<f64> = StandardNormalInverseTransform::new_shuffle(n_paths)?.generate()?;
                Ok(sigma.powi(2) * dt * n)
            },
            StationaryError::Custom { f } => f.error_func(n_paths, dt)
        }
    }
}


/// # Description
/// The Difference Stationary struct represents a difference stationary term for a discrete-time stochastic process.
/// This term is crucial in processes characterized by a unit root, where shocks have a permanent effect on the level of the series.
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Unit_root>
/// - Original Source: N/A
pub struct DifferenceStationary {
    /// Number of paths to generate
    n_paths: usize,
    /// Constancts used in-front of the autoregressive stochastic values (e.g., a_{1}X_{1}+ a_{2}X_{2})
    autoregression_params: Array1<f64>,
    /// Starting values for the process (i.e., List of arrays, where each array is `n` starting values of the process)
    starting_values: Vec<Array1<f64>>,
}

impl DifferenceStationary {

    /// # Description
    /// Creates a new `DifferenceStationary` instance.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    /// - `autoregression_params`: Constancts used in-front of the autoregressive stochastic values (e.g., a_{1}X_{1}+ a_{2}X_{2})
    ///
    /// # Errors
    /// - Returns an error if the length of `staring_values` does not match the number of paths.
    /// - Returns an error if an array in `strating_values` has different length to parameters of autoregression.
    pub fn new(n_paths: usize, autoregression_params: Array1<f64>, starting_values: Vec<Array1<f64>>) -> Result<Self, Error> {
        DifferenceStationary::validate_values(n_paths, &starting_values)?;
        Ok(DifferenceStationary { n_paths, autoregression_params, starting_values })
    }

    /// # Description
    /// Validates the array of values.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    /// - `values`: Previous values of the process
    ///
    /// # Errors
    /// - Returns an error if the length of `values` does not match the number of paths.
    /// - Returns an error if an array in `values` has different length to parameters of autoregression.
    fn validate_values(n_paths: usize, values: &Vec<Array1<f64>>) -> Result<(), Error> {
        let process_order: usize = values.len();
        // Input validation
        if values.len() != n_paths {
            return Err(input_error("Difference Stationary: The length of previous values must match the number of paths."));
        }
        for v in values {
            if v.len() != process_order {
                return Err(input_error(format!("Difference Stationary: All arrays of previous values must match the order of process, {}.", process_order)));
            }
        }
        Ok(())
    }

    /// # Description
    /// Returns the starting values of the process.
    pub fn strating_values(&self) -> Vec<Array1<f64>> {
        self.starting_values.clone()
    }

    /// # Description
    /// Calculates autoregression values for the difference stationary process.
    /// This method considers previous values of the process and applies the autoregression parameters to generate the next term in the series.
    ///
    /// # Input
    /// - `previous_values`: List of arrays, where each array is `n` previous values of the process.
    ///
    /// # Output
    /// - An array of autoregression values for each path
    ///
    /// # Errors
    /// - Returns an error if the length of `previous_values` does not match the number of paths.
    /// - Returns an error if an array in `previous_values` has different length to parameters of autoregression.
    pub fn get_autoregression(&self, previous_values: &Vec<Array1<f64>>) -> Result<Vec<f64>, Error> {
        DifferenceStationary::validate_values(self.n_paths, previous_values)?;
        // Autoregression
        let mut result: Vec<f64> = Vec::<f64>::new();
        for process in previous_values {
            result.push(process.dot(&self.autoregression_params));
        }
        Ok(result)
    }
}