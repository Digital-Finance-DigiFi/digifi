use ndarray::Array1;
use crate::error::{DigiFiError, ErrorTitle};
use crate::random_generators::{RandomGenerator, uniform_generators::FibonacciGenerator, standard_normal_generators::StandardNormalBoxMuller};


/// Type of drift that the stochastic process has.
pub enum StochasticDriftType {
    TrendStationary { trend: TrendStationary, s_0: f64, },
    DifferenceStationary { trend: DifferenceStationary, },
}


pub trait CustomTrendStationary {
    /// Custom trend-stationary trend function.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    /// - `t`: Current time step
    fn trend_func(&self, n_paths: usize, t: f64) -> Result<Array1<f64>, DigiFiError>;

    /// Error title associated with this custom trend-stationary term.
    /// 
    /// Note: This title will be displayed in the case of validation error.
    fn error_title(&self) -> String;

    /// Validate custom trend-stationary component function object to satisfy the computational requirements.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    ///
    /// # Errors
    /// - Returns an error if the custom function does not return an array of the same length as there are number of paths.
    fn validate(&self, n_paths: usize) -> Result<(), DigiFiError> {
        let t: f64 = FibonacciGenerator::new_shuffle(1)?.generate()?[0];
        if self.trend_func(n_paths, t)?.len() != n_paths {
            return Err(DigiFiError::CustomFunctionLengthVal { title: self.error_title() });
        }
        Ok(())
    }
}


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
    /// Validates the parameters of the trend stationary type.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    ///
    /// # Errors
    /// - Returns an error if the argument `a` of the quadratic trend type is zero.
    /// - Returns an error if the custom function does not return an array of the same length as there are number of paths.
    pub fn validate(&self, n_paths: usize) -> Result<(), DigiFiError> {
        match &self {
            Self::Linear { .. } => Ok(()),
            Self::Quadratic { a, .. } => {
                if *a == 0.0 {
                    return Err(DigiFiError::ParameterConstraint {
                        title: Self::error_title(),
                        constraint: "The argument `a` for quadratic trend stationary trend type must be non-zero.".to_owned(),
                    });
                }
                Ok(())
            },
            Self::Exponential { .. } => Ok(()),
            Self::Custom { f } => { f.validate(n_paths) },
        }
    }

    /// Generates an array of trend values based on the specified trend type.
    /// It accounts for different trends like linear, quadratic, exponential, or a custom-defined trend.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    /// - `t`: Current time step
    ///
    /// # Output
    /// - An array representing the trend values for each path
    pub fn get_stationary_trend(&self, n_paths: usize, t: f64) -> Result<Array1<f64>, DigiFiError> {
        let base_shape: Array1<f64> = Array1::from_vec(vec![1.0; n_paths]);
        match &self {
            Self::Linear { a, b } => { Ok((a*t + b) * base_shape) },
            Self::Quadratic { a, b, c } => { Ok((a * t.powi(2) + b * t + c) * base_shape) },
            Self::Exponential { a, b } => { Ok((b * (a * t).exp()) * base_shape) },
            Self::Custom { f } => { f.trend_func(n_paths, t) },
        }
    }
}

impl ErrorTitle for TrendStationary {
    fn error_title() -> String {
        String::from("Trend Stationary Trend Type")
    }
}


pub trait CustomStationaryError {
    /// Custom error of the diffusion component.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    /// - `dt`: Time step increment
    fn error_func(&self, n_paths: usize, dt: f64) -> Result<Array1<f64>, DigiFiError>;

    /// Error title associated with this custom stationary error term.
    /// 
    /// Note: This title will be displayed in the case of validation error.
    fn error_title(&self) -> String;

    /// Validate custom error component function object to satisfy the computational requirements.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    ///
    /// # Errors
    /// - Returns an error if the custom function does not return an array of the same length as there are number of paths.
    fn validate(&self, n_paths: usize) -> Result<(), DigiFiError> {
        let dt: f64 = FibonacciGenerator::new_shuffle(1)?.generate()?[0];
        if self.error_func(n_paths, dt)?.len() != n_paths {
            return Err(DigiFiError::CustomFunctionLengthVal { title: self.error_title() });
        }
        Ok(())
    }
}


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
    /// Validates the parameters of the stationary error.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    ///
    /// # Errors
    /// - Returns an error if the custom function does not return an array of the same length as there are number of paths.
    pub fn validate(&self, n_paths: usize) -> Result<(), DigiFiError> {
        match &self {
            Self::Custom { f } => { f.validate(n_paths) },
            _ => Ok(())
        }
    }

    /// Generates an array of error values based on the defined error type. The method supports a Weiner process or a custom error generation mechanism.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    /// - `dt`: Time step increment
    ///
    /// # Output
    /// - An array representing the error values for each path
    pub fn get_error(&self, n_paths: usize, dt: f64) -> Result<Array1<f64>, DigiFiError> {
        match &self {
            Self::Weiner { sigma } => {
                let n: Array1<f64> = StandardNormalBoxMuller::new_shuffle(n_paths)?.generate()?;
                Ok(sigma.powi(2) * dt * n)
            },
            Self::Custom { f } => f.error_func(n_paths, dt)
        }
    }
}


#[derive(Debug)]
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
    /// Creates a new `DifferenceStationary` instance.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    /// - `autoregression_params`: Constancts used in-front of the autoregressive stochastic values (e.g., a_{1}X_{1}+ a_{2}X_{2})
    ///
    /// # Errors
    /// - Returns an error if the length of `staring_values` does not match the number of paths.
    /// - Returns an error if an array in `strating_values` has different length to parameters of autoregression.
    pub fn build(n_paths: usize, autoregression_params: Array1<f64>, starting_values: Vec<Array1<f64>>) -> Result<Self, DigiFiError> {
        Self::validate_values(n_paths, &starting_values)?;
        Ok(Self { n_paths, autoregression_params, starting_values })
    }

    /// Validates the array of values.
    ///
    /// # Input
    /// - `n_paths`: Number of paths to generate
    /// - `values`: Previous values of the process
    ///
    /// # Errors
    /// - Returns an error if the length of `values` does not match the number of paths.
    /// - Returns an error if an array in `values` has different length to parameters of autoregression.
    fn validate_values(n_paths: usize, values: &Vec<Array1<f64>>) -> Result<(), DigiFiError> {
        let process_order: usize = values.len();
        // Input validation
        if values.len() != n_paths {
            return Err(DigiFiError::WrongLength { title: Self::error_title(), arg: "previous values".to_owned(), len: n_paths, });
        }
        for v in values {
            if v.len() != process_order {
                return Err(DigiFiError::Other {
                    title: Self::error_title(),
                    details: format!("All arrays of previous values must match the order of process, {}.", process_order),
                });
            }
        }
        Ok(())
    }

    /// Returns the starting values of the process.
    pub fn strating_values(&self) -> Vec<Array1<f64>> {
        self.starting_values.clone()
    }

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
    pub fn get_autoregression(&self, previous_values: &Vec<Array1<f64>>) -> Result<Vec<f64>, DigiFiError> {
        Self::validate_values(self.n_paths, previous_values)?;
        // Autoregression
        let mut result: Vec<f64> = Vec::with_capacity(previous_values.len());
        for process in previous_values {
            result.push(process.dot(&self.autoregression_params));
        }
        Ok(result)
    }
}

impl ErrorTitle for DifferenceStationary {
    fn error_title() -> String {
        String::from("Difference Stationary")
    }
}