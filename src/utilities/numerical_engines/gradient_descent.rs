use std::{ops::Add, time::Instant};
use ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::numerical_engines::{VectorFunctionWrapper, VectorNumericalMinimiser, NumericalOptimisationResult};


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LineSearch {
    /// Backtracking line search evaluating the Armijo rule at each step width.
    /// Armijo used in his paper the values 0.5, 1.0 and 0.5, respectively.
    ArmijoLineSearch { control_parameter: f64, initial_step_width: f64, decay_factor: f64, },
    /// Line search that uses a fixed step width in each iteration instead of performing the actual line search.
    FixedStepWidth { step_width: f64, },
    /// Brute-force line search minimising the objective function over a set of step width candidates, also known as exact line search.
    ExactLineSearch { start_step_width: f64, stop_step_width: f64, increase_factor: f64, },
}

impl LineSearch {
    fn range_err(name: &str, range: &str) -> DigiFiError {
        DigiFiError::ValidationError {
            title: Self::error_title(),
            details: format!("The `{name}` must be in the range `{range}`."),
        }
    }

    /// Validates parameters of line search method.
    /// 
    /// - Armijo Line Search:
    ///     - `control_parameter` ∈ `(0, 1)`
    ///     - `initial_step_width` ∈ `(0, ∞)`
    ///     - `decay_factor` ∈ `(0, 1)`
    /// - Fixed Step Width:
    ///     - `step_width` ∈ `(0, ∞)`
    /// - Exact Line Search
    ///     - `start_step_width` ∈ `(0, ∞)`
    ///     - `stop_step_width` ∈ `(start_step_width, ∞)`
    ///     - `increase_factor` ∈ `(1, ∞)`
    #[inline]
    fn validate_params(&self) -> Result<(), DigiFiError> {
        match self {
            Self::ArmijoLineSearch { control_parameter, initial_step_width, decay_factor } => {
                if (*control_parameter <= 0.0) || (1.0 <= *control_parameter) {
                    return Err(Self::range_err("control_parameter", "(0, 1)"));
                }
                if (*initial_step_width <= 0.0) || initial_step_width.is_infinite() {
                    return Err(Self::range_err("initial_step_width", "(0, ∞)"));
                }
                if (*decay_factor <= 0.0) || (1.0 <= *decay_factor) {
                    return Err(Self::range_err("decay_factor", "(0, 1)"));
                }
            },
            Self::FixedStepWidth { step_width } => {
                if step_width <= &0.0 || step_width.is_infinite() {
                    return Err(Self::range_err("step_width", "(0, ∞)"));
                }
            },
            Self::ExactLineSearch { start_step_width, stop_step_width, increase_factor } => {
                if start_step_width <= &0.0 || start_step_width.is_infinite() {
                    return Err(Self::range_err("start_step_width", "(0, ∞)"));
                }
                if stop_step_width <= start_step_width || stop_step_width.is_infinite() {
                    return Err(Self::range_err("start_step_width", "(start_step_width, ∞)"));
                }
                if increase_factor <= &1.0 || increase_factor.is_infinite() {
                    return Err(Self::range_err("start_step_width", "(1, ∞)"));
                }
            },
        }
        Ok(())
    }

    /// Performs the actual line search given the current `position`, `x` and a `direction` to go to. Returns the new position.
    #[inline]
    fn search<F: FnMut(&Array1<f64>) -> Result<f64, DigiFiError>>(&self, func: &mut VectorFunctionWrapper<F>, x_0: &Array1<f64>, initial_value: f64, gradient: &[f64]) -> Result<Array1<f64>, DigiFiError> {
        match self {
            Self::ArmijoLineSearch { control_parameter, initial_step_width, decay_factor } => {
                let m: f64 = gradient.iter().map(|g| g * -g).fold(0.0, Add::add);
                let t: f64 = -control_parameter * m;
                assert!(t > 0.0);
                let mut step_width: f64 = *initial_step_width;
                loop {
                    let position: Array1<f64> = x_0.iter().zip(gradient.iter()).map(|(x, g)| x + step_width * -g ).collect();
                    let value: f64 = func.call(&position)?;
                    if value <= initial_value - step_width * t {
                        return Ok(position);
                    }
                    step_width *= decay_factor;
                }
            },
            Self::FixedStepWidth { step_width } => {
                Ok(x_0.iter().zip(gradient).map(|(x, d)| x + step_width * d ).collect())
            },
            Self::ExactLineSearch { start_step_width, stop_step_width, increase_factor } => {
                let mut min_position: Array1<f64> = x_0.clone();
                let mut min_value: f64 = initial_value;
                let mut step_width: f64 = *start_step_width;
                loop {
                    let position: Array1<f64> = x_0.iter().zip(gradient).map(|(x, d)| x + step_width * d ).collect();
                    let value: f64 = func.call(&position)?;
                    if value < min_value {
                        min_position = position;
                        min_value = value;
                    }
                    step_width *= increase_factor;
                    if step_width >= *stop_step_width {
                        break;
                    }
                }
                Ok(min_position)
            }
        }
    }
}

impl Default for LineSearch {
    fn default() -> Self {
        Self::ArmijoLineSearch { control_parameter: 0.5, initial_step_width: 1.0, decay_factor: 0.5, }
    }
}

impl ErrorTitle for LineSearch {
    fn error_title() -> String {
        String::from("Line Search")
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// A simple gradient descent optimiser.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Gradient_descent>
/// - Original Source: <https://doi.org/10.4171%2Fdms%2F6%2F27>
pub struct GradientDescent {
    /// The maximum number of iterations to optimise.
    pub max_iterations: u64,
    /// The maximum number of function calls used to optimise.
    pub max_fun_calls: u64,
    /// Tolerance of the gradient that determines whether the flat area in the gradient has been reached (i.e., saddle point).
    pub gradient_tolerance: f64,
    /// Type of line search to perform.
    pub line_search: LineSearch,
}

impl GradientDescent {
    /// Creates a new `GradientDescent` optimiser using the default values.
    pub fn build(max_iterations: Option<u64>, max_fun_calls: Option<u64>, gradient_tolerance: Option<f64>, line_search: &LineSearch) -> Result<Self, DigiFiError> {
        let max_iterations: u64 = match max_iterations { Some(v) => v, None => 100 };
        let max_fun_calls: u64 = match max_fun_calls { Some(v) => v, None => 100 };
        let gradient_tolerance: f64 = match gradient_tolerance { Some(v) => v, None => 1e-4f64 };
        line_search.validate_params()?;
        Ok(Self { max_iterations, max_fun_calls, gradient_tolerance, line_search: *line_search, })
    }

    #[inline]
    fn minimise_wrapper<F: FnMut(&Array1<f64>) -> Result<f64, DigiFiError>>(&self, func: &mut VectorFunctionWrapper<F>, mut position: Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let mut iteration: u64 = 0;
        loop {
            let (gradient, current_value) = func.gradient(&position)?;
            // Return the point if the gradient is too small.
            if func.is_saddle_point(&position, self.gradient_tolerance) {
                return Ok(position);
            }
            position = self.line_search.search(func, &position, current_value, &gradient)?;
            iteration += 1;
            // Stop optimisation if reached maximum number of iterations or function calls.
            if (iteration >= self.max_iterations) || (func.num >= self.max_fun_calls) {
                return Ok(position);
            }
        }
    }
}

impl Default for GradientDescent {
    fn default() -> Self {
        Self { max_iterations: 1_000, max_fun_calls: 1_000, gradient_tolerance: 1e-4f64, line_search: LineSearch::default(), }
    }
}

impl<F: FnMut(&Array1<f64>) -> Result<f64, DigiFiError>> VectorNumericalMinimiser<F> for GradientDescent {
    fn minimise(&self, mut func: VectorFunctionWrapper<F>, x_0: Array1<f64>) -> Result<NumericalOptimisationResult, DigiFiError> {
        let start: Instant = <GradientDescent as VectorNumericalMinimiser<F>>::time_start(&self);
        let argmin: Array1<f64> = self.minimise_wrapper(&mut func, x_0)?;
        let min_value: f64 = func.quick_call(&argmin)?;
        let runtime: f64 = <GradientDescent as VectorNumericalMinimiser<F>>::time_end(&self, start);
        Ok(self.minimisation_result(func, argmin, min_value, runtime))
    }
}