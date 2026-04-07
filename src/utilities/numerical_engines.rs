// Re-Exports
pub use self::golden_ratio::GoldenRatio;
pub use self::gradient_descent::{LineSearch, GradientDescent};
pub use self::l_bfgs::LBFGS;
pub use self::nelder_mead::NelderMead;


mod golden_ratio;
mod gradient_descent;
mod l_bfgs;
mod nelder_mead;


use std::{convert::From, fmt::Display, time::Instant};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use ndarray::{Array1, arr1};
use crate::error::DigiFiError;
use crate::consts::H;
use crate::utilities::LARGE_TEXT_BREAK;


#[derive(Clone, Debug)]
/// Wrapper for a scalar function that is optimised by a numerical engine.
pub struct ScalarFunctionWrapper<F: FnMut(f64) -> Result<f64, DigiFiError>> {
    /// Number of function calls.
    pub num: u64,
    /// Objective function being minimsed.
    pub func: F,
    /// List of values at every call of the wrapped function.
    pub values: Vec<f64>,
    /// Whether to save the parameters at every call of the wrapped function.
    /// 
    /// Note: For complicated wrapped functions, setting this to `true` may significantly reduce performance.
    pub save_params: bool,
    /// List of parameters at every call of the wrapped function.
    pub params: Vec<f64>,
}

impl<F: FnMut(f64) -> Result<f64, DigiFiError>> ScalarFunctionWrapper<F> {
    pub fn new(func: F, save_params: bool) -> Self {
        Self { num: 0, func, values: Vec::new(), save_params, params: Vec::new(), }
    }

    /// Makes a wrapped function call with the provided argument.
    /// 
    /// # Input
    /// -`x`: Argument that is provided to the wrapped function
    /// 
    /// # Output
    /// - Value of the wrapped function with the provided `x`.
    pub fn call(&mut self, x: f64) -> Result<f64, DigiFiError> {
        // Save params
        if self.save_params {
            self.params.push(x);
        }
        // Function evaluation
        self.num += 1;
        let value: f64 = (self.func)(x)?;
        self.values.push(value);
        Ok(value)
    }

    /// Function call that does not update the state of the wrapper.
    /// 
    /// Note: This is used only as a utility.
    fn quick_call(&mut self, x: f64) -> Result<f64, DigiFiError> {
        (self.func)(x)
    }
}

impl<F: FnMut(f64) -> Result<f64, DigiFiError>> From<F> for ScalarFunctionWrapper<F> {
    fn from(value: F) -> Self {
        Self::new(value, false)
    }
}


#[derive(Clone, Debug)]
/// Wrapper for a vector function that is optimised by a numerical engine.
pub struct VectorFunctionWrapper<F: FnMut(&Array1<f64>) -> Result<f64, DigiFiError>> {
    /// Number of function calls.
    pub num: u64,
    /// Objective function being minimised.
    pub func: F,
    /// List of values at every call of the wrapped function.
    pub values: Vec<f64>,
    /// Whether to save the parameters at every call of the wrapped function.
    /// 
    /// Note: For complicated wrapped functions, setting this to `true` may significantly reduce performance.
    pub save_params: bool,
    /// List of parameters at every call of the wrapped function.
    pub params: Vec<Array1<f64>>,
}

impl<F: FnMut(&Array1<f64>) -> Result<f64, DigiFiError>> VectorFunctionWrapper<F> {
    pub fn new(func: F, save_params: bool) -> Self {
        Self { num: 0, func, values: Vec::new(), save_params, params: Vec::new(), }
    }

    #[inline]
    /// Checks if the value is finite, and if it isn't throws an `DigiFiError`.
    /// 
    /// # Input
    /// - `v`: Variable that must be finite
    /// - `name`: Name of the variable that will be included in error message
    /// 
    /// # Errors
    /// - Returns an error if the variable `v` is not finite
    fn check_finite(v: f64, name: &str) -> Result<(), DigiFiError> {
        if !v.is_finite() {
            return Err(DigiFiError::ParameterConstraint {
                title: "Vector Function Wrapper".to_owned(),
                constraint: format!("Tha parameter `{name}` is not finite."),
            });
        }
        Ok(())
    }

    /// Makes a wrapped function call with the provided arguments.
    /// 
    /// # Input
    /// -`x`: Arguments that are provided to the wrapped function
    /// 
    /// # Output
    /// - Value of the wrapped function with the provided `x`.
    pub fn call(&mut self, x: &Array1<f64>) -> Result<f64, DigiFiError> {
        // Save params
        if self.save_params {
            self.params.push(x.clone());
        }
        // Function evaluation
        self.num += 1;
        let value: f64 = (self.func)(x)?;
        self.values.push(value);
        Ok(value)
    }

    /// Function call that does not update the state of the wrapper.
    /// 
    /// Note: This is used only as a utility.
    fn quick_call(&mut self, x: &Array1<f64>) -> Result<f64, DigiFiError> {
        (self.func)(x)
    }

    #[inline]
    /// Computes the gradient of the objective function at a given position `x` (i.e., `∀ᵢ ∂/∂xᵢ f(x) = ∇f(x)`).
    /// 
    /// # Input
    /// - `x` Arguments that are provided to the wrapped function
    /// 
    /// # Output
    /// - Gradient of the wrapped function at point `x`, and the value of the function at the current point
    /// 
    /// # Errors
    /// - Returns an error if the infinitesimal increment between consequtive points is not finite.
    /// - Returns an error if the gradient is not finite.
    pub fn gradient(&mut self, x: &Array1<f64>) -> Result<(Vec<f64>, f64), DigiFiError> {
        let mut x_: Array1<f64> = x.clone();
        let current: f64 = self.call(&x)?;
        let mut gradient: Vec<f64> = Vec::with_capacity(x_.len());
        for (i, x_i) in x.iter().enumerate() {
            let h: f64 = if x_i == &0.0_f64 { H } else { (std::f64::EPSILON * x_i.abs()).sqrt() };
            Self::check_finite(h, "h")?;
            x_[i] += h;
            let forward: f64 = self.call(&x_)?;
            x_[i] -= h;
            let d_i: f64 = (forward - current) / h;
            Self::check_finite(d_i, "dy/dx")?;
            gradient.push(d_i);
        }
        Ok((gradient, current))
    }

    #[inline]
    /// Tests whether a flat area is reached (i.e., tests if all absolute gradient component lie within the `tolerance`).
    /// 
    /// # Input
    /// - `gradient`: Gradient of the wrapped function at a specific point
    /// - `tolerance`: Maximum allowed deviation of gradient from `0`, for the point to be a saddle point
    pub fn is_saddle_point(&self, gradient: &Array1<f64>, tolerance: f64) -> bool {
        gradient.iter().all(|dx| dx.abs() <= tolerance)
    }
}

impl<F: FnMut(&Array1<f64>) -> Result<f64, DigiFiError>> From<F> for VectorFunctionWrapper<F> {
    fn from(value: F) -> Self {
        Self::new(value, false)
    }
}


pub trait ScalarNumericalMinimiser<F: FnMut(f64) -> Result<f64, DigiFiError>> {
    fn time_start(&self) -> Instant {
        Instant::now()
    }

    fn time_end(&self, start: Instant) -> f64 {
        start.elapsed().as_secs_f64()
    }

    fn minimise(&self, func: ScalarFunctionWrapper<F>, x_0: f64) -> Result<NumericalOptimisationResult, DigiFiError>;

    fn minimisation_result(&self, func: ScalarFunctionWrapper<F>, argmin: f64, min_value: f64, runtime: f64) -> NumericalOptimisationResult {
        let mut params: Option<Vec<Array1<f64>>> = None;
        if func.save_params {
            params = Some(func.params.iter().map(|v| arr1(&[*v]) ).collect::<Vec<Array1<f64>>>());
        }
        NumericalOptimisationResult {
            values: func.values,
            params,
            argmin: arr1(&[argmin]),
            min_value,
            runtime,
        }
    }
}


pub trait VectorNumericalMinimiser<F: FnMut(&Array1<f64>) -> Result<f64, DigiFiError>> {
    fn time_start(&self) -> Instant {
        Instant::now()
    }

    fn time_end(&self, start: Instant) -> f64 {
        start.elapsed().as_secs_f64()
    }

    /// Search for the argmin minimising `func` given an initial guess `x_0`.
    /// 
    /// # Input
    /// - `func`: Vector function being minimised
    /// - `x_0`: Initial guess for the numerical method to start at
    /// 
    /// # Output
    /// - Result of the numerical optimisation
    fn minimise(&self, func: VectorFunctionWrapper<F>, x_0: Array1<f64>) -> Result<NumericalOptimisationResult, DigiFiError>;

    fn minimisation_result(&self, func: VectorFunctionWrapper<F>, argmin: Array1<f64>, min_value: f64, runtime: f64) -> NumericalOptimisationResult {
        let mut params: Option<Vec<Array1<f64>>> = None;
        if func.save_params {
            params = Some(func.params);
        }
        NumericalOptimisationResult {
            values: func.values,
            params,
            argmin,
            min_value,
            runtime,
        }
    }
}


#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NumericalOptimisationResult {
    /// List of values at every call of the wrapped function.
    pub values: Vec<f64>,
    /// List of parameters at every call of the wrapped function.
    pub params: Option<Vec<Array1<f64>>>,
    /// Parameters that produce the minimum value.
    pub argmin: Array1<f64>,
    /// Minimum value of the wrpped function.
    pub min_value: f64,
    /// Runtime of the optimisation denominated in seconds.
    pub runtime: f64,
}

impl Display for NumericalOptimisationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let argmin: String = self.argmin.iter().map(|v| v.to_string() ).collect::<Vec<String>>().join(", ");
        let result: String = LARGE_TEXT_BREAK.to_owned()
            + "Numerical Optimisation Result\n"
            + LARGE_TEXT_BREAK
            + &format!("Minimisation Parameters (Argmin): {argmin}\n")
            + &format!("Minimised Function Value: {}\n", self.min_value)
            + &format!("Runtime: {} Seconds\n", self.runtime)
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}