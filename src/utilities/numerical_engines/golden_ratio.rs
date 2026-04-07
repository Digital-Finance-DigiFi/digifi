use std::time::Instant;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;
use crate::consts::GR_RATIO;
use crate::utilities::numerical_engines::{ScalarFunctionWrapper, ScalarNumericalMinimiser, NumericalOptimisationResult};


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Minimisation technique for scalar function that searches for a local minimum in a given interval.
/// At every iteration, the interval is decreased in size by a constant factor, until the desired precision is obtained.
///
/// This algorithm is guaranteed to converge on a local minimum in a finite amount of steps under very light
/// smoothess criteria for the target function.
/// 
/// Note: The number of function evaluations in a bracket search is `2 + number_of_iterations`.
/// 
/// # Links
/// - Wiipedia: <https://en.wikipedia.org/wiki/Golden-section_search>
/// - Original Source: <https://doi.org/10.1090/S0002-9939-1953-0055639-3>
/// 
/// # Examples
/// 
/// 1. Minimisation of a function in a certain range
/// 
/// ```rust
/// use digifi::error::DigiFiError;
/// use digifi::utilities::{TEST_ACCURACY, ScalarNumericalMinimiser, GoldenRatio};
/// 
/// let minimiser: GoldenRatio = GoldenRatio::new(Some(1_000), Some(1.0e-7));
/// let func = |x: f64| -> Result<f64, DigiFiError> { Ok((x - 0.2).powi(2)) };
/// let res: f64 = minimiser.minimise_bracket(&mut func.into(), -1.0, 1.0).unwrap();
/// 
/// assert!((res - 0.2).abs() <= TEST_ACCURACY * 100.0);
/// ```
/// 
/// 2. Minimisation using a starting point
/// 
/// ```rust
/// use digifi::error::DigiFiError;
/// use digifi::utilities::{TEST_ACCURACY, ScalarNumericalMinimiser, NumericalOptimisationResult, GoldenRatio};
/// 
/// let minimiser: GoldenRatio = GoldenRatio::new(Some(1_000), Some(1.0e-7));
/// let func = |x: f64| -> Result<f64, DigiFiError> { Ok((x - 0.2).powi(2)) };
/// let res: NumericalOptimisationResult = minimiser.minimise(func.into(), 10.0).unwrap();
/// 
/// assert!((res.argmin[0] - 0.2).abs() <= TEST_ACCURACY * 100.0);
/// ```
/// 
/// 3. Minimisation with an extreme condition
/// 
/// ```rust
/// use digifi::error::DigiFiError;
/// use digifi::utilities::{ScalarNumericalMinimiser, NumericalOptimisationResult, GoldenRatio};
/// 
/// let minimiser: GoldenRatio = GoldenRatio::new(Some(1_000), Some(1.0e-7));
/// let func = |x: f64| -> Result<f64, DigiFiError> { Ok(x) };
/// let res: NumericalOptimisationResult = minimiser.minimise(func.into(), 10.0).unwrap();
/// 
/// assert!(res.argmin[0].is_infinite());
/// ```
pub struct GoldenRatio {
    /// The maximum number of iterations before the search terminates
    pub max_iterations: u64,
    /// The width of the interval at which convergence is satisfactory
    pub x_tolerance: f64,
}

impl GoldenRatio {
    pub fn new(max_iterations: Option<u64>, x_tolerance: Option<f64>) -> Self {
        Self {
            max_iterations: max_iterations.unwrap_or(1_000),
            x_tolerance: x_tolerance.unwrap_or(1.0e-8),
        }
    }

    pub fn max_iterations(mut self, max_iterations: u64) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Increase the step until `func` stops decreasing, then return the step.
    fn explore<F: FnMut(f64) -> Result<f64, DigiFiError>>(&self, func: &mut ScalarFunctionWrapper<F>, x_0: f64, explore_left: bool) -> Result<f64, DigiFiError> {
        let mut step: f64 = if explore_left { -1.0 } else { 1.0 }
            * 2.0_f64.powi((f64::EPSILON + x_0.abs()).log2() as i32)
            * f64::EPSILON;
        let mut f_prev: f64 = func.call(x_0)?;
        let mut f_stepped: f64 = func.call(x_0 + step)?;
        while f_stepped < f_prev {
            step *= 2.0;
            f_prev = f_stepped;
            f_stepped = func.call(x_0 + step)?;
        }
        Ok(step)
    }

    /// Search for the minimum of `func` between `left` and `right`.
    pub fn minimise_bracket<F: FnMut(f64) -> Result<f64, DigiFiError>>(&self, func: &mut ScalarFunctionWrapper<F>, left: f64, right: f64) -> Result<f64, DigiFiError> {
        let mut min: f64 = left.max(f64::MIN);
        let mut max: f64 = right.min(f64::MAX);
        let mut iter: u64 = 0;
        let mut x_b: f64 = min + (max - min) / GR_RATIO;
        let mut x_c: f64 = max - (max - min) / GR_RATIO;
        let mut f_b: f64 = func.call(x_b)?;
        let mut f_c: f64 = func.call(x_c)?;
        while (max - min).abs() > self.x_tolerance && iter < self.max_iterations {
            iter += 1;
            if f_b < f_c {
                max = x_c;
                x_c = x_b;
                x_b = min + (max - min) / GR_RATIO;
                f_c = f_b;
                f_b = func.call(x_b)?;
            } else {
                min = x_b;
                x_b = x_c;
                x_c = max - (max - min) / GR_RATIO;
                f_b = f_c;
                f_c = func.call(x_c)?;
            }
        }
        Ok((min + max) / 2.0)
    }
}

impl Default for GoldenRatio {
    fn default() -> Self {
        Self { max_iterations: 1_000, x_tolerance: 1.0e-8, }
    }
}

impl<F: FnMut(f64) -> Result<f64, DigiFiError>> ScalarNumericalMinimiser<F> for GoldenRatio {
    /// Search for the argmin minimising `func` given an initial guess `x_0`.
    ///
    /// Currently `minimise` makes, in some specific cases, at most  around `2_000` additional calls to `func` to find
    /// a bracketing interval.
    /// 
    /// # Input
    /// - `func`: Scalar function being minimised
    /// - `x_0`: Initial guess for the numerical method to start at
    /// 
    /// # Output
    /// - Result of the numerical optimisation
    fn minimise(&self, mut func: ScalarFunctionWrapper<F>, x_0: f64) -> Result<NumericalOptimisationResult, DigiFiError> {
        let start: Instant = <GoldenRatio as ScalarNumericalMinimiser<F>>::time_start(&self);
        let left: f64 = x_0 + self.explore(&mut func, x_0, true)?;
        let right: f64 = x_0 + self.explore(&mut func, x_0, false)?;
        let argmin: f64 = self.minimise_bracket(&mut func, left, right)?;
        let min_value: f64 = func.quick_call(argmin)?;
        let runtime: f64 = <GoldenRatio as ScalarNumericalMinimiser<F>>::time_end(&self, start);
        Ok(self.minimisation_result(func, argmin, min_value, runtime))
    }
}


#[cfg(test)]
mod tests {
    use crate::error::DigiFiError;
    use crate::utilities::{TEST_ACCURACY, ScalarNumericalMinimiser, NumericalOptimisationResult, GoldenRatio};

    #[test]
    fn unit_test_golden_ratio_1() {
        let minimiser: GoldenRatio = GoldenRatio::new(Some(1_000), Some(1.0e-7));
        let func = |x: f64| -> Result<f64, DigiFiError> { Ok((x - 0.2).powi(2)) };
        let res: f64 = minimiser.minimise_bracket(&mut func.into(), -1.0, 1.0).unwrap();
        assert!((res - 0.2).abs() <= TEST_ACCURACY * 100.0);
    }

    #[test]
    fn unit_test_golden_ratio_2() {
        let minimiser: GoldenRatio = GoldenRatio::new(Some(1_000), Some(1.0e-7));
        let func = |x: f64| -> Result<f64, DigiFiError> { Ok((x - 0.2).powi(2)) };
        let res: NumericalOptimisationResult = minimiser.minimise(func.into(), 10.0).unwrap();
        assert!((res.argmin[0] - 0.2).abs() <= TEST_ACCURACY * 100.0);
    }

    #[test]
    fn unit_test_golden_ratio_3() {
        let minimiser: GoldenRatio = GoldenRatio::new(Some(1_000), Some(1.0e-7));
        let func = |x: f64| -> Result<f64, DigiFiError> { Ok(x) };
        let res: NumericalOptimisationResult = minimiser.minimise(func.into(), 10.0).unwrap();
        assert!(res.argmin[0].is_infinite());
    }
}