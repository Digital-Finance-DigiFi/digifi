use std::{collections::VecDeque, time::Instant};
use ndarray::{Array1};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;
use crate::utilities::numerical_engines::{
    VectorFunctionWrapper, VectorNumericalMinimiser, NumericalOptimisationResult, GoldenRatio,
};


/// Limited-memory BFGS Quasi-Newton optimiser. This implementation uses the two-loop recursion to calculate
/// the quasi-inverse-hessian. This optimiser is known to be particularly suited for optimising high-dimensional convex functions.
/// 
/// Note: The gradient is evaluated once per iteration.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Limited-memory_BFGS>
/// - Original Source: <https://doi.org/10.2307/2006193>
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::error::DigiFiError;
/// use digifi::utilities::{TEST_ACCURACY, VectorNumericalMinimiser, NumericalOptimisationResult, LBFGS};
/// 
/// let center: Array1<f64> = arr1(&[0.9, 1.3, 0.5]);
/// let func = |x: &Array1<f64>| -> Result<f64, DigiFiError> {
///     Ok((x - &center).mapv(|xi| -(-xi * xi).exp()).sum())
/// };
/// let x_0: Array1<f64> = Array1::ones(center.len());
/// let result: NumericalOptimisationResult = LBFGS::default().minimise(func.into(), x_0).unwrap();
/// 
/// let within_accuracy: bool = (result.argmin - center).fold(true, |all, v| { all && v.abs() < 10.0 * TEST_ACCURACY } );
/// assert!(within_accuracy)
/// ```
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LBFGS {
    /// The maximum number of iterations
    pub max_iterations: u64,
    /// The maximum number of function evaluations
    pub max_fun_calls: u64,
    /// The number of data points to use to estimate the inverse Hessian.
    /// 
    /// Nopte: If `m` is larger than `x_0.len()`, then `x_0.len()` is used
    pub m: usize,
    /// The maximum step to be taken in the direction determined by the Quasi-Newton method
    pub max_step: f64,
    /// The tolerance on x used to terminate the line search
    pub x_tolerance: f64,
    /// The tolerance on the gradient used to terminate the optimisation 
    pub  g_tolerance: f64,
}

impl LBFGS {
    pub fn new(
        max_iterations: Option<u64>, max_fun_calls: Option<u64>, m: Option<usize>, max_step: Option<f64>, x_tolerance: Option<f64>, g_tolerance: Option<f64>,
    ) -> Self {
        Self {
            max_iterations: max_iterations.unwrap_or(1_500),
            max_fun_calls: max_fun_calls.unwrap_or(1_500),
            m: m.unwrap_or(5),
            max_step: max_step.unwrap_or(2.0),
            x_tolerance: x_tolerance.unwrap_or(1e-8),
            g_tolerance: g_tolerance.unwrap_or(1e-12),
        }
    }

    /// Minimise `func` starting in `x_0` using `grad` as the gradient of `func`.
    fn minimise_wrapper<F: FnMut(&Array1<f64>) -> Result<f64, DigiFiError>>(&self, func: &mut VectorFunctionWrapper<F>, x_0: Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let mut iterations: u64 = 1;
        let mut nfeval: u64 = 0;
        let m: usize = x_0.len().min(self.m).max(1);
        let mut hist: VecDeque<(Array1<f64>, Array1<f64>, f64)> = VecDeque::with_capacity(m);
        let mut x: Array1<f64> = x_0.to_owned();
        let mut g: Array1<f64> = Array1::from_vec(func.gradient(&x)?.0);
        loop {
            let dir: Array1<f64> = self.quasi_update(&g, &hist);
            let a: f64 = {
                let func = |a: f64| -> Result<f64, DigiFiError> {
                    nfeval += 1;
                    func.call(&(&x + &(a * &dir)))
                };
                GoldenRatio::default().minimise_bracket(&mut func.into(), -self.max_step, 0.0)?
            };
            let x_new: Array1<f64> = &x + &(a * &dir);
            let g_new: Array1<f64> = Array1::from_vec(func.gradient(&x_new)?.0);
            let s: Array1<f64> = &x_new - &x;
            let y: Array1<f64> = &g_new - &g;
            let r: f64 = 1.0_f64 / s.dot(&y);
            if self.finished(r, iterations, nfeval, &g_new) {
                break;
            }
            while hist.len() >= m {
                hist.pop_front();
            }
            hist.push_back((s, y, r));
            x = x_new;
            g = g_new;
            iterations += 1;
        }
        Ok(x)
    }

    #[inline]
    fn finished(&self, r: f64, iterations: u64, nfeval: u64, g_new: &Array1<f64>) -> bool {
        r.is_nan()
        || iterations > self.max_iterations
        || nfeval > self.max_fun_calls
        || g_new.mapv(f64::abs).sum() < self.g_tolerance
    }

    #[inline]
    /// Calculate the Quasi-Newton direction `H*g` efficiently, where `g` is the gradient and `H` is the inverse Hessian.
    fn quasi_update(&self, grad: &Array1<f64>, hist: &VecDeque<(Array1<f64>, Array1<f64>, f64)>) -> Array1<f64> {
        let mut q: Array1<f64> = grad.clone();
        let mut a: Vec<f64> = Vec::with_capacity(hist.len());
        for (si, yi, ri) in hist.iter().rev() {
            let ai: f64 = ri * si.dot(&q);
            q.scaled_add(-ai, &yi);
            a.push(ai);
        }
        for ((si, yi, ri), ai) in hist.iter().zip(a.iter().rev()) {
            let bi: f64 = ri * yi.dot(&q);
            q.scaled_add(ai - bi, &si);
        }
        q
    }
}

impl Default for LBFGS {
    fn default() -> Self {
        Self {
            max_iterations: 1_500,
            max_fun_calls: 1_500,
            m: 5,
            max_step: 2.0,
            x_tolerance: 1e-8,
            g_tolerance: 1e-12,
        }
    }
}

impl<F: FnMut(&Array1<f64>) -> Result<f64, DigiFiError>> VectorNumericalMinimiser<F> for LBFGS {
    fn minimise(&self, mut func: VectorFunctionWrapper<F>, x_0: Array1<f64>) -> Result<NumericalOptimisationResult, DigiFiError> {
        let start: Instant = <LBFGS as VectorNumericalMinimiser<F>>::time_start(&self);
        let argmin: Array1<f64> = self.minimise_wrapper(&mut func, x_0)?;
        let min_value: f64 = func.quick_call(&argmin)?;
        let runtime: f64 = <LBFGS as VectorNumericalMinimiser<F>>::time_end(&self, start);
        Ok(self.minimisation_result(func, argmin, min_value, runtime))
    }
}


#[cfg(test)]
mod test {
    use ndarray::{Array1, arr1};
    use crate::error::DigiFiError;
    use crate::utilities::TEST_ACCURACY;
    use crate::utilities::numerical_engines::{VectorNumericalMinimiser, NumericalOptimisationResult, LBFGS};

    #[test]
    fn unit_test_lbfgs_minimise() {
        let center: Array1<f64> = arr1(&[0.9, 1.3, 0.5]);
        let func = |x: &Array1<f64>| -> Result<f64, DigiFiError> {
            Ok((x - &center).mapv(|xi| -(-xi * xi).exp()).sum())
        };
        let x_0: Array1<f64> = Array1::ones(center.len());
        let result: NumericalOptimisationResult = LBFGS::default().minimise(func.into(), x_0).unwrap();
        let within_accuracy: bool = (result.argmin - center).fold(true, |all, v| { all && v.abs() < 10.0 * TEST_ACCURACY } );
        assert!(within_accuracy)
    }
}