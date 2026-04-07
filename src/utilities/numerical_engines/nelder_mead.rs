use std::{cmp::PartialOrd, time::Instant};
use ndarray::{Array, Array1, Array2, s};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::numerical_engines::{VectorFunctionWrapper, VectorNumericalMinimiser, NumericalOptimisationResult};


type Simplex = Vec<(f64, Array1<f64>)>;


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// A minimiser for a vector function using the Nelder-Mead method.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Nelder-Mead_method>
/// - Original Source: <https://doi.org/10.1093/comjnl/8.1.27>
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::error::DigiFiError;
/// use digifi::utilities::{TEST_ACCURACY, VectorNumericalMinimiser, NumericalOptimisationResult, NelderMead};
/// 
/// let func = |x: &Array1<f64>| -> Result<f64, DigiFiError> {
///     Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
/// };
/// let x_0: Array1<f64> = Array1::from_vec(vec![3.0, -8.3]);
/// let result: NumericalOptimisationResult = NelderMead::default().minimise(func.into(), x_0).unwrap();
/// 
/// assert!((result.argmin[0] - 1.0).abs() < 100_000.0 * TEST_ACCURACY);
/// assert!((result.argmin[1] - 1.0).abs() < 100_000.0 * TEST_ACCURACY);
/// ```
pub struct NelderMead {
    /// The maximum number of iterations to optimise.
    pub max_iterations: u64,
    /// The maximum number of function calls used to optimise.
    pub max_fun_calls: u64,
    /// Adapt algorithm parameters to dimensionality of the problem. Useful for high-dimensional minimisation.
    pub adaptive: bool,
    /// Fixed step that is used to generate the initial simplex from the initial guess.
    pub eps: f64,
    /// Error in function parameters between iterations that is acceptable for convergence.
    pub x_tolerance: f64,
    /// Error in function values between iterations that is acceptable for convergence.
    pub f_tolerance: f64,
}

impl NelderMead {
    pub fn new(max_iterations: Option<u64>, max_fun_calls: Option<u64>, adaptive: Option<bool>, eps: Option<f64>, x_tolerance: Option<f64>, f_tolerance: Option<f64>) -> Self {
        Self {
            max_iterations: max_iterations.unwrap_or(1_000),
            max_fun_calls: max_fun_calls.unwrap_or(1_000),
            adaptive: adaptive.unwrap_or(false),
            eps: eps.unwrap_or(0.05),
            x_tolerance: x_tolerance.unwrap_or(1e-4f64),
            f_tolerance: f_tolerance.unwrap_or(1e-4f64),
        }
    }

    pub fn max_iterations(mut self, max_iterations: Option<u64>) -> Self {
        self.max_iterations = max_iterations.unwrap_or(1_000);
        self
    }

    pub fn max_fun_calls(mut self, max_fun_calls: Option<u64>) -> Self {
        self.max_fun_calls = max_fun_calls.unwrap_or(1_000);
        self
    }

    pub fn adaptive(mut self, adaptive: Option<bool>) -> Self {
        self.adaptive = adaptive.unwrap_or(false);
        self
    }

    /// Search for the value minimising `func` given an initial guess in the form of a set of coordinates, the `init_simplex`.
    /// This algorithm only ever explores the space spanned by these initial vectors. If you have parameter restrictions
    /// that effectively place your parameters in a subspace, you can enforce these restrictions by setting `init_simplex`
    /// to a basis of this subspace.
    pub fn minimise_simplex<F: FnMut(&Array1<f64>) -> Result<f64, DigiFiError>>(&self, func: &mut VectorFunctionWrapper<F>, init_simplex: Array2<f64>) -> Result<Array1<f64>, DigiFiError> {
        let mut simplex: Simplex = init_simplex.outer_iter().map(|xi| (func.call(&xi.to_owned()).unwrap(), xi.to_owned())).collect::<Simplex>();
        self.order_simplex(&mut simplex);
        let mut centroid: Array1<f64> = self.centroid(&simplex);
        let n: usize = simplex.len();
        let (alpha, beta, gamma, delta) = self.initialize_parameters(n);
        let mut iterations: u64 = 1;
        while !self.finished(&simplex, iterations, self.max_iterations, func.num, self.max_fun_calls) {
            let f_n1: f64 = simplex[n-1].0;
            let f_n: f64 = simplex[n-2].0;
            let f_0: f64 = simplex[0].0;
            let reflected: Array1<f64> = &centroid + &(alpha * &(&centroid - &simplex[n-1].1));
            let f_reflected: f64 = func.call(&reflected)?;
            if f_reflected < f_n && f_reflected > f_0 {
                // Try reflecting the worst point through the centroid.
                self.lean_update(&mut simplex, &mut centroid, reflected, f_reflected);
            } else if f_reflected < f_0 {
                // Try expanding beyond the centroid.
                let expanded: Array1<f64> = &centroid + &(beta * &(&reflected - &centroid));
                let f_expanded: f64 = func.call(&expanded)?;
                if f_expanded < f_reflected {
                    self.lean_update(&mut simplex, &mut centroid, expanded, f_expanded);
                } else {
                    self.lean_update(&mut simplex, &mut centroid, reflected, f_reflected);
                }
            } else if f_reflected < f_n1 && f_reflected >= f_n {
                // Try a contraction outwards.
                let contracted: Array1<f64> = &centroid + &(gamma * (&centroid - &simplex[n-1].1));
                let f_contracted: f64 = func.call(&contracted)?;
                if f_contracted < f_reflected {
                    self.lean_update(&mut simplex, &mut centroid, contracted, f_contracted);
                } else {
                    // Shrink.
                    self.shrink(&mut simplex, func, delta, &mut centroid)?;
                }
            } else {
                // Try a contraction inwards.
                let contracted: Array1<f64> = &centroid - &(gamma * (&centroid - &simplex[n-1].1));
                let f_contracted: f64 = func.call(&contracted)?;
                if f_contracted < f_reflected {
                    self.lean_update(&mut simplex, &mut centroid, contracted, f_contracted);
                } else {
                    // Shrink.
                    self.shrink(&mut simplex, func, delta, &mut centroid)?;
                }
            }
            iterations += 1;
        }
        Ok(simplex.remove(0).1)
    }

    /// Resolves default values that can only be known after the minimise function is called.
    #[inline]
    fn initialize_parameters(&self, n: usize) -> (f64, f64, f64, f64) {
        if self.adaptive {
            let dim: f64 = n as f64;
            (1.0, 1.0 + 2.0 / dim, 0.75 - 1.0 / (2.0 * dim), 1.0 - 1.0 / dim)
        } else {
            (1.0, 2.0, 0.5, 0.5)
        }
    }

    #[inline]
    fn finished(&self, simplex: &Simplex, iterations: u64, max_iterations: u64, nfeval: u64, max_fun_calls: u64) -> bool {
        let n: usize = simplex.len();
        iterations >= max_iterations
        || nfeval >= max_fun_calls
        || ( simplex[n-1].0 - simplex[0].0 < self.f_tolerance
            && (&simplex[n-1].1 - &simplex[0].1).mapv(f64::abs).sum() < n as f64 * self.x_tolerance )
    }

    /// Update the centroid effiently, knowing only one value changed. The pattern-defeating sort of order_simplex is allready efficient
    /// given that we inserted a single out-of-place value in a sorted vec. This update is O(n).
    #[inline]
    fn lean_update(&self, simplex: &mut Simplex, centroid: &mut Array1<f64>, xnew: Array1<f64>, fnew: f64) -> () {
        let n_minus_one: usize = simplex.len() - 1;
        *centroid += &(&xnew / (n_minus_one as f64));
        simplex[n_minus_one] = (fnew, xnew);
        self.order_simplex(simplex);
        *centroid -= &(&simplex[n_minus_one].1 / (n_minus_one as f64));
    }

    /// Shrink all points towards the best point. Assumes the simplex is ordered. The centroid is updated by shrinking
    /// the centroid directly, then removing the new 'worst x' and adding in the old 'worst x'. This update of `centroid` is O(n). 
    /// Shrinkage requires n function evaluations.
    #[inline]
    fn shrink<F: FnMut(&Array1<f64>) -> Result<f64, DigiFiError>>(&self, simplex: &mut Simplex, f: &mut VectorFunctionWrapper<F>, sigma: f64, centroid: &mut Array1<f64>) -> Result<(), DigiFiError> {
        {
            let mut iter = simplex.iter_mut();
            let (_, x_0) = iter.next()
                .ok_or(DigiFiError::Other {
                    title: Self::error_title(),
                    details: "Could not grab next element from the simplex.".to_owned(),
                })?;
            for (fi, xi) in iter {
                *xi *= sigma;
                *xi += &((1.0 - sigma) * &x_0.view());
                *fi = f.call(xi)?;
            }
        }
        let n_minus_two: usize = simplex.len() - 2;
        let old_worst: Array1<f64> = simplex[n_minus_two].1.to_owned();
        *centroid *= sigma;
        *centroid += &((1.0 - sigma) * &simplex[0].1);
        self.order_simplex(simplex);
        *centroid += &((&old_worst - &simplex[n_minus_two].1) / (n_minus_two as f64));
        Ok(())
    }

    /// Calculate the centroid of all points but the worst one. Assumes that the simplex is ordered. This calculation is O(n^2).
    #[inline]
    fn centroid(&self, simplex: &Simplex) -> Array1<f64> {
        let n_minus_one: usize = simplex.len() - 1;
        let mut centroid: Array1<f64> = Array1::zeros(simplex[0].1.len());
        for (_, xi) in simplex.iter().take(n_minus_one) {
            centroid += xi;
        }
        centroid / (n_minus_one as f64) 
    }

    /// This sorting algorithm should have a runtime of O(n) if only one new element is inserted. After a shrinkage, the runtime is O(n log n).
    #[inline]
    fn order_simplex(&self, simplex: &mut Simplex) -> () {
        simplex.sort_by(|&(fa, _), &(fb, _)| fa.partial_cmp(&fb).unwrap());
    }
}

impl Default for NelderMead {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            max_fun_calls: 1_000,
            adaptive: false,
            eps: 0.05,
            x_tolerance: 1e-4f64,
            f_tolerance: 1e-4f64,
        }
    }
}

impl<F: FnMut(&Array1<f64>) -> Result<f64, DigiFiError>> VectorNumericalMinimiser<F> for NelderMead {
    fn minimise(&self, mut func: VectorFunctionWrapper<F>, x_0: Array1<f64>) -> Result<NumericalOptimisationResult, DigiFiError> {
        let start: Instant = <NelderMead as VectorNumericalMinimiser<F>>::time_start(&self);
        let n: usize = x_0.len();
        let mut init_simplex: Array2<f64> = Array2::default((n+1, n));
        init_simplex.slice_mut(s![0, ..]).assign(&x_0);
        init_simplex.slice_mut(s![1.., ..]).assign(&(Array::eye(n) * self.eps + &x_0 * (1.0 - self.eps)));
        let argmin: Array1<f64> = self.minimise_simplex(&mut func, init_simplex)?;
        let min_value: f64 = func.quick_call(&argmin)?;
        let runtime: f64 = <NelderMead as VectorNumericalMinimiser<F>>::time_end(&self, start);
        Ok(self.minimisation_result(func, argmin, min_value, runtime))
    }
}

impl ErrorTitle for NelderMead {
    fn error_title() -> String {
        String::from("Nelder-Mead Numerical Engine")
    }
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::error::DigiFiError;
    use crate::utilities::{TEST_ACCURACY, numerical_engines::{VectorNumericalMinimiser, NumericalOptimisationResult, NelderMead}};

    #[test]
    fn unit_test_nelder_mead_minimise() {
        let func = |x: &Array1<f64>| -> Result<f64, DigiFiError> {
            Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
        };
        let x_0: Array1<f64> = Array1::from_vec(vec![3.0, -8.3]);
        let result: NumericalOptimisationResult = NelderMead::default().minimise(func.into(), x_0).unwrap();
        assert!((result.argmin[0] - 1.0).abs() < 100_000.0 * TEST_ACCURACY);
        assert!((result.argmin[1] - 1.0).abs() < 100_000.0 * TEST_ACCURACY);
    }

}