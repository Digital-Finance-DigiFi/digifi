use std::cmp::PartialOrd;
use ndarray::{Array, Array1, Array2, s};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;


type Simplex = Vec<(f64, Array1<f64>)>;


#[derive(Debug)]
struct WrappedFunction<F: FnMut(&Array1<f64>) -> f64> {
    num: u64,
    func: F,
}

impl<F: FnMut(&Array1<f64>) -> f64> WrappedFunction<F> {
    fn call(&mut self, arg: &Array1<f64>) -> f64 {
        self.num += 1;
        (self.func)(arg)
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// A minimizer for a scalar function of one or more variables using the Nelder-Mead algorithm.
struct NelderMead {
    /// The maximum number of iterations to optimize.
    max_iterations: u64,
    /// The maximum number of function calls used to optimize.
    max_fun_calls: u64,
    /// Adapt algorithm parameters to dimensionality of the problem. Useful for high-dimensional minimization.
    adaptive: bool,
    /// Absolute error in function parameters between iterations that is acceptable for convergence.
    x_tolerance: f64,
    /// Absolute error in function values between iterations that is acceptable for convergence.
    f_tolerance: f64,
}

impl NelderMead {
    fn new(max_iterations: Option<u64>, max_fun_calls: Option<u64>, adaptive: Option<bool>, x_tolerance: Option<f64>, f_tolerance: Option<f64>) -> Self {
        let max_iterations: u64 = match max_iterations { Some(v) => v, None => 100 };
        let max_fun_calls: u64 = match max_fun_calls { Some(v) => v, None => 100 };
        let adaptive: bool = match adaptive { Some(v) => v, None => false };
        let x_tolerance: f64 = match x_tolerance { Some(v) => v, None => 1e-4f64 };
        let f_tolerance: f64 = match f_tolerance { Some(v) => v, None => 1e-4f64 };
        NelderMead {max_iterations, max_fun_calls, adaptive, x_tolerance, f_tolerance}
    }

    /// Search for the value minimizing `func` given an initial guess in the form of a point. The algorithm will
    /// explore the variable space without constraints.
    fn minimize<F>(&self, func: F, x0: Array1<f64>) -> Result<Array1<f64>, DigiFiError>
    where F: FnMut(&Array1<f64>) -> f64 {
        let n: usize = x0.len();
        let eps: f64 = 0.05;
        let mut init_simplex: Array2<f64> = Array2::default((n+1, n));
        init_simplex.slice_mut(s![0, ..]).assign(&x0);
        init_simplex.slice_mut(s![1.., ..]).assign(&(Array::eye(n) * eps + &x0 * (1.0-eps)));
        self.minimize_simplex(func, init_simplex)
    }

    /// Search for the value minimizing `func` given an initial guess in the form of a set of coordinates, the `init_simplex`.
    /// This algorithm only ever explores the space spanned by these initial vectors. If you have parameter restrictions
    /// that effectively place your parameters in a subspace, you can enforce these restrictions by setting `init_simplex`
    /// to a basis of this subspace.
    fn minimize_simplex<F>(&self, func: F, init_simplex: Array2<f64>) -> Result<Array1<f64>, DigiFiError>
    where F: FnMut(&Array1<f64>) -> f64 {
        let mut func: WrappedFunction<F> = WrappedFunction { num: 0, func: func };
        let mut simplex: Simplex = init_simplex.outer_iter().map(|xi| (func.call(&xi.to_owned()), xi.to_owned())).collect::<Simplex>();
        self.order_simplex(&mut simplex);
        let mut centroid: Array1<f64> = self.centroid(&simplex);
        let n: usize = simplex.len();
        let (max_iterations, max_fun_calls, alpha, beta, gamma, delta) = self.initialize_parameters(n);
        let mut iterations: u64 = 1;
        while !self.finished(&simplex, iterations, max_iterations, func.num, max_fun_calls) {
            let f_n1: f64 = simplex[n-1].0;
            let f_n: f64 = simplex[n-2].0;
            let f_0: f64 = simplex[0].0;
            let reflected: Array1<f64> = &centroid + &(alpha * &(&centroid - &simplex[n-1].1));
            let f_reflected: f64 = func.call(&reflected);
            if f_reflected < f_n && f_reflected > f_0 {
                // Try reflecting the worst point through the centroid.
                self.lean_update(&mut simplex, &mut centroid, reflected, f_reflected);
            } else if f_reflected < f_0 {
                // Try expanding beyond the centroid.
                let expanded: Array1<f64> = &centroid + &(beta * &(&reflected - &centroid));
                let f_expanded: f64 = func.call(&expanded);
                if f_expanded < f_reflected {
                    self.lean_update(&mut simplex, &mut centroid, expanded, f_expanded);
                } else {
                    self.lean_update(&mut simplex, &mut centroid, reflected, f_reflected);
                }
            } else if f_reflected < f_n1 && f_reflected >= f_n {
                // Try a contraction outwards.
                let contracted: Array1<f64> = &centroid + &(gamma * (&centroid - &simplex[n-1].1));
                let f_contracted = func.call(&contracted);
                if f_contracted < f_reflected {
                    self.lean_update(&mut simplex, &mut centroid, contracted, f_contracted);
                } else {
                    // Shrink.
                    self.shrink(&mut simplex, &mut func, delta, &mut centroid)?;
                }
            } else {
                // Try a contraction inwards.
                let contracted: Array1<f64> = &centroid - &(gamma * (&centroid - &simplex[n-1].1));
                let f_contracted: f64 = func.call(&contracted);

                if f_contracted < f_reflected {
                    self.lean_update(&mut simplex, &mut centroid, contracted, f_contracted);
                } else {
                    // Shrink.
                    self.shrink(&mut simplex, &mut func, delta, &mut centroid)?;
                }
            }
            iterations += 1;
        }
        Ok(simplex.remove(0).1)
    }

    /// Resolves default values that can only be known after the minimize function is called.
    #[inline]
    fn initialize_parameters(&self, n: usize) -> (u64, u64, f64, f64, f64, f64) {
        let max_iterations: u64 = self.max_iterations;
        let max_fun_calls: u64 = self.max_fun_calls;
        let (alpha, beta, gamma, delta): (f64, f64, f64, f64) = match self.adaptive {
            true => {
                let dim: f64 = n as f64;
                (1.0, 1.0 + 2.0 / dim, 0.75 - 1.0 / (2.0 * dim), 1.0 - 1.0 / dim)
            },
            false => (1.0, 2.0, 0.5, 0.5),
        };
        (max_iterations, max_fun_calls, alpha, beta, gamma, delta)
    }

    #[inline]
    fn finished(&self, simplex: &Simplex, iterations: u64, max_iterations: u64, nfeval: u64, max_fun_calls: u64) -> bool {
        let n: usize = simplex.len();
        iterations > max_iterations 
        || nfeval > max_fun_calls 
        || ( simplex[n-1].0 - simplex[0].0 < self.f_tolerance 
             && (&simplex[n-1].1 - &simplex[0].1).mapv(f64::abs).sum() < n as f64 * self.x_tolerance )
    }

    /// Update the centroid effiently, knowing only one value changed. The pattern-defeating sort of order_simplex is allready efficient
    /// given that we inserted a single out-of-place value in a sorted vec. This update is O(n).
    #[inline]
    fn lean_update(&self, simplex: &mut Simplex, centroid: &mut Array1<f64>, xnew: Array1<f64>, fnew: f64) -> () {
        let n: usize = simplex.len();
        *centroid += &(&xnew / (n-1) as f64);
        simplex[n-1] = (fnew, xnew);
        self.order_simplex(simplex);
        *centroid -= &(&simplex[n-1].1 / (n-1) as f64);
    }

    /// Shrink all points towards the best point. Assumes the simplex is ordered. The centroid is updated by shrinking
    /// the centroid directly, then removing the new 'worst x' and adding in the old 'worst x'. This update of `centroid` is O(n). 
    /// Shrinkage requires n function evaluations.
    #[inline]
    fn shrink<F>(&self, simplex: &mut Simplex, f: &mut WrappedFunction<F>, sigma: f64, centroid: &mut Array1<f64>) -> Result<(), DigiFiError>
        where F: FnMut(&Array1<f64>) -> f64
    {
        {
            let mut iter = simplex.iter_mut();
            let (_, x0) = iter.next()
                .ok_or(DigiFiError::Other {
                    title: "Nelder-Mead Numerical Engine".to_owned(),
                    details: "Could not grab next element from the simplex.".to_owned(),
                })?;
            for (fi, xi) in iter {
                *xi *= sigma;
                *xi += &((1.0 - sigma) * &x0.view());
                *fi = f.call(xi);
            }
        }
        let n: usize = simplex.len() - 1;
        let old_worst: Array1<f64> = simplex[n - 1].1.to_owned();
        *centroid *= sigma;
        *centroid += &((1.0 - sigma) * &simplex[0].1);
        self.order_simplex(simplex);
        *centroid += &((&old_worst - &simplex[n - 1].1) / (n - 1) as f64);
        Ok(())
    }

    /// Calculate the centroid of all points but the worst one. Assumes that the simplex is ordered. This calculation is O(n^2).
    #[inline]
    fn centroid(&self, simplex: &Simplex) -> Array1<f64> {
        let n: usize = simplex.len();
        let mut centroid: Array1<f64> = Array1::zeros(simplex[0].1.len());
        for (_, xi) in simplex.iter().take(n-1) {
            centroid += xi;
        }
        centroid / (n-1) as f64
    }

    /// This sorting algorithm should have a runtime of O(n) if only one new element is inserted. After a shrinkage, the runtime is O(n log n).
    #[inline]
    fn order_simplex(&self, simplex: &mut Simplex) -> () {
        simplex.sort_unstable_by(|&(fa, _), &(fb, _)| fa.partial_cmp(&fb).unwrap());
    }
}


/// A minimizer for a scalar function of one or more variables using the Nelder-Mead algorithm.
///
/// # Input
/// - `f`: Closure (objective function) that will be minimized
/// - `initial_guess`: An array of values where the minimization algorithm will begin
/// - `max_iterations`: Maximum number of iterations the algorithm is allowed to perform
/// - `max_fun_calls`: Maximum number of function calls the algorithm is allowed to perform
/// - `adaptive`: Whether to adapt parameters to the dimensionality of the problem (Useful for high-dimensional minimization)
/// - `x_tolerance`: Absolute error in function parameters between iterations that is acceptable for convergence
/// - `f_tolerance`: Absolute error in function values between iterations that is acceptable for convergence
///
/// # Output
/// - An array of parameters that minimize the provided objective function
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Nelder-Mead_method>
/// - Original Source: <https://doi.org/10.1093/comjnl/8.1.27>
pub fn nelder_mead<F: FnMut(&[f64]) -> f64>(
    mut f: F, initial_guess: Vec<f64>, max_iterations: Option<u64>, max_fun_calls: Option<u64>, adaptive: Option<bool>, xtol: Option<f64>, ftol: Option<f64>
) -> Result<Array1<f64>, DigiFiError> {
    let minimizer: NelderMead = NelderMead::new(max_iterations, max_fun_calls, adaptive, xtol, ftol);
    minimizer.minimize(|x: &Array1<f64>| { f(&x.to_vec()) }, Array1::from_vec(initial_guess))
}