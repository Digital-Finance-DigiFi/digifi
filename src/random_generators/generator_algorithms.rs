use std::io::Error;
use ndarray::Array1;
use crate::statistics::ProbabilityDistribution;
use crate::statistics::continuous_distributions::NormalDistribution;
use crate::utilities::compare_array_len;
use crate::random_generators::{RandomGenerator, uniform_generators::{LinearCongruentialGenerator, FibonacciGenerator}};


/// # Description
/// Implements the Accept-Reject Method, a Monte Carlo technique for generating random samples from a probability distribution.
/// 
/// # Input
/// - f_x: Target probability density function
/// - g_x: Proposal probability density function
/// - y_sample: Sample from the proposal distribution
/// - m: Constant such that m*g(x) >= f(x) for all x
/// - uniform_sample: Sample from a uniform distribution
/// 
/// # Output
/// - Sample array generated from the target distribution
/// 
/// # Errors
/// - Returns an error if the arrays produced by f_x and g_x are of different lengths
/// - Returns an error if arrays produced by f_x and g_x are different in length to y_sample
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Rejection_sampling
/// - Original Source: N/A
pub fn accept_reject(f_x: &impl ProbabilityDistribution, g_x : &impl ProbabilityDistribution, y_sample: &Array1<f64>,
                            m: f64, uniform_sample: &Array1<f64>) -> Result<Array1<f64>, Error> {
    let g: Array1<f64> = g_x.pdf(y_sample)?;
    let f: Array1<f64> = f_x.pdf(y_sample)?;
    compare_array_len(&g, &f, "g", "f")?;
    compare_array_len(&g, y_sample, "g", "sample")?;
    let mut x_sample: Vec<f64> = Vec::<f64>::new();
    for i in 0..y_sample.len() {
        if g[i].is_infinite() || f[i].is_infinite() {
            continue;
        } else if (uniform_sample[i] * m * g[i]) <= f[i] {
            x_sample.push(y_sample[i]);
        }
    }
    Ok(Array1::from_vec(x_sample))
}


/// # Description
/// Uses the Inverse Transform Method to generate random samples from a specified probability distribution function.
/// 
/// # Input
/// - f_x: Inverse of the cumulative distribution function
/// - sample_size: Number of random samples to generate
/// 
/// # Output
/// - Sample array generated from the specified distribution
/// 
/// # Latex
/// - X = F^{-1}(U)
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Inverse_transform_sampling
/// - Original SOurce: N/A
pub fn inverse_transform(f_x: &impl ProbabilityDistribution, sample_size: usize) -> Result<Array1<f64>, Error> {
    let u: Array1<f64> = FibonacciGenerator::new_shuffle(sample_size)?.generate()?;
    f_x.inverse_cdf(&u)
}


/// # Description
/// The Box-Muller algorithm transforms uniformly distributed samples into samples distributed according to the standard normal distribution.
/// 
/// # Input
/// - uniform_sample_1: First array of uniform samples
/// - uniform_sample_2: Second array of uniform samples
/// 
/// # Output
/// - Two arrays, each containing normal distributed samples
/// 
/// # Latex
/// -  Z_{0} = \\sqrt{-2ln(U_{1})} \\cos(2\\pi U_{2})
/// -  Z_{1} = \\sqrt{-2ln(U_{1})} \\sin(2\\pi U_{2})
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Box-Muller_transform
/// - Original Source: https://doi.org/10.1214%2Faoms%2F1177706645
pub fn box_muller(uniform_sample_1: &Array1<f64>, unfiform_sample_2: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let mult_1: Array1<f64> = uniform_sample_1.map(|x| if (x == &0.0) ||(x == &1.0) {0.0} else {(-2.0 * x.ln()).sqrt()} );
    let mult_21: Array1<f64> = unfiform_sample_2.map(|x| (2.0 * std::f64::consts::PI * x).cos() );
    let mult_22: Array1<f64> = unfiform_sample_2.map(|x| (2.0 * std::f64::consts::PI * x).sin() );
    (&mult_1 * &mult_21, &mult_1 * &mult_22)
}


/// # Description
/// The Marsaglia polar method for generating standard normal random variables from uniformly distributed random numbers.
/// 
/// # Input
/// - max_iterations: Maximum number of iterations
/// 
/// # Output
/// - A pair of standard normal random variables
/// 
/// # Latex
/// - S = W^{2}_{1} + W^{2}_{2}
/// - If S < 1, then:
/// - Z_{0} = W_{1} \\sqrt{\\frac{-2ln(S)}{S}}
/// - Z_{1} = W_{2} \\sqrt{\\frac{-2ln(S)}{S}}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Marsaglia_polar_method
/// - Original Source: https://doi.org/10.1137%2F1006063
pub fn marsaglia(max_iterations: usize) -> Result<Option<(f64, f64)>, Error> {
    let w_1: Array1<f64> = 2.0 * FibonacciGenerator::new_shuffle(max_iterations)?.generate()? - 1.0;
    let w_2: Array1<f64> = 2.0 * FibonacciGenerator::new_shuffle(max_iterations)?.generate()? - 1.0;
    let mut i: usize = 0;
    while i < max_iterations {
        let s: f64 = w_1[i].powi(2) + w_2[i].powi(2);
        if s < 1.0 {
            let t: f64 = (-2.0 * s.ln() / s).sqrt();
            return Ok(Some((w_1[i]*t, w_2[i]*t)));
        }
        i += 1;
    }
    Ok(None)
}


/// # Description
/// The Ziggurat algorithm is a fast method for generating random samples from a normal distribution.
/// 
/// # Input
/// - x_guess: Initial guess values
/// - sample_size: Number of random samples to generate
/// - max_iterations: Maximum number of iterations for the algorithm
/// 
/// # Output
/// - Sample arrays generated from the normal distribution
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Ziggurat_algorithm
/// - Original Source: https://doi.org/10.1145/1464291.1464310
pub fn ziggurat(x_guess: &Array1<f64>, sample_size: usize, max_iterations: usize) -> Result<Array1<f64>, Error> {
    let normal_dist: NormalDistribution = NormalDistribution::new(0.0, 1.0)?;
    let y: Array1<f64> = normal_dist.pdf(&x_guess)?;
    let mut z: Array1<f64> = Array1::from_vec(vec![1.0; sample_size]);
    for j in 0..sample_size {
        let u_1: Array1<f64> = 2.0 * FibonacciGenerator::new_shuffle(max_iterations)?.generate()? - 1.0;
        let u_2: Array1<f64> = FibonacciGenerator::new_shuffle(max_iterations)?.generate()?;
        let mut i: usize = 0;
        while (z[j]==1.0) && (i < max_iterations) {
            // Generates a random index and ensures that it is in the range [1, x_guess.len()-1]
            let rand_index: usize = (((x_guess.len() - 2) as f64) * LinearCongruentialGenerator::new_shuffle(1)?.generate()?[0]).ceil() as usize + 1;
            let x: f64 = u_1[rand_index] * x_guess[rand_index];
            if x.abs() < x_guess[rand_index-1] {
                z[j] = x;
            } else {
                let y_: f64 = y[rand_index] + u_2[rand_index] * (y[rand_index-1] - y[rand_index]);
                let point: f64 = normal_dist.pdf(&Array1::from_vec(vec![x]))?[0];
                if y_ < point {
                    z[j] = x;
                }
            }
            i += 1;
        }
    }
    Ok(z)
}