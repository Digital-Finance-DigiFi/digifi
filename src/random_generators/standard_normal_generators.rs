use ndarray::Array1;
use crate::error::DigiFiError;
use crate::random_generators::{RandomGenerator, generate_seed, uniform_generators::FibonacciGenerator};
use crate::random_generators::generator_algorithms::{accept_reject, inverse_transform, box_muller, marsaglia, ziggurat};
use crate::statistics::continuous_distributions::{LaplaceDistribution, NormalDistribution};
use crate::statistics::ProbabilityDistribution;


#[derive(Debug)]
/// # Description
/// Pseudo-random number generator for standard normal distribution.
/// 
/// It samples the Laplace distribution to generate the standard normal distribution (i.e., exponential tilting).
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Rejection_sampling>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::covariance;
/// use digifi::random_generators::{RandomGenerator, StandardNormalAcceptReject};
///
/// let snar: StandardNormalAcceptReject = StandardNormalAcceptReject::new_shuffle(100_000).unwrap();
/// let sample: Array1<f64> = snar.generate().unwrap();
///
/// assert!((sample.mean().unwrap() - 0.0).abs() < 1_000_000.0 * TEST_ACCURACY);
/// assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 30_000_000.0 * TEST_ACCURACY);
/// ```
pub struct StandardNormalAcceptReject {
    /// The maximum size of the sample to generate
    max_sample_size: usize,
    /// Scale parameter for the Laplace distribution
    lap_b: f64,
    /// Seed for the first random number generator
    seed_1: u32,
    /// Seed for the second random number generator
    seed_2: u32,
}

impl StandardNormalAcceptReject {
    /// # Description
    /// Creates a new `StandardNormalAcceptReject` instance.
    /// 
    /// # Input
    /// - `sample_size`: The maximum size of the sample to generate
    /// - `lap_b`: Scale parameter for the Laplace distribution
    /// - `seed_1`: Seed for the first random number generator
    /// - `seed_2`: Seed for the second random number generator
    /// 
    /// # Errors
    /// - Returns an error if the argument `lap_b` is not positive.
    pub fn new(max_sample_size: usize, lap_b: f64, seed_1: u32, seed_2: u32) -> Result<Self, DigiFiError> {
        if lap_b <= 0.0 {
            return Err(DigiFiError::ParameterConstraint { title: "Accept-Reject Algorithm".to_owned(), constraint: "The argument `lap_b` must be positive.".to_owned(), });
        }
        Ok(StandardNormalAcceptReject { max_sample_size, lap_b, seed_1, seed_2 })
    }
}

impl RandomGenerator<StandardNormalAcceptReject> for StandardNormalAcceptReject {
    /// # Description
    /// Creates a new `StandardNormalAcceptReject` instance with random parameters.
    /// 
    /// # Input
    /// - `sample_size`: The maximum size of the sample to generate
    fn new_shuffle(sample_size: usize) -> Result<Self, DigiFiError> {
        Ok(StandardNormalAcceptReject { max_sample_size: sample_size, lap_b: 1.0, seed_1: generate_seed()?, seed_2: generate_seed()? })
    }

    /// # Description
    /// Array of pseudo-random generated numbers based on the Accept-Reject Method and the probability of the Laplace Distribution lap_p.
    /// 
    /// # Output
    /// - An array of pseudo-random numbers following a standard normal distribution
    fn generate(&self) -> Result<Array1<f64>, DigiFiError> {
        let u_1: Array1<f64> = FibonacciGenerator::new(self.seed_1, self.max_sample_size, 5, 17, 714_025, 1_366, 150_889).generate()?;
        let m: f64 = (2.0 * std::f64::consts::E / std::f64::consts::PI).sqrt();
        let u_2: Array1<f64> = FibonacciGenerator::new(self.seed_2, self.max_sample_size, 5, 17, 714_025, 1_366, 150_889).generate()?;
        // Laplace distribution sampling.
        let laplace_dist: LaplaceDistribution = LaplaceDistribution::new(0.0, self.lap_b)?;
        let l: Array1<f64> = laplace_dist.inverse_cdf(&u_1)?
            .map(|i| { if i.is_infinite() && i.is_sign_positive() { 1.0 } else if i.is_infinite() && i.is_sign_negative() { 0.0 } else { *i } });
        // Accept-Reject algorithm.
        let standard_normal_dist: NormalDistribution = NormalDistribution::new(0.0, 1.0)?;
        accept_reject(&standard_normal_dist, &laplace_dist, &l, m, &u_2)
    }
}


#[derive(Debug)]
/// # Description
/// Pseudo-random number generator for standard normal distribution.
/// 
/// It returns the array of values from sampling an inverse CDF.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Inverse_transform_sampling>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::covariance;
/// use digifi::random_generators::{RandomGenerator, StandardNormalInverseTransform};
///
/// let snit: StandardNormalInverseTransform = StandardNormalInverseTransform::new_shuffle(100_000).unwrap();
/// let sample: Array1<f64> = snit.generate().unwrap();
///
/// assert!((sample.mean().unwrap() - 0.0).abs() < 1_000_000.0 * TEST_ACCURACY);
/// assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 30_000_000.0 * TEST_ACCURACY);
/// ```
pub struct StandardNormalInverseTransform {
    /// Number of random samples to generate
    sample_size: usize,
}

impl StandardNormalInverseTransform {
    /// # Description
    /// Creates a new `StandardNormalInverseTransform` instance.
    /// 
    /// # Input
    /// - `sample_size`: Number of random samples to generate
    pub fn new(sample_size: usize) -> Self {
        StandardNormalInverseTransform { sample_size }
    }
}

impl RandomGenerator<StandardNormalInverseTransform> for StandardNormalInverseTransform {
    /// # Description
    /// Creates a new `StandardNormalInverseTransform` instance.
    /// 
    /// # Input
    /// - `sample_size`: Number of pseudo-random numbers to generate
    fn new_shuffle(sample_size: usize) -> Result<Self, DigiFiError> {
        Ok(StandardNormalInverseTransform { sample_size })
    }

    /// # Description
    /// Array of pseudo-random generated numbers based on the Inverse Transform Method.
    /// 
    /// # Output
    /// - An array of pseudo-random numbers following a standard normal distribution
    fn generate(&self) -> Result<Array1<f64>, DigiFiError> {
        let standard_normal_dist: NormalDistribution = NormalDistribution::new(0.0, 1.0)?;
        inverse_transform(&standard_normal_dist, self.sample_size)
    }
}


#[derive(Debug)]
/// # Description
/// Pseudo-random number generator for standard normal distribution.
/// 
/// It returns two independent pseudo-random arrays.
/// 
/// # LaTeX Formula
/// - Z_{0} = \\sqrt{-2ln(U_{1})} \\cos(2\\pi U_{2})
/// - Z_{1} = \\sqrt{-2ln(U_{1})} \\sin(2\\pi U_{2})
/// - U_{1}, U_{2} are independent uniform random variables
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Box–Muller_transform>
/// - Original Source: <https://doi.org/10.1214%2Faoms%2F1177706645>
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::covariance;
/// use digifi::random_generators::{RandomGenerator, StandardNormalBoxMuller};
///
/// let snbm: StandardNormalBoxMuller = StandardNormalBoxMuller::new_shuffle(100_000).unwrap();
/// let sample: Array1<f64> = snbm.generate().unwrap();
///
/// assert!((sample.mean().unwrap() - 0.0).abs() < 30_000_000.0 * TEST_ACCURACY);
/// assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 30_000_000.0 * TEST_ACCURACY);
/// ```
pub struct StandardNormalBoxMuller {
    /// Number of random samples to generate
    sample_size: usize,
    /// Seed for the first uniform random number generator
    seed_1: u32,
    /// Seed for the second uniform random number generator
    seed_2: u32,
}

impl StandardNormalBoxMuller {
    /// # Description
    /// Creates a new `StandardNormalBoxMuller` instance.
    /// 
    /// # Input
    /// - `sample_size`: Number of random samples to generate
    /// - `seed_1`: Seed for the first uniform random number generator
    /// - `seed_2`: Seed for the second uniform random number generator
    pub fn new(sample_size: usize, seed_1: u32, seed_2: u32) -> Self {
        StandardNormalBoxMuller { sample_size, seed_1, seed_2 }
    }
}

impl RandomGenerator<StandardNormalBoxMuller> for StandardNormalBoxMuller {
    /// # Description
    /// Creates a new `StandardNormalBoxMuller` instance with random parameters.
    /// 
    /// # Input
    /// - `sample_size`: Number of pseudo-random numbers to generate
    fn new_shuffle(sample_size: usize) -> Result<Self, DigiFiError> {
        Ok(StandardNormalBoxMuller { sample_size, seed_1: generate_seed()?, seed_2: generate_seed()? })
    }

    /// # Description
    /// One of two independent arrays of pseudo-random generated numbers produced using the Box-Muller Algorithm.
    /// 
    /// # Output
    /// - Two arrays of pseudo-random numbers following a standard normal distribution
    fn generate(&self) -> Result<Array1<f64>, DigiFiError> {
        let u_1: Array1<f64> = FibonacciGenerator::new(self.seed_1, self.sample_size, 5, 17, 714_025, 1_366, 150_889).generate()?;
        let u_2: Array1<f64> = FibonacciGenerator::new(self.seed_2, self.sample_size, 5, 17, 714_025, 1_366, 150_889).generate()?;
        let normal_arrays: (Array1<f64>, Array1<f64>) = box_muller(&u_1, &u_2);
        Ok(normal_arrays.0)
    }
}


#[derive(Debug)]
/// # Description
/// Pseudo-random number generator for standard normal distribution.
/// 
/// It returns two independent pseudo-random arrays.
/// 
/// # LaTeX Formula
/// - S = W^{2}_{1} + W^{2}_{2}
/// - If S < 1, then:
/// - Z_{0} = W_{1} \\sqrt{\\frac{-2ln(S)}{S}}
/// - Z_{1} = W_{2} \\sqrt{\\frac{-2ln(S)}{S}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Marsaglia_polar_method>
/// - Original Source: <https://doi.org/10.1137%2F1006063>
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::covariance;
/// use digifi::random_generators::{RandomGenerator, StandardNormalMarsaglia};
///
/// let snm: StandardNormalMarsaglia = StandardNormalMarsaglia::new_shuffle(10_000).unwrap();
/// let sample: Array1<f64> = snm.generate().unwrap();
///
/// assert!((sample.mean().unwrap() - 0.0).abs() < 100_000_000.0 * TEST_ACCURACY);
/// assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 100_000_000.0 * TEST_ACCURACY);
/// ```
pub struct StandardNormalMarsaglia {
    /// Number of random samples to generate
    sample_size: usize,
    /// Maximum number of iterations for the algorithm
    max_iterations: usize,
}

impl StandardNormalMarsaglia {
    /// # Description
    /// Creates a new `StandardNormalMarsaglia` instance.
    /// 
    /// # Input
    /// - `sample_size`: Number of random samples to generate
    /// - `max_iterations`: Maximum number of iterations for the algorithm
    pub fn new(sample_size: usize, max_iterations: usize) -> Self {
        StandardNormalMarsaglia { sample_size, max_iterations }
    }
}

impl RandomGenerator<StandardNormalMarsaglia> for StandardNormalMarsaglia {
    /// # Description
    /// Creates a new `StandardNormalMarsaglia` instance with random parameters.
    /// 
    /// # Input
    /// - `sample_size`: Number of pseudo-random numbers to generate
    fn new_shuffle(sample_size: usize) -> Result<Self, DigiFiError> {
        Ok(StandardNormalMarsaglia { sample_size, max_iterations: 1_000 })
    }

    /// # Description
    /// Two independent arrays of pseudo-random generated numbers based on the Marsaglie Method.
    /// 
    /// # Output
    /// - Two arrays of pseudo-random numbers following a standard normal distribution
    fn generate(&self) -> Result<Array1<f64>, DigiFiError> {
        let mut z_1: Array1<f64> = Array1::from_vec(vec![1.0; self.sample_size]);
        let mut z_2: Array1<f64> = Array1::from_vec(vec![1.0; self.sample_size]);
        //  Marsaglia method.
        for i in 0..self.sample_size {
            (z_1[i], z_2[i]) = marsaglia(self.max_iterations)?
                .ok_or(DigiFiError::Other { title: "Marsaglia Algorithm".to_owned(), details: "Marsaglia algorithm failed to generate pseudo-random numbers.".to_owned(), })?;
        }
        Ok(z_1)
    }
}


#[derive(Debug)]
/// # Description
/// Pseudo-random number generator for standard normal distribution.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Ziggurat_algorithm>
/// - Original Source: <https://doi.org/10.1145/1464291.1464310>
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::random_generators::{RandomGenerator, StandardNormalZiggurat};
///
/// let snz: StandardNormalZiggurat = StandardNormalZiggurat::new_shuffle(1_000).unwrap();
/// let sample: Array1<f64> = snz.generate().unwrap();
///
/// assert!((sample.mean().unwrap() - 0.0).abs() < 10_000_000.0 * TEST_ACCURACY);
/// ```
pub struct StandardNormalZiggurat {
    /// Number of random samples to generate
    sample_size: usize,
    /// Number of regions for the Ziggurat method
    rectangle_size: f64,
    /// Maximum number of iterations for the algorithm
    max_iterations: usize,
    /// The step size for the initial guess values
    dx: f64,
    /// The limit for the x-axis in the normal distribution
    limit: f64,
}

impl StandardNormalZiggurat {
    /// # Description
    /// Creates a new `StandardNormalZiggurat` instance.
    /// 
    /// # Input
    /// - `sample_size`: Number of random samples to generate
    /// - `regions`: Number of regions for the Ziggurat method
    /// - `max_iterations`: Maximum number of iterations for the algorithm
    pub fn new(sample_size: usize, rectangle_size: f64, max_iterations: usize, dx: f64, limit: f64) -> Self {
        StandardNormalZiggurat { sample_size, rectangle_size, max_iterations, dx, limit }
    }
}

impl RandomGenerator<StandardNormalZiggurat> for StandardNormalZiggurat {
    /// # Description
    /// Creates a new `StandardNormalZiggurat` instance with random parameters.
    /// 
    /// # Input
    /// - `sample_size`: Number of pseudo-random numbers to generate
    fn new_shuffle(sample_size: usize) -> Result<Self, DigiFiError> {
        Ok(StandardNormalZiggurat { sample_size, rectangle_size: 1.0/256.0, max_iterations: 10_000, dx: 0.001, limit: 6.0 })
    }

    /// # Description
    /// Array of pseudo-random generated numbers based on the Ziggurat Algorithm.
    ///
    /// Note: This version of Ziggurat algorithm does not implement a fallback algorithm for the tail.
    /// 
    /// # Output
    /// - An array of pseudo-random numbers following a standard normal distribution
    fn generate(&self) -> Result<Array1<f64>, DigiFiError> {
        let mut x_guess: Vec<f64> = Vec::<f64>::new();
        let mut current_x: f64 = 0.0;
        let mut rectangle_length: f64 = 0.0;
        let standard_normal_dist: NormalDistribution = NormalDistribution::new(0.0, 1.0)?;
        // Initial guess.
        while current_x < self.limit {
            rectangle_length = rectangle_length + self.dx;
            let current_area: f64 = (standard_normal_dist.pdf(&Array1::from_vec(vec![current_x]))? - standard_normal_dist.pdf(&Array1::from_vec(vec![rectangle_length]))?)[0] * rectangle_length;
            if self.rectangle_size < current_area {
                x_guess.push(rectangle_length);
                current_x = rectangle_length;
            }
        }
        // Ziggurat algorithm.
        ziggurat(&Array1::from_vec(x_guess), self.sample_size, self.max_iterations)
    }
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::{random_generators::RandomGenerator, utilities::TEST_ACCURACY};

    #[test]
    fn unit_test_accept_reject() -> () {
        use crate::random_generators::standard_normal_generators::StandardNormalAcceptReject;
        use crate::statistics::covariance;
        let snar: StandardNormalAcceptReject = StandardNormalAcceptReject::new_shuffle(100_000).unwrap();
        let sample: Array1<f64> = snar.generate().unwrap();
        assert!((sample.mean().unwrap() - 0.0).abs() < 1_000_000.0 * TEST_ACCURACY);
        assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 30_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_inverse_transform() -> () {
        use crate::random_generators::standard_normal_generators::StandardNormalInverseTransform;
        use crate::statistics::covariance;
        let snit: StandardNormalInverseTransform = StandardNormalInverseTransform::new_shuffle(100_000).unwrap();
        let sample: Array1<f64> = snit.generate().unwrap();
        assert!((sample.mean().unwrap() - 0.0).abs() < 1_000_000.0 * TEST_ACCURACY);
        assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 30_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_box_muller() -> () {
        use crate::random_generators::standard_normal_generators::StandardNormalBoxMuller;
        use crate::statistics::covariance;
        let snbm: StandardNormalBoxMuller = StandardNormalBoxMuller::new_shuffle(100_000).unwrap();
        let sample: Array1<f64> = snbm.generate().unwrap();
        assert!((sample.mean().unwrap() - 0.0).abs() < 30_000_000.0 * TEST_ACCURACY);
        assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 30_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_marsaglia() -> () {
        use crate::random_generators::standard_normal_generators::StandardNormalMarsaglia;
        use crate::statistics::covariance;
        let snm: StandardNormalMarsaglia = StandardNormalMarsaglia::new_shuffle(10_000).unwrap();
        let sample: Array1<f64> = snm.generate().unwrap();
        assert!((sample.mean().unwrap() - 0.0).abs() < 100_000_000.0 * TEST_ACCURACY);
        assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 100_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_ziggurat() -> () {
        use crate::random_generators::standard_normal_generators::StandardNormalZiggurat;
        let snz: StandardNormalZiggurat = StandardNormalZiggurat::new_shuffle(1_000).unwrap();
        let sample: Array1<f64> = snz.generate().unwrap();
        assert!((sample.mean().unwrap() - 0.0).abs() < 10_000_000.0 * TEST_ACCURACY);
    }
}