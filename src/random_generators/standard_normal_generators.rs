use ndarray::{Array1, Axis, concatenate};
use crate::error::{DigiFiError, ErrorTitle};
use crate::random_generators::{RandomGenerator, generate_seed, uniform_generators::FibonacciGenerator};
use crate::random_generators::generator_algorithms::{accept_reject, inverse_transform, box_muller, marsaglia, ziggurat};
use crate::statistics::continuous_distributions::{LaplaceDistribution, NormalDistribution};
use crate::statistics::ProbabilityDistribution;


#[derive(Clone, Copy, Debug)]
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
/// let snar: StandardNormalAcceptReject = StandardNormalAcceptReject::new_shuffle(1_000_000).unwrap();
/// let sample: Array1<f64> = snar.generate().unwrap();
///
/// assert!((sample.mean().unwrap() - 0.0).abs() < 100_000.0 * TEST_ACCURACY);
/// assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 100_000.0 * TEST_ACCURACY);
/// ```
pub struct StandardNormalAcceptReject {
    /// The maximum size of the sample to generate
    max_sample_size: usize,
    /// Scale parameter for the Laplace distribution
    lap_b: f64,
    /// Seed for the random number generator
    seed: f64,
}

impl StandardNormalAcceptReject {
    /// Creates a new `StandardNormalAcceptReject` instance.
    /// 
    /// # Input
    /// - `sample_size`: The maximum size of the sample to generate
    /// - `lap_b`: Scale parameter for the Laplace distribution
    /// - `seed`: Seed for the random number generator
    /// 
    /// # Errors
    /// - Returns an error if the argument `lap_b` is not positive.
    pub fn build(max_sample_size: usize, lap_b: f64, seed: u32) -> Result<Self, DigiFiError> {
        if lap_b <= 0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `lap_b` must be positive.".to_owned(),
            });
        }
        Ok(Self { max_sample_size, lap_b, seed: seed as f64 })
    }
}

impl ErrorTitle for StandardNormalAcceptReject {
    fn error_title() -> String {
        String::from("Accept-Reject Algorithm")
    }
}

impl RandomGenerator<StandardNormalAcceptReject> for StandardNormalAcceptReject {
    /// Creates a new `StandardNormalAcceptReject` instance with random parameters.
    /// 
    /// # Input
    /// - `sample_size`: The maximum size of the sample to generate
    fn new_shuffle(sample_size: usize) -> Result<Self, DigiFiError> {
        Ok(Self { max_sample_size: sample_size, lap_b: 1.0, seed: generate_seed()?, })
    }

    /// Array of pseudo-random generated numbers based on the Accept-Reject Method and the probability of the Laplace Distribution lap_p.
    /// 
    /// # Output
    /// - An array of pseudo-random numbers following a standard normal distribution
    fn generate(&self) -> Result<Array1<f64>, DigiFiError> {
        let m: f64 = (2.0 * std::f64::consts::E / std::f64::consts::PI).sqrt();
        let mut u: Array1<f64> = FibonacciGenerator::build(self.seed as u32, self.max_sample_size + 1, 5, 17, 714_025, 1_366, 150_889)?.generate()?;
        // Laplace distribution sampling
        let laplace_dist: LaplaceDistribution = LaplaceDistribution::build(0.0, self.lap_b)?;
        let mut l: Array1<f64> = laplace_dist.inverse_cdf_iter(u.iter())?
            .map(|i| { if i.is_infinite() && i.is_sign_positive() { 1.0 } else if i.is_infinite() && i.is_sign_negative() { 0.0 } else { *i } });
        let last_index: usize = u.len() - 1;
        l.remove_index(Axis(0), 0);
        u.remove_index(Axis(0), last_index);
        // Accept-Reject algorithm
        let standard_normal_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0)?;
        accept_reject(&standard_normal_dist, &laplace_dist, &l, m, &u)
    }
}


#[derive(Clone, Copy, Debug)]
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
    /// Creates a new `StandardNormalInverseTransform` instance.
    /// 
    /// # Input
    /// - `sample_size`: Number of random samples to generate
    pub fn new(sample_size: usize) -> Self {
        Self { sample_size }
    }
}

impl RandomGenerator<StandardNormalInverseTransform> for StandardNormalInverseTransform {
    /// Creates a new `StandardNormalInverseTransform` instance.
    /// 
    /// # Input
    /// - `sample_size`: Number of pseudo-random numbers to generate
    fn new_shuffle(sample_size: usize) -> Result<Self, DigiFiError> {
        Ok(Self { sample_size })
    }

    /// Array of pseudo-random generated numbers based on the Inverse Transform Method.
    /// 
    /// # Output
    /// - An array of pseudo-random numbers following a standard normal distribution
    fn generate(&self) -> Result<Array1<f64>, DigiFiError> {
        inverse_transform(&NormalDistribution::build(0.0, 1.0)?, self.sample_size)
    }
}


#[derive(Clone, Copy, Debug)]
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
/// let snbm: StandardNormalBoxMuller = StandardNormalBoxMuller::new_shuffle(1_000_000).unwrap();
/// let sample: Array1<f64> = snbm.generate().unwrap();
///
/// assert!((sample.mean().unwrap() - 0.0).abs() < 1_000_000.0 * TEST_ACCURACY);
/// assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 1_000_000.0 * TEST_ACCURACY);
/// ```
pub struct StandardNormalBoxMuller {
    /// Number of random samples to generate
    sample_size: usize,
    /// Seed for the uniform random number generator
    seed: f64,
}

impl StandardNormalBoxMuller {
    /// Creates a new `StandardNormalBoxMuller` instance.
    /// 
    /// # Input
    /// - `sample_size`: Number of random samples to generate
    /// - `seed`: Seed for the uniform random number generator
    pub fn new(sample_size: usize, seed: u32) -> Self {
        Self { sample_size, seed: seed as f64 }
    }
}

impl RandomGenerator<StandardNormalBoxMuller> for StandardNormalBoxMuller {
    /// Creates a new `StandardNormalBoxMuller` instance with random parameters.
    /// 
    /// # Input
    /// - `sample_size`: Number of pseudo-random numbers to generate
    fn new_shuffle(sample_size: usize) -> Result<Self, DigiFiError> {
        Ok(Self { sample_size, seed: generate_seed()?, })
    }

    /// One of two independent arrays of pseudo-random generated numbers produced using the Box-Muller Algorithm.
    /// 
    /// # Output
    /// - Two arrays of pseudo-random numbers following a standard normal distribution
    fn generate(&self) -> Result<Array1<f64>, DigiFiError> {
        let n: usize = if self.sample_size.is_multiple_of(2) { self.sample_size } else { self.sample_size + 1 };
        let u: Array1<f64> = FibonacciGenerator::build(self.seed as u32, n, 5, 17, 714_025, 1_366, 150_889)?.generate()?;
        let (u_1, u_2) = u.into_iter().enumerate()
            .fold((Vec::with_capacity(self.sample_size / 2), Vec::with_capacity(self.sample_size / 2)), |(mut u_1, mut u_2), (i, u)| {
                if  i.is_multiple_of(2) { u_2.push(u); } else { u_1.push(u); }
                (u_1, u_2)
            } );
        let (n_1, n_2) = box_muller(&Array1::from_vec(u_1), &Array1::from_vec(u_2))?;
        let mut n_: Array1<f64> = concatenate(Axis(0), &[n_1.view(), n_2.view()])?;
        if n != self.sample_size { n_.remove_index(Axis(0), 0) }
        Ok(n_)
    }
}


#[derive(Clone, Copy, Debug)]
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
/// let snm: StandardNormalMarsaglia = StandardNormalMarsaglia::new_shuffle(1_000_000).unwrap();
/// let sample: Array1<f64> = snm.generate().unwrap();
///
/// assert!((sample.mean().unwrap() - 0.0).abs() < 3_000_000.0 * TEST_ACCURACY);
/// assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 5_000_000.0 * TEST_ACCURACY);
/// ```
pub struct StandardNormalMarsaglia {
    /// Number of random samples to generate
    sample_size: usize,
}

impl StandardNormalMarsaglia {
    /// Creates a new `StandardNormalMarsaglia` instance.
    /// 
    /// # Input
    /// - `sample_size`: Number of random samples to generate
    pub fn new(sample_size: usize) -> Self {
        Self { sample_size }
    }
}

impl RandomGenerator<StandardNormalMarsaglia> for StandardNormalMarsaglia {
    /// Creates a new `StandardNormalMarsaglia` instance with random parameters.
    /// 
    /// # Input
    /// - `sample_size`: Number of pseudo-random numbers to generate
    fn new_shuffle(sample_size: usize) -> Result<Self, DigiFiError> {
        Ok(Self { sample_size })
    }

    /// Two independent arrays of pseudo-random generated numbers based on the Marsaglie Method.
    /// 
    /// # Output
    /// - Two arrays of pseudo-random numbers following a standard normal distribution
    fn generate(&self) -> Result<Array1<f64>, DigiFiError> {
        let n: usize = if self.sample_size.is_multiple_of(2) { self.sample_size } else { self.sample_size + 1 };
        let mut z: Vec<f64> = Vec::with_capacity(n);
        for _ in (0..(n / 2)).into_iter() {
            let (z_1, z_2) = marsaglia()?;
            z.push(z_1);
            z.push(z_2);
        }
        if n != self.sample_size { z.pop(); }
        Ok(Array1::from_vec(z))
    }
}


#[derive(Clone, Copy, Debug)]
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
    /// Creates a new `StandardNormalZiggurat` instance.
    /// 
    /// # Input
    /// - `sample_size`: Number of random samples to generate
    /// - `regions`: Number of regions for the Ziggurat method
    /// - `max_iterations`: Maximum number of iterations for the algorithm
    pub fn new(sample_size: usize, rectangle_size: f64, max_iterations: usize, dx: f64, limit: f64) -> Self {
        Self { sample_size, rectangle_size, max_iterations, dx, limit }
    }
}

impl RandomGenerator<StandardNormalZiggurat> for StandardNormalZiggurat {
    /// Creates a new `StandardNormalZiggurat` instance with random parameters.
    /// 
    /// # Input
    /// - `sample_size`: Number of pseudo-random numbers to generate
    fn new_shuffle(sample_size: usize) -> Result<Self, DigiFiError> {
        Ok(Self { sample_size, rectangle_size: 1.0/256.0, max_iterations: 10_000, dx: 0.001, limit: 6.0 })
    }

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
        let standard_normal_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0)?;
        // Initial guess.
        while current_x < self.limit {
            rectangle_length = rectangle_length + self.dx;
            let current_area: f64 = (standard_normal_dist.pdf(current_x)? - standard_normal_dist.pdf(rectangle_length)?) * rectangle_length;
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
        let snar: StandardNormalAcceptReject = StandardNormalAcceptReject::new_shuffle(1_000_000).unwrap();
        let sample: Array1<f64> = snar.generate().unwrap();
        assert!((sample.mean().unwrap() - 0.0).abs() < 100_000.0 * TEST_ACCURACY);
        assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 100_000.0 * TEST_ACCURACY);
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
        let snbm: StandardNormalBoxMuller = StandardNormalBoxMuller::new_shuffle(1_000_000).unwrap();
        let sample: Array1<f64> = snbm.generate().unwrap();
        assert!((sample.mean().unwrap() - 0.0).abs() < 1_000_000.0 * TEST_ACCURACY);
        assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 1_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_marsaglia() -> () {
        use crate::random_generators::standard_normal_generators::StandardNormalMarsaglia;
        use crate::statistics::covariance;
        let snm: StandardNormalMarsaglia = StandardNormalMarsaglia::new_shuffle(1_000_000).unwrap();
        let sample: Array1<f64> = snm.generate().unwrap();
        assert!((sample.mean().unwrap() - 0.0).abs() < 3_000_000.0 * TEST_ACCURACY);
        assert!((covariance(&sample, &sample, 0).unwrap() - 1.0).abs() < 5_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_ziggurat() -> () {
        use crate::random_generators::standard_normal_generators::StandardNormalZiggurat;
        let snz: StandardNormalZiggurat = StandardNormalZiggurat::new_shuffle(1_000).unwrap();
        let sample: Array1<f64> = snz.generate().unwrap();
        assert!((sample.mean().unwrap() - 0.0).abs() < 10_000_000.0 * TEST_ACCURACY);
    }
}