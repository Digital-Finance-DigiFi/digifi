use ndarray::Array1;
use crate::error::{DigiFiError, ErrorTitle};
use crate::random_generators::{RandomGenerator, generate_seed};


#[derive(Clone, Copy, Debug)]
/// Pseudo-random number generator for uniform distribution.
/// 
/// # LaTeX Formula
/// - N_{i} = (aN_{i-1}+b) mod M
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Linear_congruential_generator>
/// - Original Source: <https://archive.org/details/proceedings_of_a_second_symposium_on_large-scale_/mode/2up>
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::random_generators::{RandomGenerator, LinearCongruentialGenerator};
///
/// let lcg: LinearCongruentialGenerator = LinearCongruentialGenerator::new(12_345, 1_000_000, 244_944, 1_597, 51_749);
/// let sample: Array1<f64> = lcg.generate().unwrap();
///
/// assert_eq!(sample.len(), 1_000_000);
/// assert!((sample.mean().unwrap() - 0.5).abs() < 1_000.0 * TEST_ACCURACY)
/// ```
pub struct LinearCongruentialGenerator {
    /// Seed of the generator
    seed: f64,
    /// Number of pseudo-random numbers to generate
    sample_size: usize,
    /// Mod of the linear congruential generator
    m: f64,
    /// Multiplierof the linear congruential generator
    a: f64,
    /// Increment of the linear congruential generator
    b: f64,
}

impl LinearCongruentialGenerator {
    /// Creates a new `LinearCongruentialGenerator` instance.
    /// 
    /// # Input
    /// - `seed`: Seed of the generator
    /// - `sample_size`: Number of pseudo-random numbers to generate
    /// - `m`: Mod of the linear congruential generator
    /// - `a`: Multiplierof the linear congruential generator
    /// - `b`: Increment of the linear congruential generator
    pub fn new(seed: u32, sample_size: usize, m: u32, a: u32, b: u32) -> Self {
        LinearCongruentialGenerator { seed: seed as f64, sample_size, m: m as f64, a: a as f64, b: b as f64, }
    }
}

impl RandomGenerator<LinearCongruentialGenerator> for LinearCongruentialGenerator {
    /// Creates a new `LinearCongruentialGenerator` instance with random parameters.
    /// 
    /// # Input
    /// - `sample_size`: Number of pseudo-random numbers to generate
    fn new_shuffle(sample_size: usize) -> Result<Self, DigiFiError> {
        let seed: f64 = generate_seed()?;
        let m: f64 = seed * 1_234.0;
        let a: f64 = seed / 10.0;
        let b: f64 = m / 10.0;
        Ok(LinearCongruentialGenerator { seed, sample_size, m, a, b })
    }

    /// Array of pseudo-random generated numbers based on Linear Congruential Generator.
    /// 
    /// # Output
    /// - An array pseudo-random numbers following Uniform distribution
    fn generate(&self) -> Result<Array1<f64>, DigiFiError> {
        let (_, u) = (0..self.sample_size).into_iter()
            .fold((self.seed, Vec::with_capacity(self.sample_size)), |(prev, mut u), _| {
                let next: f64 = (self.a * prev + self.b) % self.m;
                u.push(next / self.m);
                (next, u)
            } );
        Ok(Array1::from_vec(u))
    }
}


#[derive(Clone, Copy, Debug)]
/// Pseudo-random number generator for uniform distribution.
/// 
/// # LaTeX Formula
/// - N_{i} = (N_{i-nu}+N_{i-mu}) mod M
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Lagged_Fibonacci_generator>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::random_generators::{RandomGenerator, FibonacciGenerator};
///
/// // Input seed
/// let fg: FibonacciGenerator = FibonacciGenerator::build(12_345, 1_000_000, 5, 17, 714_025, 1_366, 150_889).unwrap();
/// let sample: Array1<f64> = fg.generate().unwrap();
///
/// assert_eq!(sample.len(), 1_000_000);
/// assert!((sample.mean().unwrap() - 0.5).abs() < 10_000.0 * TEST_ACCURACY);
///
/// // Auto-generated seed
/// let fg: FibonacciGenerator = FibonacciGenerator::new_shuffle(1_000_000).unwrap();
/// let sample: Array1<f64> = fg.generate().unwrap();
///
/// assert_eq!(sample.len(), 1_000_000);
/// assert!((sample.mean().unwrap() - 0.5).abs() < 100_000.0 * TEST_ACCURACY);
/// ```
pub struct FibonacciGenerator {
    /// Seed of the generator
    seed: f64,
    /// Number of pseudo-random numbers to generate
    sample_size: usize,
    /// First primitive polynomial degree
    mu: usize,
    ///Second primitive polynomial degree
    nu: usize,
    /// Mod of the linear congruential generator
    m: f64,
    /// Multiplierof the linear congruential generator
    a: f64,
    /// Increment of the linear congruential generator
    b: f64,
}

impl FibonacciGenerator {
    /// Creates a new `FibonacciGenerator` instance.
    /// 
    /// # Input
    /// - `seed`: Seed of the generator
    /// - `sample_size`: Number of pseudo-random numbers to generate
    /// - `mu`: First primitive polynomial degree
    /// - `nu`: Second primitive polynomial degree
    /// - `m`: Mod of the linear congruential generator
    /// - `a`: Multiplierof the linear congruential generator
    /// - `b`: Increment of the linear congruential generator
    pub fn build(seed: u32, sample_size: usize, mu: usize, nu: usize, m: u32, a: u32, b: u32) -> Result<Self, DigiFiError> {
        if nu <= mu {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "Parameter `nu` must be larger than parameter `mu`".to_owned(),
            })
        }
        Ok(FibonacciGenerator { seed: seed as f64, sample_size, mu, nu, m: m as f64, a: a as f64, b: b as f64, })
    }
}

impl ErrorTitle for FibonacciGenerator {
    fn error_title() -> String {
        String::from("Fibonacci Generator")
    }
}

impl RandomGenerator<FibonacciGenerator> for FibonacciGenerator {
    /// Creates a new `FibonacciGenerator` instance with random parameters.
    /// 
    /// # Input
    /// - `sample_size`: Number of pseudo-random numbers to generate
    fn new_shuffle(sample_size: usize) -> Result<Self, DigiFiError> {
        let seed: f64 = generate_seed()?;
        let m: f64 = seed * 1_234.0;
        let a: f64 = seed / 10.0;
        let b: f64 = m / 10.0;
        Ok(FibonacciGenerator { seed, sample_size, mu: 5, nu: 17, m, a, b })
    }

    /// Array of pseudo-random generated numbers based on Fibonacci Generator.
    /// 
    /// # Output
    /// - An array pseudo-random numberss following Uniform distribution
    fn generate(&self) -> Result<Array1<f64>, DigiFiError> {
        let (_, mut u) = (0..self.sample_size).into_iter()
            .fold((self.seed, Vec::with_capacity(self.sample_size)), |(prev, mut u), _| {
                let next: f64 = (self.a * prev + self.b) % self.m;
                u.push(next);
                (next, u)
            } );
        let _ = ((self.nu + 1)..self.sample_size).into_iter().map(|i| { u[i] = (u[i - self.nu] + u[i - self.mu]) % self.m; } );
        Ok(Array1::from_vec(u) / self.m)
    }
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::utilities::TEST_ACCURACY;
    use crate::random_generators::RandomGenerator;

    #[test]
    fn unit_test_linear_congruential_generator() -> () {
        use crate::random_generators::uniform_generators::LinearCongruentialGenerator;
        let lcg: LinearCongruentialGenerator = LinearCongruentialGenerator::new(12_345, 1_000_000, 244_944, 1_597, 51_749);
        let sample: Array1<f64> = lcg.generate().unwrap();
        assert_eq!(sample.len(), 1_000_000);
        println!("{}", sample.mean().unwrap());
        assert!((sample.mean().unwrap() - 0.5).abs() < 1_000.0 * TEST_ACCURACY)
    }

    #[test]
    fn unit_test_fibonacci_generator() -> () {
        use crate::random_generators::uniform_generators::FibonacciGenerator;
        // Input seed
        let fg: FibonacciGenerator = FibonacciGenerator::build(12_345, 1_000_000, 5, 17, 714_025, 1_366, 150_889).unwrap();
        let sample: Array1<f64> = fg.generate().unwrap();
        assert_eq!(sample.len(), 1_000_000);
        assert!((sample.mean().unwrap() - 0.5).abs() < 10_000.0 * TEST_ACCURACY);
        // Auto-generated seed
        let fg: FibonacciGenerator = FibonacciGenerator::new_shuffle(1_000_000).unwrap();
        let sample: Array1<f64> = fg.generate().unwrap();
        assert_eq!(sample.len(), 1_000_000);
        println!("{}", sample.mean().unwrap());
        assert!((sample.mean().unwrap() - 0.5).abs() < 100_000.0 * TEST_ACCURACY);
    }
}