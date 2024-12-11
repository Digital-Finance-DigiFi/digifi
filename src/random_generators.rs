// Re-Exports
pub use self::generator_algorithms::{accept_reject, inverse_transform, box_muller, marsaglia, ziggurat};
pub use self::uniform_generators::{LinearCongruentialGenerator, FibonacciGenerator};
pub use self::standard_normal_generators::{
    StandardNormalAcceptReject, StandardNormalInverseTransform, StandardNormalBoxMuller, StandardNormalMarsaglia, StandardNormalZiggurat,
};


pub mod generator_algorithms;
pub mod standard_normal_generators;
pub mod uniform_generators;


use std::io::{Error, ErrorKind};
use std::time::{SystemTime, UNIX_EPOCH};
use ndarray::Array1;


pub trait RandomGenerator<T> {
    fn new_shuffle(sample_size: usize) -> Result<T, Error>;

    fn generate(&self) -> Result<Array1<f64>, Error>;
}


/// # Description
/// Generates a seed from nanosecond timestamp of the system.
///
/// # Examples
///
/// ```rust
/// use digifi::random_generators::generate_seed;
///
/// let seed_1: u32 = generate_seed().unwrap();
/// let seed_2: u32 = generate_seed().unwrap();
///
/// assert!(seed_1 != seed_2);
/// ```
pub fn generate_seed () -> Result<u32, Error> {
    let start: SystemTime = SystemTime::now();
    let delta: f64;
    match start.duration_since(UNIX_EPOCH) {
        Ok(duration) => {
            delta = duration.subsec_nanos() as f64
        },
        Err(e) => {
            return Err(Error::new(ErrorKind::Other, e.to_string()));
        },
    }
    // Drop the first two digits and last two digits from delta
    let delta: u32 = (delta / 100.0) as u32;
    let remainder: u32 = delta.rem_euclid(100_000);
    let big_digit_number: u32 = delta - remainder;
    Ok(delta - big_digit_number)
}


#[cfg(test)]
mod tests {
    
    #[test]
    fn unit_test_generate_seed() -> () {
        use crate::random_generators::generate_seed;
        let seed_1: u32 = generate_seed().unwrap();
        let seed_2: u32 = generate_seed().unwrap();
        assert!(seed_1 != seed_2);
    }
}