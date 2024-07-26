pub mod generator_algorithms;
pub mod standard_normal_generators;
pub mod uniform_generators;


use std::io::Error;
use std::time::{SystemTime, UNIX_EPOCH};
use ndarray::Array1;


pub trait RandomGenerator {
    fn new_shuffle(sample_size: usize) -> Self;

    fn generate(&self) -> Result<Array1<f64>, Error>;
}


/// # Description
/// Generates a seed from nanosecond timestamp of the system.
pub fn generate_seed () -> u32 {
    let start: SystemTime = SystemTime::now();
    let delta: f64 = start.duration_since(UNIX_EPOCH).expect("Could not compute duration since last unix epoch.").subsec_nanos() as f64;
    (delta / (10_000.0)) as u32
}


#[cfg(test)]
mod tests {
    
    #[test]
    fn unit_test_generate_seed() -> () {
        use crate::random_generators::generate_seed;
        let seed_1: u32 = generate_seed();
        let seed_2: u32 = generate_seed();
        assert!(seed_1 != seed_2);
    }
}