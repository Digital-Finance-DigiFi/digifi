pub fn factorial(n: u128) -> u128 {
    (1..=n).product()
}


pub fn n_choose_r(n: u128, r: u128) -> u128 {
    if n < r {
        panic!("The value of variable n must be larger or equal to the value of variable r");
    }
    factorial(n) / (factorial(n - r) * factorial(r))
}

pub fn erf(x: f64, n_terms: usize) -> f64 {
    let mut total = 0.0;
    let mut sign = 1.0;
    for n in 0..n_terms {
        let exp = (2 * n + 1).try_into().unwrap(); // safely convert usize to i32, panic if it doesn't fit
        total += sign * x.powi(exp) / (factorial(n as u128) as f64 * (2 * n + 1) as f64);
        sign *= -1.0; // Flip the sign for the next term
    }
    (2.0 / f64::sqrt(std::f64::consts::PI)) * total
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_test_factorial() {
        assert_eq!(3628800, factorial(10));
        assert_eq!(1, factorial(0));
    }

    #[test]
    fn unit_test_n_choose_r() {
        assert_eq!(10, n_choose_r(5, 2));
    }

    #[test]
    fn unit_test_erf() {
        let approximation = erf(1.0, 10);
        assert!((approximation - 0.8427007929497149).abs() < 0.01); // Allow some margin
    }

    #[test]
    fn maths_utils_tests_message() {
        println!("All unit tests in module `maths_utils` have been completed.");
    }
}