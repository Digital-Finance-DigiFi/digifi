pub fn factorial(n: u128) -> u128 {
    (1..=n).product()
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
    fn maths_utils_tests_message() {
        println!("All unit tests in module `maths_utils` have been completed.");
    }
}