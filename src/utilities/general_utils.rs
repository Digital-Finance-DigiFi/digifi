pub fn compare_vec_len<T>(vec_1: &Vec<T>, vec_2: &Vec<T>, vec_1_name: &str, vec_2_name: &str) {
    if vec_1.len() != vec_2.len() {
        panic!("The length of {} and {} do not coincide.", vec_1_name, vec_2_name);
    }
}


pub fn rolling_window(vec: &Vec<f64>, window: usize) -> Vec<&[f64]> {
    vec.windows(window).collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_test_compare_vec_len() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        compare_vec_len(&a, &b, "a", "b");
    }

    #[test]
    fn unit_test_rolling_window() {
        let vec = vec![1.0, 2.0, 3.0, 4.0];
        let result = rolling_window(&vec, 3);
        println!("{:?}", [1.0, 2.0]);
        // assert_eq!(result, vec![&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0]]);
        assert_eq!(result, vec![&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0]]);
    }
}