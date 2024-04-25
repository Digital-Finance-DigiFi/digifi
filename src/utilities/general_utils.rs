pub fn compare_vec_len<T>(vec1: &[T], vec2: &[T], vec1_name: &str, vec2_name: &str) {
    if vec1.len() != vec2.len() {
        panic!("The length of {} and {} do not coincide.", vec1_name, vec2_name);
    }
}

pub fn rolling_window(vec: &[f64], window: usize) -> Vec<&[f64]> {
    vec.windows(window).collect()
}

/// Data class validation doesn't make sense to do as there are no classes in the first place

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
        assert_eq!(result, vec! [&[1.0, 2.0, 3.0][..], &[2.0, 3.0, 4.0][..]]);
    }
}