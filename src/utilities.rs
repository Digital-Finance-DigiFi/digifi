// Re-Exports
pub use self::loss_functions::{LossFunction, MAE, MSE, SSE, StraddleLoss};
pub use self::maths_utils::{FunctionEvalMethod, factorial, rising_factorial, erf, erfinv, euclidean_distance, differencing, derivative, definite_integral};
pub use self::minimal_spanning_tree::{MSTDistance, MSTNode, MSTEdge, MST};
pub use self::numerical_engines::nelder_mead;
#[cfg(feature = "sample_data")]
pub use self::sample_data::SampleData;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
pub use self::time_value_utils::{
    CompoundingType, present_value, net_present_value, future_value, internal_rate_of_return, real_interest_rate,
    ptp_compounding_transformation, ptc_compounding_transformation, ctp_compounding_transformation, Compounding, forward_rate, Cashflow, Perpetuity, Annuity,
};


pub mod maths_utils;
pub mod minimal_spanning_tree;
pub mod numerical_engines;
pub mod loss_functions;
pub mod time_value_utils;
#[cfg(feature = "sample_data")]
pub mod sample_data;


use ndarray::{Array1, Array2};
use nalgebra::DMatrix;
use crate::error::DigiFiError;


pub const TEST_ACCURACY: f64 = 0.00000001;


#[derive(Debug, Clone)]
/// # Description
/// Type of parameter used in calculations.
pub enum ParameterType {
    Value { value: f64 },
    TimeSeries { values: Array1<f64> },
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Struct for generating time array.
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, Time};
/// let time_1: Time = Time::Range { initial_time: 0.0, final_time: 1.0, time_step: 0.2 };
/// let time_2: Time = Time::Sequence { times: Array1::from_vec(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) };
///
/// let time_array_1: Array1<f64> = time_1.get_time_array();
/// let time_array_2: Array1<f64> = time_2.get_time_array();
/// 
/// assert!((time_array_1 - time_array_2).sum().abs() < TEST_ACCURACY);
/// ```
pub enum Time {
    /// Creates a range of time steps given the provided settings.
    Range { initial_time: f64, final_time: f64, time_step: f64 },
    /// Uses provided array as the space of time steps.
    Sequence { times: Array1<f64> },
}

impl Time {
    /// # Description
    /// Generates the time array from the provided settings.
    pub fn get_time_array(&self) -> Array1<f64> {
        match self {
            Time::Range { initial_time, final_time, time_step } => {
                Array1::range(*initial_time, *final_time + *time_step, *time_step)
            },
            Time::Sequence { times } => {
                times.clone()
            },
        }
    }
}


/// # Description
/// Asserts that the two arrays provided are of the same length, while also verifying that both arrays are of np.ndarray type.
/// 
/// # Input
/// - `array_1`: First array
/// - `array_2`: Second array
/// - `array_1_name`: Name of the first array, which will be printed in case of an error
/// - `array_1_name`: Name of the second array, which will be printed in case of an error
/// 
/// # Errors
/// - Returns an error if the length of arrays do not match.
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::compare_array_len;
///
/// let a: Array1<i32> = Array1::from_vec(vec![1, 2, 3]);
/// let b: Array1<i32> = Array1::from_vec(vec![4, 5, 6]);
///
/// compare_array_len(&a, &b, "a", "b").unwrap();
/// ```
pub fn compare_array_len<T>(array_1: &Array1<T>, array_2: &Array1<T>, array_1_name: &str, array_2_name: &str) -> Result<(), DigiFiError> {
    if array_1.len() != array_2.len() {
        return Err(DigiFiError::UnmatchingLength { array_1: array_1_name.to_owned(), array_2: array_2_name.to_owned(), });
    }
    Ok(())
}


/// # Description
/// Methods for converting matrices from ndarray to nalgebra and vice versa.
pub struct MatrixConversion;

impl MatrixConversion {

    /// # Description
    /// Converts ndarray matrix to nalgebra matrix
    /// 
    /// # Input
    /// - `matrix`: ndarray matrix
    /// 
    /// # Output
    /// - nalgebra matrix
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{arr2, Array2};
    /// use nalgebra::DMatrix;
    /// use digifi::utilities::MatrixConversion;
    ///
    /// let matrix: Array2<f64> = arr2(&[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]);
    /// let matrix_dim: (usize, usize) = matrix.dim();
    /// let result: DMatrix<f64> = MatrixConversion::ndarray_to_nalgebra(&matrix);
    /// assert_eq!(result.row(0).len(), matrix_dim.1);
    /// assert_eq!(result.column(0).len(), matrix_dim.0);
    /// ```
    pub fn ndarray_to_nalgebra(matrix: &Array2<f64>) -> DMatrix<f64> {
        let (n_rows, n_columns) = matrix.dim();
        let n_matrix: DMatrix<f64> = DMatrix::from_vec(n_columns, n_rows, matrix.clone().into_raw_vec_and_offset().0);
        n_matrix.transpose()
    }

    /// # Description
    /// Converts nalgebra matrix to ndarray matrix
    /// 
    /// # Input
    /// - `matrix`: nalgebra matrix
    /// 
    /// # Ouput
    /// - ndarray matrix
    pub fn nalgebra_to_ndarray(matrix: &DMatrix<f64>) -> Result<Array2<f64>, DigiFiError> {
        let dim = (matrix.row(0).len(), matrix.column(0).len());
        Ok(Array2::from_shape_vec(dim, matrix.as_slice().to_vec())?)
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use nalgebra::DMatrix;
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_time_struct() -> () {
        use crate::utilities::Time;
        let time_1: Time = Time::Range { initial_time: 0.0, final_time: 1.0, time_step: 0.2 };
        let time_2: Time = Time::Sequence { times: Array1::from_vec(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) };
        let time_array_1: Array1<f64> = time_1.get_time_array();
        let time_array_2: Array1<f64> = time_2.get_time_array();
        assert!((time_array_1 - time_array_2).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_compare_vec_len() -> () {
        use crate::utilities::compare_array_len;
        let a: Array1<i32> = Array1::from_vec(vec![1, 2, 3]);
        let b: Array1<i32> = Array1::from_vec(vec![4, 5, 6]);
        compare_array_len(&a, &b, "a", "b").unwrap();
    }

    #[test]
    fn unit_test_ndarray_to_nalgebra() -> () {
        use ndarray::arr2;
        use crate::utilities::MatrixConversion;
        let matrix: Array2<f64> = arr2(&[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]);
        let matrix_dim: (usize, usize) = matrix.dim();
        let result: DMatrix<f64> = MatrixConversion::ndarray_to_nalgebra(&matrix);
        assert_eq!(result.row(0).len(), matrix_dim.1);
        assert_eq!(result.column(0).len(), matrix_dim.0);
    }
}