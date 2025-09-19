//! # Utilities
//! 
//! Contains general utility tools such as different data transformation functions, loss functions, numerical optimization engine, minimal spanning tree,
//! and utilities for doing time-value of money calculations.


// Re-Exports
pub use self::maths_utils::{factorial, rising_factorial, erf, erfinv, euclidean_distance, derivative, definite_integral};
pub use self::minimal_spanning_tree::{MSTDistance, MSTNode, MSTEdge, MST};
pub use self::numerical_engines::nelder_mead;
pub use self::loss_functions::{LossFunction, MAE, MSE, SSE, StraddleLoss};
pub use self::data_transformations::{
    min_max_scaling, percent_change, log_return_transformation, differencing, rank_transformation, unit_vector_normalization,
    TransformationType, DataTransformations,
};
pub use self::time_value_utils::{
    CompoundingType, present_value, net_present_value, future_value, internal_rate_of_return, real_interest_rate,
    ptp_compounding_transformation, ptc_compounding_transformation, ctp_compounding_transformation, Compounding, forward_rate, Cashflow, Perpetuity, Annuity,
};
pub use self::feature_collection::FeatureCollection;
#[cfg(feature = "sample_data")]
pub use self::sample_data::SampleData;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};


pub mod maths_utils;
pub mod minimal_spanning_tree;
pub mod numerical_engines;
pub mod loss_functions;
pub mod data_transformations;
pub mod time_value_utils;
mod feature_collection;
#[cfg(feature = "sample_data")]
pub mod sample_data;


use ndarray::{Array1, Array2};
use nalgebra::DMatrix;
use crate::error::{DigiFiError, ErrorTitle};


pub const TEST_ACCURACY: f64 = 0.00000001;

/// Constant used for numerical corrections of values in order to make the numerical method well-defined
/// (i.e., Numerical integration upper and lower bounds, log returns)
pub const NUMERICAL_CORRECTION: f64 = 0.00000000000001;

pub const LARGE_TEXT_BREAK: &str = "--------------------------------------------------------------------------------------\n";

pub const SMALL_TEXT_BREAK: &str = "\t----------------------------------------------------\n";


#[derive(Clone, Debug)]
/// Type of parameter used in calculations.
pub enum ParameterType {
    Value { value: f64 },
    TimeSeries { values: Array1<f64> },
}

impl ParameterType {

    /// Converts `ParameterType` to `Array1`.
    /// 
    /// # Input
    /// - `len`: The length of the output array (Needed to create an array from single value, or to validate the length of time series array)
    /// 
    /// #  Errors
    /// - Returns an error if `values` inside `ParameterType::TimeSeries` variant is not of length `len`.
    pub fn into_array(self, len: usize) -> Result<Array1<f64>, DigiFiError> {
        match self {
            ParameterType::Value { value } => Ok(Array1::from_vec(vec![value; len])),
            ParameterType::TimeSeries { values } => {
                if values.len() != len {
                    return Err(DigiFiError::WrongLength { title: Self::error_title(), arg: "values".to_owned(), len, });
                }
                Ok(values)
            },
        }
    }
}

impl ErrorTitle for ParameterType {
    fn error_title() -> String {
        String::from("Parameter Type")
    }
}


#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Struct that stores an array of time steps.
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, Time};
/// let time_1: Time = Time::new_from_range(0.0, 1.0, 0.2);
/// let time_2: Time = Time::new(Array1::from_vec(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]));
/// 
/// assert!((time_1.time_array() - time_2.time_array()).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub struct Time(Array1<f64>);

impl Time {

    /// Creates a new instance of `Time`.
    /// 
    /// # Input
    /// - `time_array`: Array of time steps
    pub fn new(time_array: Array1<f64>) -> Self {
        Time(time_array)
    }

    /// Creates a range of time steps given the provided range definition.
    /// 
    /// # Input
    /// - `initial_time`: Initial time setp
    /// - `final_time`: Final time step (inclusive)
    /// - `dt`: Difference between consequtive time steps.
    pub fn new_from_range(initial_time: f64, final_time: f64, dt: f64) -> Self {
        Time(Array1::range(initial_time, final_time + dt, dt))
    }

    /// Returns the length of the time array.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns a reference to an array of time steps.
    /// 
    /// Note: This is a useful method for code readability.
    pub fn time_array(&self) -> &Array1<f64> {
        &self.0
    }

    /// Returns a mutable reference to an array of time steps.
    /// 
    /// Note: This is a useful method for code readability.
    pub fn time_array_mut(&mut self) -> &mut Array1<f64> {
        &mut self.0
    }
}


/// Asserts that the two iterators provided are of the same length.
/// 
/// # Input
/// - `iter_1`: First iterator
/// - `iter_2`: Second iterator
/// - `iter_1_name`: Name of the first iterator
/// - `iter_1_name`: Name of the second iterator
/// 
/// # Errors
/// - Returns an error if the length of iterators do not match.
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::compare_len;
///
/// let a: Vec<i32> = vec![1, 2, 3];
/// let b: Vec<i32> = vec![4, 5, 6];
///
/// compare_len(&a.iter(), &b.iter(), "a", "b").unwrap();
/// ```
pub fn compare_len<U: ExactSizeIterator, V: ExactSizeIterator>(iter_1: &U, iter_2: &V, iter_1_name: &str, iter_2_name: &str) -> Result<(), DigiFiError> {
    if iter_1.len() != iter_2.len() {
        return Err(DigiFiError::UnmatchingLength { array_1: iter_1_name.to_owned(), array_2: iter_2_name.to_owned(), });
    }
    Ok(())
}


/// Methods for converting matrices from ndarray to nalgebra and vice versa.
pub struct MatrixConversion;

impl MatrixConversion {

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
    /// let result: DMatrix<f64> = MatrixConversion::ndarray_to_nalgebra(matrix);
    /// 
    /// assert_eq!(result.row(0).len(), matrix_dim.1);
    /// assert_eq!(result.column(0).len(), matrix_dim.0);
    /// ```
    pub fn ndarray_to_nalgebra(matrix: Array2<f64>) -> DMatrix<f64> {
        let (n_rows, n_columns) = matrix.dim();
        let n_matrix: DMatrix<f64> = DMatrix::from_vec(n_columns, n_rows, matrix.clone().into_raw_vec_and_offset().0);
        n_matrix.transpose()
    }

    /// Converts nalgebra matrix to ndarray matrix
    /// 
    /// # Input
    /// - `matrix`: nalgebra matrix
    /// 
    /// # Ouput
    /// - ndarray matrix
    pub fn nalgebra_to_ndarray(matrix: DMatrix<f64>) -> Result<Array2<f64>, DigiFiError> {
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
        let time_1: Time = Time::new_from_range(0.0, 1.0, 0.2);
        let time_2: Time = Time::new(Array1::from_vec(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]));
        assert!((time_1.time_array() - time_2.time_array()).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_compare_len() -> () {
        use crate::utilities::compare_len;
        let a: Vec<i32> = vec![1, 2, 3];
        let b: Vec<i32> = vec![4, 5, 6];
        compare_len(&a.iter(), &b.iter(), "a", "b").unwrap();
    }

    #[test]
    fn unit_test_ndarray_to_nalgebra() -> () {
        use ndarray::arr2;
        use crate::utilities::MatrixConversion;
        let matrix: Array2<f64> = arr2(&[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]);
        let matrix_dim: (usize, usize) = matrix.dim();
        let result: DMatrix<f64> = MatrixConversion::ndarray_to_nalgebra(matrix);
        assert_eq!(result.row(0).len(), matrix_dim.1);
        assert_eq!(result.column(0).len(), matrix_dim.0);
    }
}