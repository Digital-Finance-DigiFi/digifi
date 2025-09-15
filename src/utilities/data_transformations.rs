use std::cmp::Ordering;
use ndarray::{Array1, Axis, s};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;
use crate::utilities::NUMERICAL_CORRECTION;
use crate::statistics::n_choose_r;


/// General form of the min-max normalization where the array is normalized to the interval `\[a,b\]`.
///
/// # Input
/// - `x`: An array to normalize
/// - `a`: Lower bound for the normalized array
/// - `b`: Upper bound for the normalized array
///
/// # Output
/// - Normalized array `x`
///
/// # LaTeX Formula
/// - x_{norm} = a + \\frac{(x-min(x))(b-a)}{max(x)-min(x)}
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::min_max_scaling;
///
/// let x: Array1<f64> = Array1::from_vec(vec![-10.0, -4.0, 5.0]);
///
/// let x_norm: Array1<f64> = min_max_scaling(x, 0.0, 1.0);
///
/// assert_eq!(x_norm, Array1::from_vec(vec![0.0, 0.4, 1.0]));
/// ```
pub fn min_max_scaling(x: Array1<f64>, a: f64, b: f64) -> Array1<f64> {
    let first_value: f64 = x[0];
    let (min, max): (f64, f64) = x.iter().fold((first_value, first_value), |(min, max), curr| {
        if *curr < min { (*curr, max) } else if max < *curr { (min, *curr) } else { (min, max) }
    } );
    if min == max {
        // Default array to prevent NaN values
        Array1::from_vec(vec![1.0; x.len()])
    } else {
        a + ((x - min) * (b - a)) / (max - min)
    }
}


/// Percent change of values in the time series compared to the previous values.
/// 
/// # Input
/// - `x`: Time series of values
/// 
/// # Output
/// - Percent change array
/// 
/// # LaTeX Formula
/// - R_{t} = \\frac{S_{t} - S_{t-1}}{S_{t-1}}
/// 
/// # Links
/// - Wikipedia: N/A
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::{Array1, array};
/// use digifi::utilities::{TEST_ACCURACY, percent_change};
///
/// let x: Array1<f64> = array![1.0, 2.0, 1.6, 1.6, 0.0, 0.0];
/// 
/// assert!((percent_change(&x) - array![1.0, -0.2, 0.0, -1.0, 0.0]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub fn percent_change(x: &Array1<f64>) -> Array1<f64> {
    let result: Vec<f64> = x.slice(s![1..(x.len())]).iter().zip(x.slice(s![0..(x.len()-1)]).iter()).map(|(final_, initial)| {
        if initial == &0.0 && final_ == &0.0 {
            0.0
        } else if initial != &0.0 && final_ != &0.0 {
            (final_ / initial) - 1.0
        } else {
            if initial < final_ { 1.0 } else { -1.0 }
        }
    } ).collect();
    Array1::from_vec(result)
}


/// Log-return transformation of values in the time series.
/// 
/// # Input
/// -`x`: Time series of values
/// 
/// # Output
/// - Log change array
/// 
/// # LaTeX Formula
/// - r_{t} = ln(S_{t}) - ln(S_{t-1})
/// 
/// # Links
/// - Wikipedia: N/A
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::{Array1, array};
/// use digifi::utilities::{TEST_ACCURACY, log_return_transformation};
///
/// let x: Array1<f64> = array![1.0, 2.0, 1.6, 1.6, 0.0, 0.0];
/// 
/// assert!((log_return_transformation(&x) - array![0.6931471805599453, -0.2231435513142097, 0.0, -32.236990899346836, 0.0]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub fn log_return_transformation(x: &Array1<f64>) -> Array1<f64> {
    let result: Vec<f64> = x.slice(s![1..(x.len())]).iter().zip(x.slice(s![0..(x.len()-1)]).iter()).map(|(final_, initial)| {
        let percent_change: f64 = if initial == &0.0 && final_ == &0.0 {
            0.0
        } else if initial != &0.0 && final_ != &0.0 {
            (final_ / initial) - 1.0
        } else {
            if initial < final_ { 1.0 } else { -1.0 }
        };
        let v: f64 = match percent_change { -1.0 => percent_change + NUMERICAL_CORRECTION, _ => percent_change, };
        (v + 1.0).ln()
    } ).collect();
    Array1::from_vec(result)
}


/// Differencing in statistics is a transformation applied to a non-stationary time-series in order to make
/// it trend stationary (i.e., stationary in the mean sense), by removing or subtracting the trend or non-constant mean.
/// 
/// # Input
/// - `v`: Time series to compute the differencing time series from.
/// - `n`: Order of differencing
/// 
/// # LaTeX
/// - y^{n}_{t} = \\sum^{n}_{i=0}(-1)^{i}{n\\choose i}y_{t-i}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing>
/// - Original Source: N/A
/// 
/// # Example
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::differencing;
/// 
/// let v: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// 
/// assert_eq!(differencing(&v, 1).unwrap(), Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]));
/// assert_eq!(differencing(&v, 2).unwrap(), Array1::from_vec(vec![0.0, 0.0, 0.0]));
/// assert_eq!(differencing(&v, 3).unwrap(), Array1::from_vec(vec![0.0, 0.0]));
/// ```
pub fn differencing(v: &Array1<f64>, n: usize) -> Result<Array1<f64>, DigiFiError> {
    let v_len: usize = v.len();
    if v_len < n {
        return Err(DigiFiError::Other { title: "Differencing".to_owned(), details: "The `n` must be smaller than the length of the array `v`.".to_owned(), })
    }
    let mut diff: Vec<f64> = Vec::with_capacity(v_len - n);
    let n_u128: u128 = n as u128;
    // Iterate over slices of the time series to compute the time series of differenced values
    for j in (n..v_len).rev() {
        if (j as i32 - n as i32) < 0 {
            continue;
        }
        // Cut array and reverse the order of elements so that they are in time descending order (i.e., t, t-1, etc.)
        let x: Vec<f64> = v.slice(s![(j - n)..(j + 1)]).into_iter().map(|v_| *v_ ).rev().collect();
        let mut d: f64 = 0.0;
        // Apply differencing (via Binomial expansions for the specific `n`)
        let mut sign: f64 = 1.0;
        for i in 0..(n + 1) {
            d += sign * (n_choose_r(n_u128, i as u128)? as f64) * x[i];
            sign *= -1.0;
        }
        diff.insert(0, d);
    }
    Ok(Array1::from_vec(diff))
}


/// Data transformation where numerical values are replaced by their rank when the data is sorted.
/// 
/// # Input
/// -`x`: Array of values
/// 
/// # Output
/// - Array of ranks
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Ranking_(statistics)>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::rank_transformation;
///
/// let x: Array1<f64> = Array1::from_vec(vec![1.0, 3.0, 5.0, 2.0, 5.0, 6.0]);
/// 
/// let x_transformed: Array1<f64> = Array1::from_vec(vec![1.0, 3.0, 4.0, 2.0, 4.0, 5.0]);
/// 
/// assert_eq!(x_transformed, rank_transformation(&x));
/// ```
pub fn rank_transformation(x: &Array1<f64>) -> Array1<f64> {
    let mut ranked: Vec<(usize, f64)> = x.iter().enumerate().map(|(i, val)| { (i, *val) } ).collect();
    ranked.sort_by(|a, b| { a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal) } );
    let mut result: Vec<f64> = vec![0.0; x.len()];
    let mut current_rank: f64 = 1.0;
    for (i, pair) in ranked.iter().enumerate() {
        if 0 < i && ranked[i].1 != ranked[i-1].1 {
            current_rank += 1.0;
        }
        result[pair.0] = current_rank;
    }
    Array1::from_vec(result)
}


/// General form of the unit vector normalization using the p-norm.
///
/// # Input
/// - `x`: An array to normalize
/// - `p`: Order of the p-nrom
///
/// # Output
/// - Normalized array `x`
///
/// # LaTeX Formula
/// - x_{norm} = (\\frac{x_{1}}{(\\lvert x_{1}\\rvert^{p} + ... + \\lvert x_{n}\\rvert^{p})^{\\frac{1}{p}}}, ..., \\frac{x_{n}}{(\\lvert x_{1}\\rvert^{p} + ... + \\lvert x_{n}\\rvert^{p})^{\\frac{1}{p}}})
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Feature_scaling#Unit_vector_normalization>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, unit_vector_normalization};
///
/// let x: Array1<f64> = Array1::from_vec(vec![-15.0, 3.0, 5.0]);
///
/// let x_norm: Array1<f64> = unit_vector_normalization(x, 2);
///
/// assert!(((&x_norm * &x_norm).sum() - 1.0).abs() < TEST_ACCURACY)
/// ```
pub fn unit_vector_normalization(x: Array1<f64>, p: usize) -> Array1<f64> {
    let p_: i32 = p as i32;
    let one_over_p: f64 = 1.0 / p_ as f64;
    let norm: f64 = x.iter().fold(0.0, |prev, curr| { prev + curr.abs().powi(p_) } ).powf(one_over_p);
    x / norm
}


#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Type of data transformation that can be applied to an array of data.
pub enum TransformationType {
    MinMaxScaling {a: f64, b: f64},
    PercentChange,
    LogReturnTransformation,
    Differencing { order: usize },
    RankTransformation,
    UnitVectorNormalization { p: usize },
    #[default]
    No,
}

impl TransformationType {

    /// Performs specified data transformation on the data passed in.
    /// 
    /// # Input
    /// - `data`: Array of data to be transformed
    /// 
    /// # Output
    /// - Transformed data array
    pub fn transformation(&self, data: &Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
        let transformed_data: Array1<f64> = match self {
            TransformationType::MinMaxScaling { a, b } => min_max_scaling(data.clone(), *a, *b),
            TransformationType::PercentChange => percent_change(data),
            TransformationType::LogReturnTransformation => log_return_transformation(data),
            TransformationType::Differencing { order } => differencing(data, *order)?,
            TransformationType::RankTransformation => rank_transformation(data),
            TransformationType::UnitVectorNormalization { p } => unit_vector_normalization(data.clone(), *p),
            TransformationType::No => data.clone(),
        };
        Ok(transformed_data)
    }
}


/// Struct that collect all data transformation utilities under one namespace.
pub struct DataTransformations;

impl DataTransformations {

    /// Performs specified data transformation on the data passed in.
    /// 
    /// # Input
    /// - `data`: Array of data to be transformed
    /// - `transformation_type`: Type of data transformation to apply to the data
    /// 
    /// # Output
    /// - Transformed data array
    pub fn transformation(data: &Array1<f64>, transformation_type: &TransformationType) -> Result<Array1<f64>, DigiFiError> {
        transformation_type.transformation(data)
    }

    /// Utility that crops two arrays so that they are of the same length.
    /// 
    /// Note: This utility is useful to apply after the data transforamtion has been performed on two or more arrays since some transformations return
    /// less data points than the origianl arrays before the transformation (e.g., Percent Change, Log Return, etc.).
    /// 
    /// # Input
    /// - `v1`: Transformed data series
    /// - `v2`: Transformed data series
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use ndarray::Array1;
    /// use digifi::utilities::DataTransformations;
    /// 
    /// let mut x: Array1<f64> = Array1::from_vec(vec![-15.0, 3.0, 5.0, 16.0, 43.0]);
    /// let mut y: Array1<f64> = Array1::from_vec(vec![26.0, 1.0, 5.0, -9.0, 13.0, 45.0, 12.0]);
    /// DataTransformations::sync_transformations(&mut x, &mut y);
    /// 
    /// assert_eq!(x.len(), 5);
    /// assert_eq!(y.len(), 5);
    /// assert_eq!(y, Array1::from_vec(vec![5.0, -9.0, 13.0, 45.0, 12.0]));
    /// ```
    pub fn sync_transformations(v1: &mut Array1<f64>, v2: &mut Array1<f64>) -> () {
        let (len_diff, large_v) = if v1.len() < v2.len() {
            (v2.len().checked_sub(v1.len()).unwrap_or(0), v2)
        } else if v2.len() < v1.len() {
            (v1.len().checked_sub(v2.len()).unwrap_or(0), v1)
        } else {
            return ();
        };
        for _ in 0..len_diff {
            large_v.remove_index(Axis(0), 0);
        }
    }

    /// Utility that crops multiple arrays so that they are of the same length.
    /// 
    /// Note: This utility is useful to apply after the data transforamtion has been performed on two or more arrays since some transformations return
    /// less data points than the origianl arrays before the transformation (e.g., Percent Change, Log Return, etc.).
    /// 
    /// # Input
    /// - `data`: Collection of data arrays to be transformed
    /// - `transformation_types`: Transformation type that will be applied to data arrays
    /// 
    /// # Errors
    /// - Returns an error if the length of `data` doe not match the length of `transformation_types`.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use ndarray::Array1;
    /// use digifi::utilities::{TransformationType, DataTransformations};
    /// 
    /// let x1: Array1<f64> = Array1::from_vec(vec![-15.0, 3.0, 5.0, 16.0, 43.0]);
    /// let x2: Array1<f64> = Array1::from_vec(vec![26.0, 1.0, 5.0, -9.0, 13.0, 45.0, 12.0]);
    /// let x3: Array1<f64> = Array1::from_vec(vec![36.0, 18.0, 5.0, 2.0, 60.0]);
    /// let data: Vec<Array1<f64>> = vec![x1, x2, x3];
    /// 
    /// let tt: TransformationType = TransformationType::No;
    /// let transformation_types: Vec<TransformationType> = vec![tt.clone(), tt.clone(), tt.clone()];
    /// 
    /// let transformed_data: Vec<Array1<f64>> = DataTransformations::transformation_multiple(&data, &transformation_types).unwrap();
    /// 
    /// assert_eq!(transformed_data.len(), 3);
    /// assert_eq!(transformed_data[0].len(), 5);
    /// assert_eq!(transformed_data[1].len(), 5);
    /// assert_eq!(transformed_data[2].len(), 5);
    /// assert_eq!(transformed_data[1], Array1::from_vec(vec![5.0, -9.0, 13.0, 45.0, 12.0]));
    /// ```
    pub fn transformation_multiple(data: &Vec<Array1<f64>>, transformation_types: &Vec<TransformationType>) -> Result<Vec<Array1<f64>>, DigiFiError> {
        // Input validation
        if data.len() != transformation_types.len() {
            return Err(DigiFiError::UnmatchingLength { array_1: "data".to_owned(), array_2: "transformation_types".to_owned(), });
        }
        if data.is_empty() { return Ok(vec![]) }
        // Data transformations
        let mut transformed_data: Vec<Array1<f64>> = vec![];
        for (d, t) in data.iter().zip(transformation_types.iter()) {
            transformed_data.push(t.transformation(d)?);
        }
        // Sync transformations
        // Find benchmark series with shortest length
        let shortest_series_index: usize = {
            let mut shortest_index: usize = 0;
            let mut shortest_len: usize = transformed_data[shortest_index].len();
            for i in 0..transformed_data.len() {
                if transformed_data[i].len() < shortest_len {
                    shortest_index = i;
                    shortest_len = transformed_data[i].len();
                }
            }
            shortest_index
        };
        // Use shortest series as a benchmark to readjust the lengths of other transformed series
        for i in 0..transformed_data.len() {
            if i != shortest_series_index {
                let len_diff: usize = transformed_data[i].len().checked_sub(transformed_data[shortest_series_index].len()).unwrap_or(0);
                for _ in 0..len_diff { transformed_data[i].remove_index(Axis(0), 0); }
            }
        }
        Ok(transformed_data)
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_min_max_scaling() -> () {
        use crate::utilities::data_transformations::min_max_scaling;
        let x: Array1<f64> = Array1::from_vec(vec![-10.0, -4.0, 5.0]);
        let x_norm: Array1<f64> = min_max_scaling(x, 0.0, 1.0);
        assert_eq!(x_norm, Array1::from_vec(vec![0.0, 0.4, 1.0]));
    }

    #[test]
    fn unit_test_percent_change() -> () {
        use crate::utilities::data_transformations::percent_change;
        let x: Array1<f64> = array![1.0, 2.0, 1.6, 1.6, 0.0, 0.0];
        assert!((percent_change(&x) - array![1.0, -0.2, 0.0, -1.0, 0.0]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_log_return_transformation() -> () {
        use crate::utilities::data_transformations::log_return_transformation;
        let x: Array1<f64> = array![1.0, 2.0, 1.6, 1.6, 0.0, 0.0];
        assert!((log_return_transformation(&x) - array![0.6931471805599453, -0.2231435513142097, 0.0, -32.236990899346836, 0.0]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_differencing() -> () {
        use crate::utilities::data_transformations::differencing;
        let v: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(differencing(&v, 1).unwrap(), Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]));
        assert_eq!(differencing(&v, 2).unwrap(), Array1::from_vec(vec![0.0, 0.0, 0.0]));
        assert_eq!(differencing(&v, 3).unwrap(), Array1::from_vec(vec![0.0, 0.0]));
    }

    #[test]
    fn unit_test_rank_transformation() -> () {
        use crate::utilities::data_transformations::rank_transformation;
        let x: Array1<f64> = Array1::from_vec(vec![1.0, 3.0, 5.0, 2.0, 5.0, 6.0]);
        let x_transformed: Array1<f64> = Array1::from_vec(vec![1.0, 3.0, 4.0, 2.0, 4.0, 5.0]);
        assert_eq!(x_transformed, rank_transformation(&x));
    }

    #[test]
    fn unit_test_unit_vector_normalization() -> () {
        use crate::utilities::data_transformations::unit_vector_normalization;
        let x: Array1<f64> = Array1::from_vec(vec![-15.0, 3.0, 5.0]);
        let x_norm: Array1<f64> = unit_vector_normalization(x, 2);
        assert!(((&x_norm * &x_norm).sum() - 1.0).abs() < TEST_ACCURACY)
    }

    #[test]
    fn unit_test_sync_transformations() -> () {
        use crate::utilities::data_transformations::DataTransformations;
        let mut x: Array1<f64> = Array1::from_vec(vec![-15.0, 3.0, 5.0, 16.0, 43.0]);
        let mut y: Array1<f64> = Array1::from_vec(vec![26.0, 1.0, 5.0, -9.0, 13.0, 45.0, 12.0]);
        DataTransformations::sync_transformations(&mut x, &mut y);
        assert_eq!(x.len(), 5);
        assert_eq!(y.len(), 5);
        assert_eq!(y, Array1::from_vec(vec![5.0, -9.0, 13.0, 45.0, 12.0]));
    }

    #[test]
    fn unit_test_transformation_multiple() -> () {
        use crate::utilities::data_transformations::{TransformationType, DataTransformations};
        let x1: Array1<f64> = Array1::from_vec(vec![-15.0, 3.0, 5.0, 16.0, 43.0]);
        let x2: Array1<f64> = Array1::from_vec(vec![26.0, 1.0, 5.0, -9.0, 13.0, 45.0, 12.0]);
        let x3: Array1<f64> = Array1::from_vec(vec![36.0, 18.0, 5.0, 2.0, 60.0]);
        let data: Vec<Array1<f64>> = vec![x1, x2, x3];
        let tt: TransformationType = TransformationType::No;
        let transformation_types: Vec<TransformationType> = vec![tt.clone(), tt.clone(), tt.clone()];
        let transformed_data: Vec<Array1<f64>> = DataTransformations::transformation_multiple(&data, &transformation_types).unwrap();
        assert_eq!(transformed_data.len(), 3);
        assert_eq!(transformed_data[0].len(), 5);
        assert_eq!(transformed_data[1].len(), 5);
        assert_eq!(transformed_data[2].len(), 5);
        assert_eq!(transformed_data[1], Array1::from_vec(vec![5.0, -9.0, 13.0, 45.0, 12.0]));
    }
}