use std::io::Error;
use ndarray::Array1;
use crate::utilities::compare_array_len;


/// # Description
/// Measures an error between paired observations (Usually, empirical observations vs simulated observations).
/// 
/// # Input
/// - observed_values: Array of observed/empirical values
/// - predicted_values: Array of values predicted by a model
/// 
/// # Output
/// - Mean absolute error
///
/// # LaTeX Formula
/// - MAE = \\frac{\\sum^{n}_{i=1}\\lvert y_{i}-x_{i}\rvert}{n}
///
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Mean_absolute_error
/// - Original Source: N/A
pub fn mae(observed_values: &Array1<f64>, predicted_values: &Array1<f64>) -> Result<f64, Error> {
    compare_array_len(observed_values, predicted_values, "observed_values", "predicted_values")?;
    Ok((observed_values - predicted_values).map(|v| { v.abs() } ).sum() / (observed_values.len() as f64))
}


/// # Description
/// Measures an error between paired observations (Usually, empirical observations vs simulated observations).
/// 
/// # Input
/// - observed_values: Array of observed/empirical values
/// - predicted_values: Array of values predicted by a model
/// 
/// # Output
/// - Mean squared error
///
/// # LaTeX Formula
/// - MSE = \\frac{1}{n}\\sum^{n}_{i=1}(y_{i}-x_{i})^{2}
///
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Mean_squared_error
/// - Original Source: N/A
pub fn mse(observed_values: &Array1<f64>, predicted_values: &Array1<f64>) -> Result<f64, Error> {
    compare_array_len(observed_values, predicted_values, "observed_values", "predicted_values")?;
    Ok((observed_values - predicted_values).map(|v| { v.powi(2) } ).sum() / (observed_values.len() as f64))
}