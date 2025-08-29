use ndarray::Array1;
use crate::error::DigiFiError;
use crate::utilities::compare_array_len;


/// Trait for defining a loss function.
pub trait LossFunction {

    /// Measures an error between observed and predicted values.
    ///
    /// # Input
    /// - `observed_value`: Observed/empirical value
    /// - `predicted_value`: Value predicted by the model
    ///
    /// # Output
    /// - An error/loss
    fn loss(&self, observed_value: f64, predicted_value: f64) -> f64;

    /// Measures an error between observed and predicted values.
    ///
    /// # Input
    /// - `observed_value`: An array of observed/empirical values
    /// - `predicted_value`: An array of values predicted by the model
    ///
    /// # Output
    /// - An array of errors/losses
    fn loss_array(&self, observed_values: &Array1<f64>, predicted_values: &Array1<f64>) -> Result<f64, DigiFiError>;
}


/// Measures an error between paired observations (Usually, empirical observations vs simulated observations).
///
/// # LaTeX Formula
/// - MAE = \\frac{\\sum^{n}\_{i=1}\\lvert y_{i}-x_{i}\\rvert}{n}
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Mean_absolute_error>
/// - Original Source: N/A
pub struct MAE;

impl LossFunction for MAE {

    fn loss(&self, observed_value: f64, predicted_value: f64) -> f64 {
        (observed_value - predicted_value).abs()
    }

    /// Measures an error between observed and predicted values.
    ///
    /// # Input
    /// - `observed_value`: An array of observed/empirical values
    /// - `predicted_value`: An array of values predicted by the model
    ///
    /// # Output
    /// - An array of errors/losses
    ///
    /// # Errors
    /// - Returns an error if the lengths of `observed_values` and `predicted_values` do not coincide.
    fn loss_array(&self, observed_values: &Array1<f64>, predicted_values: &Array1<f64>) -> Result<f64, DigiFiError> {
        compare_array_len(observed_values, predicted_values, "observed_values", "predicted_values")?;
        Ok((observed_values - predicted_values).map(|v| { v.abs() } ).sum() / (observed_values.len() as f64))
    }
}


/// Measures an error between paired observations (Usually, empirical observations vs simulated observations).
///
/// # LaTeX Formula
/// - MSE = \\frac{1}{n}\\sum^{n}\_{i=1}(y_{i}-x_{i})^{2}
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Mean_squared_error>
/// - Original Source: N/A
pub struct MSE;

impl LossFunction for MSE {

    fn loss(&self, observed_value: f64, predicted_value: f64) -> f64 {
        (observed_value - predicted_value).powi(2)
    }

    /// Measures an error between observed and predicted values.
    ///
    /// # Input
    /// - `observed_value`: An array of observed/empirical values
    /// - `predicted_value`: An array of values predicted by the model
    ///
    /// # Output
    /// - An array of errors/losses
    ///
    /// # Errors
    /// - Returns an error if the lengths of `observed_values` and `predicted_values` do not coincide.
    fn loss_array(&self, observed_values: &Array1<f64>, predicted_values: &Array1<f64>) -> Result<f64, DigiFiError> {
        compare_array_len(observed_values, predicted_values, "observed_values", "predicted_values")?;
        Ok((observed_values - predicted_values).map(|v| { v.powi(2) } ).sum() / (observed_values.len() as f64))
    }
}


/// Measures an error between paired observations (Usually, empirical observations vs simulated observations).
///
/// # LaTeX Formula
/// - MSE = \\sum^{n}\_{i=1}(y_{i}-x_{i})^{2}
///
/// # Links
/// - Wikipedia: N/A
/// - Original Source: N/A
pub struct SSE;

impl LossFunction for SSE {

    fn loss(&self, observed_value: f64, predicted_value: f64) -> f64 {
        (observed_value - predicted_value).powi(2)
    }

    /// Measures an error between observed and predicted values.
    ///
    /// # Input
    /// - `observed_value`: An array of observed/empirical values
    /// - `predicted_value`: An array of values predicted by the model
    ///
    /// # Output
    /// - An array of errors/losses
    ///
    /// # Errors
    /// - Returns an error if the lengths of `observed_values` and `predicted_values` do not coincide.
    fn loss_array(&self, observed_values: &Array1<f64>, predicted_values: &Array1<f64>) -> Result<f64, DigiFiError> {
        compare_array_len(observed_values, predicted_values, "observed_values", "predicted_values")?;
        Ok((observed_values - predicted_values).map(|v| { v.powi(2) } ).sum())
    }
}


/// Measures an error between paired observations (Usually, empirical observations vs simulated observations).
///
/// # LaTeX Formula
/// - Loss = \\frac{1}{n}\\sum^{n}\_{i=1}(\\lvert\\frac{x_{i}}{y_{i}}-1\\rvert + \\lvert 1-\\frac{x_{i}}{y_{i}}\\rvert)
///
/// # Links
/// - Wikipedia: N/A
/// - Original Source: N/A
pub struct StraddleLoss;

impl LossFunction for StraddleLoss {

    fn loss(&self, observed_value: f64, predicted_value: f64) -> f64 {
        let delta: f64 = predicted_value / observed_value;
        (delta - 1.0).abs() + (1.0  - delta).abs()
    }

    /// Measures an error between observed and predicted values.
    ///
    /// # Input
    /// - `observed_value`: An array of observed/empirical values
    /// - `predicted_value`: An array of values predicted by the model
    ///
    /// # Output
    /// - An array of errors/losses
    ///
    /// # Errors
    /// - Returns an error if the lengths of `observed_values` and `predicted_values` do not coincide.
    fn loss_array(&self, observed_values: &Array1<f64>, predicted_values: &Array1<f64>) -> Result<f64, DigiFiError> {
        compare_array_len(observed_values, predicted_values, "observed_values", "predicted_values")?;
        Ok((predicted_values / observed_values).map(|v| { (v - 1.0).abs() + (1.0 - v).abs() } ).sum() / (observed_values.len() as f64))
    }
}