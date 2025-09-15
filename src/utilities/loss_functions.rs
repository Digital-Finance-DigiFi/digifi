use std::{borrow::Borrow, iter::zip};
use crate::error::DigiFiError;
use crate::utilities::compare_len;


/// Trait for defining a loss function.
pub trait LossFunction {

    /// Measures an error between observed and predicted values.
    ///
    /// # Input
    /// - `observation`: Observed/empirical value
    /// - `prediction`: Value predicted by the model
    ///
    /// # Output
    /// - An error/loss
    fn loss(&self, observation: f64, prediction: f64) -> f64;

    /// Measures an error between observed and predicted values.
    ///
    /// # Input
    /// - `observations`: An iterator of observed/empirical values
    /// - `predictions`: An iterator of values predicted by the model
    ///
    /// # Output
    /// - Error/losse
    fn loss_iter<T, I>(&self, observations: T, predictions: T) -> Result<f64, DigiFiError>
    where
        T: Iterator<Item = I> + ExactSizeIterator,
        I: Borrow<f64>;
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

    fn loss(&self, observation: f64, prediction: f64) -> f64 {
        (observation - prediction).abs()
    }

    /// Measures an error between observed and predicted values.
    ///
    /// # Input
    /// - `observations`: An iterator of observed/empirical values
    /// - `predictions`: An iterator of values predicted by the model
    ///
    /// # Output
    /// - Error/losse
    ///
    /// # Errors
    /// - Returns an error if the lengths of `observations` and `predictions` do not coincide.
    fn loss_iter<T, I>(&self, observations: T, predictions: T) -> Result<f64, DigiFiError>
    where
        T: Iterator<Item = I> + ExactSizeIterator,
        I: Borrow<f64>,
    {
        compare_len(&observations, &predictions, "observations", "predictions")?;
        let len: f64 = observations.len() as f64;
        Ok(zip(observations, predictions).fold(0.0, |sum, (o, p)| { sum + (o.borrow() - p.borrow()).abs() } ) / len)
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

    fn loss(&self, observation: f64, prediction: f64) -> f64 {
        (observation - prediction).powi(2)
    }

    /// Measures an error between observed and predicted values.
    ///
    /// # Input
    /// - `observations`: An iterator of observed/empirical values
    /// - `predictions`: An iterator of values predicted by the model
    ///
    /// # Output
    /// - Error/losse
    ///
    /// # Errors
    /// - Returns an error if the lengths of `observations` and `predictions` do not coincide.
    fn loss_iter<T, I>(&self, observations: T, predictions: T) -> Result<f64, DigiFiError>
    where
        T: Iterator<Item = I> + ExactSizeIterator,
        I: Borrow<f64>,
    {
        compare_len(&observations, &predictions, "observations", "predictions")?;
        let len: f64 = observations.len() as f64;
        Ok(zip(observations, predictions).fold(0.0, |sum, (o, p)| { sum + (o.borrow() - p.borrow()).powi(2) } ) / len)
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

    fn loss(&self, observation: f64, prediction: f64) -> f64 {
        (observation - prediction).powi(2)
    }

    /// Measures an error between observed and predicted values.
    ///
    /// # Input
    /// - `observations`: An iterator of observed/empirical values
    /// - `predictions`: An iterator of values predicted by the model
    ///
    /// # Output
    /// - Error/losse
    ///
    /// # Errors
    /// - Returns an error if the lengths of `observations` and `predictions` do not coincide.
    fn loss_iter<T, I>(&self, observations: T, predictions: T) -> Result<f64, DigiFiError>
    where
        T: Iterator<Item = I> + ExactSizeIterator,
        I: Borrow<f64>,
    {
        compare_len(&observations, &predictions, "observations", "predictions")?;
        Ok(zip(observations, predictions).fold(0.0, |sum, (o, p)| { sum + (o.borrow() - p.borrow()).powi(2) } ))
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

    fn loss(&self, observation: f64, prediction: f64) -> f64 {
        2.0 * (prediction / observation - 1.0).abs()
    }

    /// Measures an error between observed and predicted values.
    ///
    /// # Input
    /// - `observations`: An iterator of observed/empirical values
    /// - `predictions`: An iterator of values predicted by the model
    ///
    /// # Output
    /// - Error/losse
    ///
    /// # Errors
    /// - Returns an error if the lengths of `observations` and `predictions` do not coincide.
    fn loss_iter<T, I>(&self, observations: T, predictions: T) -> Result<f64, DigiFiError>
    where
        T: Iterator<Item = I> + ExactSizeIterator,
        I: Borrow<f64>,
    {
        compare_len(&observations, &predictions, "observations", "predictions")?;
        let len: f64 = observations.len() as f64;
        Ok(zip(observations, predictions).fold(0.0, |sum, (o, p)| { sum + 2.0 * (p.borrow() / o.borrow() - 1.0).abs() } ) / len)
    }
}