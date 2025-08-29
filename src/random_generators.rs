//! # random Generators
//! 
//! Contains different pseudo-random number generator algorithms for generating numbers from continuous uniform and standard normal distributions.
//! This module also provides algorithms to generate pseudo-random numbers from different probability distributions provided by this library.


// Re-Exports
pub use self::generator_algorithms::{accept_reject, inverse_transform, box_muller, marsaglia, ziggurat};
pub use self::uniform_generators::{LinearCongruentialGenerator, FibonacciGenerator};
pub use self::standard_normal_generators::{
    StandardNormalAcceptReject, StandardNormalInverseTransform, StandardNormalBoxMuller, StandardNormalMarsaglia, StandardNormalZiggurat,
};


pub mod generator_algorithms;
pub mod standard_normal_generators;
pub mod uniform_generators;


use std::time::{SystemTime, UNIX_EPOCH};
use ndarray::Array1;
#[cfg(feature = "plotly")]
use plotly::{Plot, Histogram, Scatter, Scatter3D, common::Mode, traces::histogram::HistNorm};
use crate::error::DigiFiError;


pub trait RandomGenerator<T> {
    fn new_shuffle(sample_size: usize) -> Result<T, DigiFiError>;

    fn generate(&self) -> Result<Array1<f64>, DigiFiError>;
}


/// Generates a seed from nanosecond timestamp of the system.
///
/// # Examples
///
/// ```rust
/// use digifi::random_generators::generate_seed;
///
/// let seed_1: u32 = generate_seed().unwrap();
/// let seed_2: u32 = generate_seed().unwrap();
///
/// assert!(seed_1 != seed_2);
/// ```
pub fn generate_seed () -> Result<u32, DigiFiError> {
    let start: SystemTime = SystemTime::now();
    let delta: f64 = start.duration_since(UNIX_EPOCH)?.subsec_nanos() as f64;
    // Drop the first two digits and last two digits from delta
    let delta: u32 = (delta / 100.0) as u32;
    let remainder: u32 = delta.rem_euclid(100_000);
    let big_digit_number: u32 = delta - remainder;
    Ok(delta - big_digit_number)
}


#[cfg(feature = "plotly")]
/// Plots the probability density of the generated points.
///
/// # Input
/// - `points`: An array of generated points
/// - `n_bins`: Number of bins to use in the historgram plot
///
/// # Output
/// - Probability density historgram plot
///
/// # Examples
///
/// ```rust,ignore
/// use ndarray::Array1;
/// use digifi::random_generators::{RandomGenerator, StandardNormalInverseTransform};
///
/// #[cfg(feature = "plotly")]
/// fn test_plot_pdf() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_pdf;
///
///     // Array of pseudo-random numbers
///     let points: Array1<f64> = StandardNormalInverseTransform::new_shuffle(10_000).unwrap().generate().unwrap();
///
///     // PDF plot
///     let plot: Plot = plot_pdf(&points, 150);
///     plot.show();
/// }
/// ```
pub fn plot_pdf(points: &Array1<f64>, n_bins: usize) -> Plot {
    let mut plot: Plot = Plot::new();
    plot.add_trace(Histogram::new(points.to_vec()).hist_norm(HistNorm::ProbabilityDensity).n_bins_x(n_bins).name("Probability Density"));
    plot
}


#[cfg(feature = "plotly")]
/// Plots the 2D scatter plot of an array of points against itself. Assuming the points are randomly distributed, the plot can be used to
/// validate the distribution of points (e.g., normally distributed numbers are distributed in an ellipsical shape).
///
/// # Input
/// - `points`: An array of generated points
///
/// # Output
/// - 2D scatter plot of the generated points
///
/// # Example
///
/// ```rust,ignore
/// use ndarray::Array1;
/// use digifi::random_generators::{RandomGenerator, StandardNormalInverseTransform};
///
/// #[cfg(feature = "plotly")]
/// fn test_plot_2d_scatter_points() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_2d_scatter_points;
///
///     // Array of pseudo-random numbers
///     let points: Array1<f64> = StandardNormalInverseTransform::new_shuffle(10_000).unwrap().generate().unwrap();
///
///     // 2D scatter plot
///     let plot: Plot = plot_2d_scatter_points(&points);
///     plot.show();
/// }
/// ```
pub fn plot_2d_scatter_points(points: &Array1<f64>) -> Plot {
    let mut y: Vec<f64> = points.to_vec();
    y.rotate_left(1);
    let mut plot: Plot = Plot::new();
    plot.add_trace(Scatter::new(points.to_vec(), y).name("2D Scatter Plot").mode(Mode::Markers));
    plot
}


#[cfg(feature = "plotly")]
/// Plots the 3D scatter plot of an array of points against itself. Assuming the points are randomly distributed, the plot can be used to
/// validate the distribution of points (e.g., normally distributed numbers are distributed in an ellipsoidal shape).
///
/// # Input
/// - `points`: An array of generated points
///
/// # Output
/// - 3D scatter plot of the generated points
///
/// # Example
///
/// ```rust,ignore
/// use ndarray::Array1;
/// use digifi::random_generators::{RandomGenerator, StandardNormalInverseTransform};
///
/// #[cfg(feature = "plotly")]
/// fn test_plot_3d_scatter_points() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_3d_scatter_points;
///
///     // Array of pseudo-random numbers
///     let points: Array1<f64> = StandardNormalInverseTransform::new_shuffle(10_000).unwrap().generate().unwrap();
///
///     // 3D scatter plot
///     let plot: Plot = plot_3d_scatter_points(&points);
///     plot.show();
/// }
/// ```
pub fn plot_3d_scatter_points(points: &Array1<f64>) -> Plot {
    let mut y: Vec<f64> = points.to_vec();
    y.rotate_left(1);
    let mut z: Vec<f64> = points.to_vec();
    z.rotate_left(2);
    let mut plot: Plot = Plot::new();
    plot.add_trace(Scatter3D::new(points.to_vec(), y, z).name("3D Scatter Plot").mode(Mode::Markers));
    plot
}


#[cfg(test)]
mod tests {
    
    #[test]
    fn unit_test_generate_seed() -> () {
        use crate::random_generators::generate_seed;
        let seed_1: u32 = generate_seed().unwrap();
        let seed_2: u32 = generate_seed().unwrap();
        assert!(seed_1 != seed_2);
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_plot_pdf() -> () {
        use ndarray::Array1;
        use plotly::Plot;
        use crate::random_generators::{RandomGenerator, standard_normal_generators::StandardNormalInverseTransform, plot_pdf};
        // Array of pseudo-random numbers
        let points: Array1<f64> = StandardNormalInverseTransform::new_shuffle(10_000).unwrap().generate().unwrap();
        // PDF plot
        let plot: Plot = plot_pdf(&points, 150);
        plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_plot_2d_scatter_points() -> () {
        use ndarray::Array1;
        use plotly::Plot;
        use crate::random_generators::{RandomGenerator, standard_normal_generators::StandardNormalInverseTransform, plot_2d_scatter_points};
        // Array of pseudo-random numbers
        let points: Array1<f64> = StandardNormalInverseTransform::new_shuffle(10_000).unwrap().generate().unwrap();
        // 2D scatter plot
        let plot: Plot = plot_2d_scatter_points(&points);
        plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_plot_3d_scatter_points() -> () {
        use ndarray::Array1;
        use plotly::Plot;
        use crate::random_generators::{RandomGenerator, standard_normal_generators::StandardNormalInverseTransform, plot_3d_scatter_points};
        // Array of pseudo-random numbers
        let points: Array1<f64> = StandardNormalInverseTransform::new_shuffle(10_000).unwrap().generate().unwrap();
        // 3D scatter plot
        let plot: Plot = plot_3d_scatter_points(&points);
        plot.show();
    }
}