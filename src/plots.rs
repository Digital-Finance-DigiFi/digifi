//! # Plots
//! 
//! Contains functionality for plotting different results produced by the library.


// Re-Exports
pub use crate::financial_instruments::{plot_payoff, plot_profit};
pub use crate::financial_instruments::derivatives::plot_present_value_surface;
pub use crate::portfolio_applications::portfolio_composition::plot_efficient_frontier;
pub use crate::random_generators::{plot_pdf, plot_2d_scatter_points, plot_3d_scatter_points};
pub use crate::stochastic_processes::plot_stochastic_paths;
pub use crate::technical_indicators::{plot_moving_average, plot_macd, plot_bollinger_bands, plot_rsi, plot_adx, plot_obv};


use ndarray::Array1;
use plotly::{Plot, Bar, Candlestick, Layout, layout::{Axis, RangeSlider}};
use crate::error::DigiFiError;
use crate::utilities::compare_array_len;


/// Plots the candlestick chart.
///
/// # Input
/// - `open`: An array of open prices
/// - `high`: An array of high prices
/// - `low`: An array of low prices
/// - `close`: An array of close prices
/// - `times`: Vector of times
///
/// # Output
/// - Candlestick chart
///
/// # Errors
/// - Returns an error if the length of time vector does not match the lengths of price arrays.
///
/// # Examples
///
/// ```rust,ignore
/// 
/// #[cfg(all(feature = "plotly", feature = "sample_data"))]
/// fn test_candlestick_chart() -> () {
///     use plotly::Plot;
///     use digifi::utilities::sample_data::SampleData;
///     use digifi::plots::plot_candlestick_chart;
///
///     // Sample stock data
///     let sample: SampleData = SampleData::Stock;
///     let (dates, stock_data) = sample.load_sample_data();
///     let dates: Vec<String> = dates.into_iter().map(|v| String::from(v) ).collect();
///
///     // Candlestick chart
///     let plot: Plot = plot_candlestick_chart(stock_data.get("Open").unwrap(), stock_data.get("High").unwrap(), stock_data.get("Low").unwrap(),
///                                             stock_data.get("Close").unwrap(), &dates).unwrap();
///     plot.show();
/// }
/// ```
pub fn plot_candlestick_chart(open: &Array1<f64>, high: &Array1<f64>, low: &Array1<f64>, close: &Array1<f64>, times: &Vec<String>) -> Result<Plot, DigiFiError> {
    compare_array_len(&open, &high, "open", "high")?;
    compare_array_len(&open, &low, "open", "low")?;
    compare_array_len(&open, &close, "open", "close")?;
    if open.len() != times.len() {
        return Err(DigiFiError::UnmatchingLength { array_1: "times".to_owned(), array_2: "open price".to_owned(), });
    }
    // Candlestick chart
    let mut plot: Plot = Plot::new();
    plot.add_trace(Candlestick::new(times.clone(), open.to_vec(), high.to_vec(), low.to_vec(), close.to_vec()));
    let x_axis_range_slider: RangeSlider = RangeSlider::new().visible(false);
    let x_axis: Axis = Axis::new().title("Time").range_slider(x_axis_range_slider);
    let y_axis: Axis = Axis::new().title("Price");
    let layout: Layout = Layout::new().title("<b>Chandlestick Chart</b>").x_axis(x_axis).y_axis(y_axis);
    plot.set_layout(layout);
    Ok(plot)
}


/// Plots the volume.
///
/// # Input
/// - `volume`: An array of volumes
/// - `times`: Vector of times
///
/// # Output
/// - Volume plot
///
/// # Errors
/// - Returns an error if the length of time vector does not match the length of the volume array.
///
/// # Examples
///
/// ```rust,ignore
/// 
/// #[cfg(all(feature = "plotly", feature = "sample_data"))]
/// fn test_candlestick_chart() -> () {
///     use plotly::Plot;
///     use digifi::utilities::sample_data::SampleData;
///     use digifi::plots::plot_volume;
///
///     // Sample stock data
///     let sample: SampleData = SampleData::Stock;
///     let (dates, stock_data) = sample.load_sample_data();
///     let dates: Vec<String> = dates.into_iter().map(|v| String::from(v) ).collect();
///
///     // Volume plot
///     let plot: Plot = plot_volume(stock_data.get("Volume").unwrap(), &dates).unwrap();
///     plot.show();
/// }
/// ```
pub fn plot_volume(volume: &Array1<f64>, times: &Vec<String>) -> Result<Plot, DigiFiError> {
    if volume.len() != times.len() {
        return Err(DigiFiError::UnmatchingLength { array_1: "times".to_owned(), array_2: "volume".to_owned(), });
    }
    let mut plot: Plot = Plot::new();
    plot.add_trace(Bar::new(times.clone(), volume.to_vec()));
    let x_axis: Axis = Axis::new().title("Time");
    let y_axis: Axis = Axis::new().title("Volume");
    let layout: Layout = Layout::new().title("<b>Volume</b>").x_axis(x_axis).y_axis(y_axis);
    plot.set_layout(layout);
    Ok(plot)

}


#[cfg(all(test, feature = "plotly", feature = "sample_data"))]
mod tests {
    use plotly::Plot;
    use crate::utilities::sample_data::SampleData;

    #[test]
    #[ignore]
    fn unit_test_plot_candlestick_chart() -> () {
        use crate::plots::plot_candlestick_chart;
        // Sample stock data
        let sample: SampleData = SampleData::Stock;
        let (dates, stock_data) = sample.load_sample_data();
        let dates: Vec<String> = dates.into_iter().map(|v| String::from(v) ).collect();
        // Candlestick chart
        let plot: Plot = plot_candlestick_chart(stock_data.get("Open").unwrap(), stock_data.get("High").unwrap(), stock_data.get("Low").unwrap(),
        stock_data.get("Close").unwrap(), &dates).unwrap();
        plot.show();
    }

    #[test]
    #[ignore]
    fn unit_test_plot_volume() -> () {
        use crate::plots::plot_volume;
        // Sample stock data
        let sample: SampleData = SampleData::Stock;
        let (dates, stock_data) = sample.load_sample_data();
        let dates: Vec<String> = dates.into_iter().map(|v| String::from(v) ).collect();
        // Volume plot
        let plot: Plot = plot_volume(stock_data.get("Volume").unwrap(), &dates).unwrap();
        plot.show();
    }
}