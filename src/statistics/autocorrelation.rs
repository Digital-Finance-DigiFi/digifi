use ndarray::{Array1, Array2, s};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
#[cfg(feature = "plotly")]
use plotly::{Plot, Layout, Scatter, common::{Line, color::NamedColor}, layout::{Axis, Shape, ShapeType, ShapeLine}};
use crate::error::DigiFiError;
use crate::statistics::pearson_correlation;


/// Autocorrelation, which measures the Pearson correlation of a signal with a delayed copy of itself.
/// 
/// # Input
/// - `array`: Array of data
/// - `n_lag`: Number of lags to be used as a delay
/// 
/// # Ouput
/// - Autocorrelation
/// 
/// # Errors
/// - Returns an error if the number of lags used for autocorrelation exceeds the number of data points in the `array`.
/// 
/// # LaTeX Formula
/// - \\rho_{X}(k) = \\frac{E\[(X_{t+k} - \\mu)\\bar{(X_{t} - \\mu)}\]}{\\sigma^{2}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Autocorrelation>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::autocorrelation;
/// 
/// let array: Array1<f64> = Array1::from_vec(vec![0.0, 0.25, 0.15, -0.45, 0.3, -0.05, 0.12, 0.07]);
/// 
/// // Answer obtained using `corr()` method for pandas DataFrame
/// assert!((autocorrelation(&array, 1).unwrap() - -0.5160922669123634).abs() < TEST_ACCURACY);
/// assert!((autocorrelation(&array, 2).unwrap() - -0.04864833358838002).abs() < TEST_ACCURACY);
/// assert!((autocorrelation(&array, 3).unwrap() - 0.1033387585496719).abs() < TEST_ACCURACY);
/// assert!((autocorrelation(&array, 4).unwrap() - -0.15200675246896855).abs() < TEST_ACCURACY);
/// ```
pub fn autocorrelation(array: &Array1<f64>, n_lag: usize) -> Result<f64, DigiFiError> {
    let array_len: usize = array.len();
    if array_len <= n_lag {
        return Err(DigiFiError::ValidationError {
            title: "Autocorrelation".to_owned(),
            details: "Number of lags used for autocorrelation exceeds the number of data points in the array.".to_owned(),
        });
    }
    let array_t: Array1<f64> = array.slice(s![0..(array_len - n_lag)]).to_owned();
    let array_t_plus_n: Array1<f64> = array.slice(s![n_lag..array_len]).to_owned();
    pearson_correlation(&array_t, &array_t_plus_n, 0)
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Autocorrelation data.
pub struct Autocorrelation {
    /// Autocorrelation for different number of lags starting from lag 0 up to and including the autocorrelation of `max_lag`
    pub autocorrelations: Array1<f64>,
    /// Lag sizes that correspond to the autocorrelations
    pub lags: Vec<usize>,
}


/// Autocorrelation, which measures the Pearson correlation of a signal with a delayed copy of itself. This function computes
/// all autocorrelations up until the `max_lag`.
/// 
/// # Input
/// - `array`: Array of data
/// - `max_lag`: Maximum lag to use to compute partial autocorrelations up to (inclusive)
/// 
/// # Ouput
/// - Autocorrelation data, which includes an array of autocorrelations up to and including the autocorrelation for the specified lag,
/// and other data used in computing the autocorrelations.
/// 
/// # Errors
/// - Returns an error if the number of lags used for autocorrelation exceeds the number of data points in the `array`.
/// 
/// # LaTeX Formula
/// - \\rho_{X}(k) = \\frac{E\[(X_{t+k} - \\mu)\\bar{(X_{t} - \\mu)}\]}{\\sigma^{2}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Autocorrelation>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{Autocorrelation, autocorrelation_array};
/// 
/// let array: Array1<f64> = Array1::from_vec(vec![0.0, 0.25, 0.15, -0.45, 0.3, -0.05, 0.12, 0.07]);
/// let acs: Autocorrelation = autocorrelation_array(&array, 4).unwrap();
/// 
/// // Answer obtained using `corr()` method for pandas DataFrame
/// let tested_result: Array1<f64> = Array1::from_vec(vec![1.0, -0.5160922669123634, -0.04864833358838002, 0.1033387585496719, -0.15200675246896855]);
/// assert!((acs.autocorrelations - tested_result).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub fn autocorrelation_array(array: &Array1<f64>, max_lag: usize) -> Result<Autocorrelation, DigiFiError> {
    let n_acs: usize = max_lag + 1;
    let mut acs: Vec<f64> = Vec::with_capacity(n_acs);
    let mut lags: Vec<usize> = Vec::with_capacity(n_acs);
    for n in (0..=max_lag).into_iter() {
        match n {
            0 => {
                acs.push(1.0);
                lags.push(0);
            },
            _ => {
                acs.push(autocorrelation(array, n)?);
                lags.push(n);
            }
        }
    }
    Ok(Autocorrelation {
        autocorrelations: Array1::from_vec(acs),
        lags,
    })
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Partial autocorrelation data.
pub struct PartialAutocorrelation {
    /// Partial autocorrelation for different number of lags starting from lag 0 up to and including the partial autocorrelation of `max_lag`
    pub partial_autocorrelations: Array1<f64>,
    /// Lag sizes that correspond to the autocorrelations
    pub lags: Vec<usize>,
    /// Matrix of Levinson-Durbin phi coefficients that are iteratively used to compute partial autocorrelations for the consecutively increasing lags
    pub levinson_durbin_phis: Array2<f64>,
}


/// Partial autocorrelation, which is the autocorrelation between z_{t} and z_{t+k} that is not accounted for by lags `1` through `k-1` inclusive.
/// 
/// Note: This implementation uses Levinson-Durbin algorithm to compute the theoretical partial autocorrelation values iteratively.
/// It requires that the time series being analysed is stationary.
/// 
/// # Input
/// - `array`: Array of data
/// - `max_lag`: Maximum lag to use to compute partial autocorrelations up to (inclusive)
/// 
/// # Ouput
/// - Partial autocorrelation data, which includes an array of partial autocorrelations up to and including the partial autocorrelation for the specified lag,
/// and other data used in computing the partial autocorrelations.
/// 
/// # Errors
/// - Returns an error if the number of lags used for autocorrelation exceeds the number of data points in the `array`.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Partial_autocorrelation_function>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{PartialAutocorrelation, partial_autocorrelation};
/// 
/// let array: Array1<f64> = Array1::from_vec(vec![0.0, 0.25, 0.15, -0.45, 0.3, -0.05, 0.12, 0.07]);
/// let pacs: PartialAutocorrelation = partial_autocorrelation(&array, 4).unwrap();
/// 
/// // Answer obtained using `statsmodels.tsa.stattools.pacf` function from statsmodels with method set to `ldb`
/// let tested_result: Array1<f64> = Array1::from_vec(vec![1.0, -0.51428607, -0.41256983, -0.24459419, -0.290511]);
/// assert!((pacs.partial_autocorrelations - tested_result).map(|v| v.abs() ).sum() < 20_000_000.0 * TEST_ACCURACY);
/// ```
pub fn partial_autocorrelation(array: &Array1<f64>, max_lag: usize) -> Result<PartialAutocorrelation, DigiFiError> {
    let n_pacs: usize = max_lag + 1;
    let mut pacs: Vec<f64> = Vec::with_capacity(n_pacs);
    let mut lags: Vec<usize> = Vec::with_capacity(n_pacs);
    let mut phis: Array2<f64> = Array2::from_shape_fn((n_pacs, n_pacs), |_| { f64::NAN } );
    for n in (0..=max_lag).into_iter() {
        match n {
            0 => {
                phis[[0,0]] = 1.0;
                pacs.push(1.0);
            },
            1 => {
                let phi: f64 = autocorrelation(array, 1)?;
                phis[[1,1]] = phi;
                pacs.push(phi);
            },
            _ => {
                let n_minus_one: usize = n - 1;
                let n_minus_two: usize = n - 2;
                let mut numerator_sum: f64 = 0.0;
                let mut denominator_sum: f64 = 0.0;
                for k in (1..=n_minus_one).into_iter() {
                    if n_minus_one!=k {
                        phis[[n_minus_one, k]] = phis[[n_minus_two, k]] - phis[[n_minus_one, n_minus_one]] * phis[[n_minus_two, n_minus_one - k]];
                    }
                    numerator_sum += phis[[n_minus_one, k]] * autocorrelation(array, n - k)?;
                    denominator_sum += phis[[n_minus_one, k]] * autocorrelation(array, k)?;
                }
                let phi: f64 = (autocorrelation(array, n)? - numerator_sum) / (1.0 - denominator_sum);
                phis[[n, n]] = phi;
                pacs.push(phi);
            },
        }
        lags.push(n);
    }
    Ok(PartialAutocorrelation {
        partial_autocorrelations: Array1::from_vec(pacs),
        lags,
        levinson_durbin_phis: phis,
    })
}


#[cfg(feature = "plotly")]
/// Plots autocorrelation for each associated lag
/// 
/// # Input
/// - `autocorrelations`: Autocorrelations
/// - `lags`: Lags that correspond to the autocorrelations
/// - `partial`: Whether the autocorrelations are partial autocorrelations ot not (Sets the appropriate title for the y-axis and plot)
/// - `upper_cl`: Upper critical value bound
/// - `lower_cl`: Lower critical value bound
///
/// # Output
/// - Autocorrelations plot
///
/// # Examples
/// 
/// 1. Autocorrelation plot
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::statistics::{Autocorrelation, autocorrelation_array};
/// 
/// #[cfg(feature = "plotly")]
/// fn test_plot_autocorrelation() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_autocorrelation;
///
///     let array: Array1<f64> = Array1::from_vec(vec![0.0, 0.25, 0.15, -0.45, 0.3, -0.05, 0.12, 0.07]);
///     let acs: Autocorrelation = autocorrelation_array(&array, 4).unwrap();
///
///     // Autocorrelations plot
///     let plot: Plot = plot_autocorrelation(acs.autocorrelations, acs.lags, false, Some(0.15), Some(-0.15));
///     plot.show();
/// }
/// ```
/// 
/// 2. Partial autocorrelation plot
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::statistics::{PartialAutocorrelation, partial_autocorrelation};
/// 
/// #[cfg(feature = "plotly")]
/// fn test_plot_partial_autocorrelation() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_autocorrelation;
///
///     let array: Array1<f64> = Array1::from_vec(vec![0.0, 0.25, 0.15, -0.45, 0.3, -0.05, 0.12, 0.07]);
///     let pacs: PartialAutocorrelation = partial_autocorrelation(&array, 4).unwrap();
/// 
///     // Partial autocorrelation plot
///     let plot: Plot = plot_autocorrelation(pacs.partial_autocorrelations, pacs.lags, true, Some(0.15), Some(-0.15));
///     plot.show()
/// }
/// ```
pub fn plot_autocorrelation(autocorrelations: Array1<f64>, lags: Vec<usize>, partial: bool, upper_crit: Option<f64>, lower_crit: Option<f64>) -> Plot {
    let title: String = if partial { String::from("Partial Autocorrelation") } else { String::from("Autocorrelation") };
    let mut plot: Plot = Plot::new();
    // Upper confidence level
    if let Some(upper_crit) = upper_crit {
        let upper_crit_line: Line = Line::new().color(NamedColor::Blue);
        plot.add_trace(Scatter::new(lags.clone(), vec![upper_crit; lags.len()]).name("Upper Critical Value").line(upper_crit_line));
    }
    // Lower confidence level
    if let Some(lower_crit) = lower_crit {
        let lower_crit_line: Line = Line::new().color(NamedColor::Blue);
        plot.add_trace(Scatter::new(lags.clone(), vec![lower_crit; lags.len()]).name("Lower Critical Value").line(lower_crit_line));
    }
    // Layout and autocorrelation values
    let x_axis: Axis = Axis::new().title("Lag").range(vec![-1, lags[lags.len() - 1] as i64 + 1]).zero_line(false);
    let y_axis: Axis = Axis::new().title(&title).range(vec![-1.0, 1.0]);
    let mut layout: Layout = Layout::new().title(format!("<b>{}</b>", title)).x_axis(x_axis).y_axis(y_axis);
    for (ac, lag) in autocorrelations.iter().zip(lags.iter()) {
        let shape: Shape = Shape::new()
            .shape_type(ShapeType::Line)
            .x0(*lag).x1(*lag)
            .y0(0.0).y1(*ac)
            .line(ShapeLine::new().color(NamedColor::Black).width(2.0));
        layout.add_shape(shape);
    }
    plot.set_layout(layout);
    plot
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_autocorrelation() -> () {
        use crate::statistics::autocorrelation::autocorrelation;
        let array: Array1<f64> = Array1::from_vec(vec![0.0, 0.25, 0.15, -0.45, 0.3, -0.05, 0.12, 0.07]);
        // Answer obtained using `corr()` method for pandas DataFrame
        assert!((autocorrelation(&array, 1).unwrap() - -0.5160922669123634).abs() < TEST_ACCURACY);
        assert!((autocorrelation(&array, 2).unwrap() - -0.04864833358838002).abs() < TEST_ACCURACY);
        assert!((autocorrelation(&array, 3).unwrap() - 0.1033387585496719).abs() < TEST_ACCURACY);
        assert!((autocorrelation(&array, 4).unwrap() - -0.15200675246896855).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_autocorrelation_array() -> () {
        use crate::statistics::autocorrelation::{Autocorrelation, autocorrelation_array};
        let array: Array1<f64> = Array1::from_vec(vec![0.0, 0.25, 0.15, -0.45, 0.3, -0.05, 0.12, 0.07]);
        let acs: Autocorrelation = autocorrelation_array(&array, 4).unwrap();
        // Answer obtained using `corr()` method for pandas DataFrame
        let tested_result: Array1<f64> = Array1::from_vec(vec![1.0, -0.5160922669123634, -0.04864833358838002, 0.1033387585496719, -0.15200675246896855]);
        assert!((acs.autocorrelations - tested_result).map(|v| v.abs() ).sum() < TEST_ACCURACY);

    }

    #[test]
    fn unit_test_partial_autocorrelation() -> () {
        use crate::statistics::autocorrelation::{PartialAutocorrelation, partial_autocorrelation};
        let array: Array1<f64> = Array1::from_vec(vec![0.0, 0.25, 0.15, -0.45, 0.3, -0.05, 0.12, 0.07]);
        let pacs: PartialAutocorrelation = partial_autocorrelation(&array, 4).unwrap();
        // Answer obtained using `statsmodels.tsa.stattools.pacf` function from statsmodels with method set to `ldb`
        let tested_result: Array1<f64> = Array1::from_vec(vec![1.0, -0.51428607, -0.41256983, -0.24459419, -0.290511]);
        assert!((pacs.partial_autocorrelations - tested_result).map(|v| v.abs() ).sum() < 20_000_000.0 * TEST_ACCURACY);
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_plot_autocorrelation() -> () {
        use plotly::Plot;
        use crate::statistics::autocorrelation::{Autocorrelation, autocorrelation_array, plot_autocorrelation};
        let array: Array1<f64> = Array1::from_vec(vec![0.0, 0.25, 0.15, -0.45, 0.3, -0.05, 0.12, 0.07]);
        let acs: Autocorrelation = autocorrelation_array(&array, 4).unwrap();
        // Autocorrelations plot
        let plot: Plot = plot_autocorrelation(acs.autocorrelations, acs.lags, false, Some(0.15), Some(-0.15));
        plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_plot_partial_autocorrelation() -> () {
        use plotly::Plot;
        use crate::statistics::autocorrelation::{PartialAutocorrelation, partial_autocorrelation, plot_autocorrelation};
        let array: Array1<f64> = Array1::from_vec(vec![0.0, 0.25, 0.15, -0.45, 0.3, -0.05, 0.12, 0.07]);
        let pacs: PartialAutocorrelation = partial_autocorrelation(&array, 4).unwrap();
        // Partial autocorrelation plot
        let plot: Plot = plot_autocorrelation(pacs.partial_autocorrelations, pacs.lags, true, Some(0.15), Some(-0.15));
        plot.show();
    }
}