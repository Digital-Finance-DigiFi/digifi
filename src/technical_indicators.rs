//! # Technical Indicators
//! 
//! Contains commmonly used technical indicators such as `SMA`, `EMA`, `MACD`, `RSI`, etc.


use ndarray::{Array1, s};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
#[cfg(feature = "plotly")]
use plotly::{Plot, Trace, Scatter, Bar, Layout, common::{Line, color::NamedColor}, layout::Axis};
use crate::error::DigiFiError;
use crate::utilities::compare_len;


/// Measure of the decline of the asset from its historical peak.
/// 
/// Maximum Drawdown = (Peak Value - Trough Value) / Peak Value
/// 
/// # Input
/// - `asset_value`: Time series of asset value
/// 
/// # Output
/// - Maximum drawdown of the asset value
/// 
/// # LaTeX Formula
/// - \\textit{Maximum Drawdown} = \\frac{\\textit{Peak Value} - \\textit{Trough Value}}{\\textit{Peak Value}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Drawdown_(economics)>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::technical_indicators::maximum_drawdown;
/// 
/// let asset_price: Array1<f64> = Array1::from_vec(vec![100.0, 101.0, 104.0, 102.0, 103.0, 106.0]);
/// let max_dd: f64 = maximum_drawdown(&asset_price);
/// 
/// assert_eq!(max_dd, 100.0*2.0/104.0);
/// ```
pub fn maximum_drawdown(asset_value: &Array1<f64>) -> f64 {
    let mut maximum_drawdown: f64 = 0.0;
    let mut peak: f64 = asset_value[0];
    // Selection of maximum drawdown candidates
    for i in asset_value {
        if peak < *i {
            peak = *i;
        }
        let last_drawdown: f64 = 100.0 * (peak - *i) / peak;
        if maximum_drawdown < last_drawdown {
            maximum_drawdown = last_drawdown;
        }
    }
    maximum_drawdown
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Simple Moving Average (SMA) data.
pub struct SMA {
    /// Size of the rolling window for the SMA
    pub period: usize,
    /// An array of SMA readings
    pub sma: Array1<f64>,
}


/// Simple Moving Average (SMA) describes the direction of the trend, and is computed using the mean over the certain window of readings.
/// 
/// # Input
/// - `price_array`: Array of prices
/// - `period`: Size of the rolling window for the SMA
/// 
/// # Output
/// - An array of SMA readings
///
/// # Errors
/// - Returns an error if the mean of SMA slice cannot be computed.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::{Array1, s};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::technical_indicators::{SMA, sma};
/// 
/// let price_array: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
/// let sma: SMA = sma(&price_array, 3).unwrap();
/// 
/// assert!((sma.sma - Array1::from_vec(vec![f64::NAN, f64::NAN, 11.0, 12.0, 12.0, 13.0])).slice(s![2..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub fn sma(price_array: &Array1<f64>, period: usize) -> Result<SMA, DigiFiError> {
    let mut sma: Vec<f64> = vec![f64::NAN; period - 1];
    let windows = price_array.windows(period).to_owned();
    for window in windows {
        sma.push(window.mean().ok_or(DigiFiError::MeanCalculation { title: "SMA".to_owned(), series: "price array slice".to_owned() })?);
    }
    Ok(SMA { period, sma: Array1::from_vec(sma) })
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Exponential Moving Average (SMA) data.
pub struct EMA {
    /// Size of the rolling window for the EMA
    pub period: usize,
    /// Smoothing of the EMA line
    pub smoothing: i32,
    /// An array of EMA readings
    pub ema: Array1<f64>,
}


/// Exponential Moving Average (EMA) describes the direction of the trend, and requires previous EMA and the latest price to compute;
/// the first EMA reading will be same as SMA.
/// 
/// # Input
/// - `price_array`: Array of prices
/// - `period`: Size of the rolling window for the EMA
/// - `smoothing`: Smoothing of the EMA line
/// 
/// # Output
/// - An array of EMA readings
///
/// # Errors
/// - Returns an error if the mean of price array cannot be computed.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::{Array1, s};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::technical_indicators::{EMA, ema};
/// 
/// let price_array: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
/// let ema: EMA = ema(&price_array, 3, 2).unwrap();
/// 
/// assert!((ema.ema - Array1::from_vec(vec![f64::NAN, f64::NAN, 11.0, 12.0, 11.5, 13.25])).slice(s![2..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub fn ema(price_array: &Array1<f64>, period: usize, smoothing: i32) -> Result<EMA, DigiFiError> {
    let multiplier: f64 = smoothing as f64 / (1.0 + period as f64);
    let mut ema: Vec<f64> = vec![f64::NAN; period - 1];
    ema.push(price_array.slice(s![0..period]).mean().ok_or(DigiFiError::MeanCalculation { title: "EMA".to_owned(), series: "price array slice".to_owned() })?);
    for i in period..price_array.len() {
        ema.push(price_array[i]*multiplier + ema[i-1]*(1.0-multiplier));
    }
    Ok(EMA { period, smoothing, ema: Array1::from_vec(ema) })
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Moving Average Convergence/Divergence (MACD) data.
pub struct MACD {
    /// Size of the rolling window for the smaller EMA
    pub small_ema_period: usize,
    /// Size of the rolling window for the larger EMA
    pub large_ema_period: usize,
    /// Size of rolling window for the signal line
    pub  signal_line_period: usize,
    /// Smoothing of the EMA lines
    pub smoothing: i32,
    /// An array of smaller EMA readings
    pub small_ema: Array1<f64>,
    /// An array of larger EMA readings
    pub large_ema: Array1<f64>,
    /// An array of MACD readings
    pub macd: Array1<f64>,
    /// An array of singal line readings
    pub signal_line: Array1<f64>,
    /// An array of MACD histogram sizes
    pub macd_hist: Array1<f64>,
}


/// Moving Average Convergence/Divergence (MACD) describes changes in the strength, direction, momentum, and duration of a trend.
/// 
/// # Input
/// - `price_array`: Array of prices
/// - `small_ema_period`: Size of the rolling window for the smaller EMA
/// - `large_ema_period`: Size of the rolling window for the larger EMA
/// - `signal_line_period`: Size of rolling window for the signal line
/// - `smoothing`: Smoothing of the EMA lines
/// 
/// # Output
/// - MACD data
/// 
/// # Errors
/// - Returns an error if the large ema pariod is smaller or equal to the small ema period.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/MACD>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::{Array1, s};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::technical_indicators::{ema, macd, MACD};
/// 
/// let price_array: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
/// let macd_info: MACD = macd(&price_array, 3, 4, 2, 2).unwrap();
/// 
/// assert!((macd_info.macd - ema(&price_array, 3, 2).unwrap().ema + ema(&price_array, 4, 2).unwrap().ema).slice(s![3..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((macd_info.signal_line.slice(s![4..]).to_owned() - Array1::from_vec(vec![0.35, 0.43])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub fn macd(price_array: &Array1<f64>, small_ema_period: usize, large_ema_period: usize, signal_line_period: usize, smoothing: i32) -> Result<MACD, DigiFiError> {
    let error_title: String = String::from("MACD");
    if large_ema_period <= small_ema_period {
        return Err(DigiFiError::ParameterConstraint {
            title: error_title,
            constraint: "The argument large_ema_period must be bigger than the argument small_ema_period.".to_owned(),
        });
    }
    let signal_line_mult: f64 = smoothing as f64 / (1.0 + signal_line_period as f64);
    // Small EMA
    let small_ema: Array1<f64> = ema(&price_array, small_ema_period, smoothing)?.ema;
    // Large Ema
    let large_ema: Array1<f64> = ema(&price_array, large_ema_period, smoothing)?.ema;
    // MACD
    let macd: Array1<f64> = &small_ema - &large_ema;
    // Signal Line
    let mut signal_line_: Vec<f64> = vec![f64::NAN; price_array.len()];
    signal_line_[large_ema_period-2+signal_line_period] = macd.slice(s![large_ema_period-1..large_ema_period-1+signal_line_period]).mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title, series: "price array slice".to_owned(), })?;
    for i in (large_ema_period-1+signal_line_period)..small_ema.len() {
        signal_line_[i] = macd[i]*signal_line_mult + signal_line_[i-1]*(1.0-signal_line_mult);
    }
    let signal_line_: Array1<f64> = Array1::from_vec(signal_line_);
    // MACD Histogram
    let macd_hist: Array1<f64> = &macd - &signal_line_;
    Ok(MACD { small_ema_period, large_ema_period, signal_line_period, smoothing, small_ema, large_ema, macd, signal_line: signal_line_, macd_hist })
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Bollinger Band data.
pub struct BollingerBands {
    /// Size of the rolling window for the SMA
    pub period: usize,
    /// Number of standard deviations used to construct Bollinger bands around the SMA
    pub n_std: usize,
    /// An array of SMA readings
    pub sma: Array1<f64>,
    /// An array of upper Bollinger band readings
    pub lower_band: Array1<f64>,
    /// An array of lower Bollinger band readings
    pub upper_band: Array1<f64>,
}


/// Bollinger Band is an SMA with additional upper and lower bands contain price action within n_deviations away from the SMA line.
/// 
/// # Input
/// - `price_array`: An array of prices
/// - `period`: Size of the rolling window for the SMA
/// - `n_std`: Number of standard deviations used to construct Bollinger bands around the SMA
/// 
/// # Output
/// - Bollinger Bands data
/// 
/// # Errors
/// - Returns an error if the mean of SMA slice cannot be computed.
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Bollinger_Bands>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::{Array1, s};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::technical_indicators::{sma, bollinger_bands, BollingerBands};
/// 
/// let price_array: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
/// let deviations: Array1<f64> = 2.0 * Array1::from_vec(vec![f64::NAN, f64::NAN, (2.0/3.0_f64).sqrt(), (2.0/3.0_f64).sqrt(), (2.0/3.0_f64).sqrt(), (8.0/3.0_f64).sqrt()]);
/// let bb_info: BollingerBands = bollinger_bands(&price_array, 3, 2).unwrap();
/// 
/// assert!((bb_info.upper_band - sma(&price_array, 3).unwrap().sma - &deviations).slice(s![2..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub fn bollinger_bands(price_array: &Array1<f64>, period: usize, n_std: usize) -> Result<BollingerBands, DigiFiError> {
    let n_std_: f64 = n_std as f64;
    let sma: Array1<f64> = sma(price_array, period)?.sma;
    let mut deviation: Vec<f64> = price_array.windows(period).into_iter().map(|window| {
        n_std_ * window.std(0.0)
    } ).collect();
    let _: Vec<_> = (0..(period - 1)).map(|_| deviation.insert(0, f64::NAN) ).collect();
    let (upper_band, lower_band) = sma.iter().zip(deviation.iter())
        .map(|(sma, deviation)| (sma + deviation, sma - deviation) ).unzip();
    Ok(BollingerBands { period, n_std, sma, upper_band: Array1::from_vec(upper_band), lower_band: Array1::from_vec(lower_band) })
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Relative Strength Index (RSI) data.
pub struct RSI {
    /// Size of the rolling window for the RSI
    pub period: usize,
    /// Constant value of the oversold band
    pub oversold_band: f64,
    /// Constant value of the overbought band
    pub overbought_band: f64,
    /// An array of upward price changes
    pub u: Array1<f64>,
    /// An array of downward price changes
    pub d: Array1<f64>,
    /// An array of smoothed modified moving average readings of upward price changes
    pub u_smma: Array1<f64>,
    /// An array of smoothed modified moving average readings of downward price changes
    pub d_smma: Array1<f64>,
    /// An array relative strength factor readings
    pub rs: Array1<f64>,
    /// An array of RSI readings
    pub rsi: Array1<f64>,
    /// An array of oversold band readings
    pub oversold: Array1<f64>,
    /// An array of overbought band readings
    pub overbought: Array1<f64>,
}


/// Relative Strength Index (RSI) is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought
/// or oversold conditions.
/// 
/// # Input
/// - `price_array`: An array of prices
/// - `period`: Size of the rolling window for the RSI
/// - `oversold_band`: Constant value of the oversold band
/// - `overbought_band`: Constant value of the overbought band
/// 
/// # Output
/// - RSI data 
///
/// # Errors
/// - Returns an error if the mean of `U` slice cannot be computed.
/// - Returns an error if the mean of `D` slide cannot be computed.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Relative_strength_index>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::{Array1, s};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::technical_indicators::{rsi, RSI};
/// 
/// let price_array: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
/// let rsi_info: RSI = rsi(&price_array, 3, 30.0, 70.0).unwrap();
/// 
/// assert!((rsi_info.u - Array1::from_vec(vec![0.0, 1.0, 1.0, 1.0, 0.0, 4.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((rsi_info.d - Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 2.0, 0.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((rsi_info.u_smma - Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN, 3.0/3.0, (1.0*2.0 + 0.0)/3.0, (2.0/3.0*2.0 + 4.0)/3.0])).slice(s![3..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((rsi_info.d_smma - Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN, 0.0, (0.0*2.0 + 2.0)/3.0, (2.0/3.0*2.0 + 0.0)/3.0])).slice(s![3..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((rsi_info.rs - Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, 1.0, 4.0])).slice(s![4..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((rsi_info.rsi - Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, 50.0, 80.0])).slice(s![4..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub fn rsi(price_array: &Array1<f64>, period: usize, oversold_band: f64, overbought_band: f64) -> Result<RSI, DigiFiError> {
    let error_title: String = String::from("RSI");
    let price_array_length: usize = price_array.len();
    let mut u: Vec<f64> = vec![0.0; price_array_length];
    let mut d: Vec<f64> = vec![0.0; price_array_length];
    let mut u_smma: Vec<f64> = vec![f64::NAN; price_array_length];
    let mut d_smma: Vec<f64> = vec![f64::NAN; price_array_length];
    // U and D
    for i in 1..price_array_length {
        if price_array[i-1] <= price_array[i] {
            u[i] = price_array[i] - price_array[i-1];
        } else {
            d[i] = price_array[i-1] - price_array[i];
        }
    }
    let u: Array1<f64> = Array1::from_vec(u);
    let d: Array1<f64> = Array1::from_vec(d);
    // U SMMA and D SMMA
    u_smma[period] = u.slice(s![1..period+1]).mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "U slice".to_owned(), })?;
    d_smma[period] = d.slice(s![1..period+1]).mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "D slice".to_owned(), })?;
    for i in (period+1)..price_array_length {
        u_smma[i] = (u_smma[i-1]*(period as f64 - 1.0) + u[i]) / period as f64;
        d_smma[i] = (d_smma[i-1]*(period as f64 - 1.0) + d[i]) / period as f64;
    }
    let u_smma: Array1<f64> = Array1::from_vec(u_smma);
    let d_smma: Array1<f64> = Array1::from_vec(d_smma);
    let mut rs: Array1<f64> = &u_smma / &d_smma;
    // Removes division by zero (division by zero results in infinity).
    rs = rs.map(| i | if (*i == f64::INFINITY) || (*i == -f64::INFINITY) { f64::NAN } else { *i });
    let rsi: Array1<f64> = 100.0 - 100.0/(1.0 + &rs);
    Ok(RSI {
        period, oversold_band, overbought_band, u, d, u_smma, d_smma, rs, rsi,
        oversold: Array1::from_vec(vec![oversold_band; price_array_length]), overbought: Array1::from_vec(vec![overbought_band; price_array_length])
    })
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Average Directional Index (ADX) data.
pub struct ADX {
    /// Size of the rolling window for ADX
    pub period: usize,
    /// Constant value of the benchmark array
    pub benchmark: f64,
    /// +DM. An array of directional movement up readings
    pub pdm: Array1<f64>,
    /// -DM. An array of directional movement down readings
    pub mdm: Array1<f64>,
    /// +DI. An array of positive directional indicator readings
    pub pdi: Array1<f64>,
    /// -DI. An array of negative directional indicator readings
    pub mdi: Array1<f64>,
    /// An array of ADX readings
    pub adx: Array1<f64>,
    /// An array of benchmark readings
    pub benchmark_array: Array1<f64>,
}


/// Average Directional Index (ADX) is an indicator that describes the relative strength of the trend.
/// 
/// # Input
/// - `high_price`: An array of high prices
/// - `low_price`: An array of low prices
/// - `close_price`: An array of close prices
/// - `period`: Size of the rolling window for ADX
/// - `benchmark`: Constant value of the benchmark array
/// 
/// # Output
/// - ADX data 
/// 
/// # Errors
/// - Returns an error if the lengths of price arrays do not coincide.
/// - Returns an error if the lengths of price arrays are smaller than `2*period + 1`.
/// - Returns an error if the mean of `TR` slice cannot be computed.
/// - Returns an error if the mean of `+DM` slice cannot be computed.
/// - Returns an error if the mean of `-DM` slice cannot be computed.
/// - Returns an error if the mean of `|+DM - -DM|` slice cannot be computed.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Average_directional_movement_index>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::{Array1, s};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::technical_indicators::{adx, ADX};
/// 
/// let close_price: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
/// let low_price: Array1<f64> = Array1::from_vec(vec![9.5, 10.7, 11.8, 12.6, 10.7, 14.9]);
/// let high_price: Array1<f64> = Array1::from_vec(vec![10.2, 11.4, 12.3, 13.4, 11.3, 15.2]);
/// let adx_info: ADX = adx(&high_price, &low_price, &close_price, 2, 25.0).unwrap();
/// 
/// assert!((adx_info.pdm - Array1::from_vec(vec![0.0, 1.2, 0.9, 1.1, 0.0, 3.9])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((adx_info.mdm - Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.9, 0.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((adx_info.pdi - Array1::from_vec(vec![f64::NAN, f64::NAN, 105.0, 89.58333333333331, 30.71428571428571, 74.57983193277309])).slice(s![2..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((adx_info.mdi - Array1::from_vec(vec![f64::NAN, f64::NAN, 0.0, 0.0, 54.28571428571428, 15.966386554621852])).slice(s![2..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((adx_info.adx - Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, 85.55088702147526, 72.52191286414025])).slice(s![4..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub fn adx(high_price: &Array1<f64>, low_price: &Array1<f64>, close_price: &Array1<f64>, period: usize, benchmark: f64) -> Result<ADX, DigiFiError> {
    let error_title: String = String::from("ADX");
    compare_len(&close_price.iter(), &high_price.iter(), "close_price", "high_price")?;
    compare_len(&close_price.iter(), &low_price.iter(), "close_price", "low_price")?;
    let price_array_length = close_price.len();
    if price_array_length < (2 * period + 1) {
        return Err(DigiFiError::ParameterConstraint { title: error_title,
            constraint: "The argument period must be defined such that the length of price array is smaller than (2*period + 1).".to_owned(), });
    }
    let mut pdm: Vec<f64> = vec![0.0; price_array_length];
    let mut mdm: Vec<f64> = vec![0.0; price_array_length];
    let mut tr: Vec<f64> = vec![0.0; price_array_length];
    let mut atr: Vec<f64> = vec![f64::NAN; price_array_length];
    let mut pdm_smma: Vec<f64> = vec![f64::NAN; price_array_length];
    let mut mdm_smma: Vec<f64> = vec![f64::NAN; price_array_length];
    let mut adx: Array1<f64> = Array1::from_vec(vec![f64::NAN; price_array_length]);
    // +DM, -DM and TR
    for i in 1..price_array_length {
        // +DM and -DM
        let up_move: f64 = high_price[i] - high_price[i-1];
        let down_move: f64 = low_price[i-1] - low_price[i];
        if (down_move < up_move) && (0.0 < up_move) {
            pdm[i] = up_move;
        } 
        if (up_move < down_move) && (0.0 < down_move) {
            mdm[i] = down_move;
        }
        // TR
        tr[i] = f64::max(high_price[i], close_price[i-1]) - f64::min(low_price[i], close_price[i-1]);
    }
    let pdm: Array1<f64> = Array1::from_vec(pdm);
    let mdm: Array1<f64> = Array1::from_vec(mdm);
    let tr: Array1<f64> = Array1::from_vec(tr);
    // ATR
    atr[period-1] = tr.slice(s![0..period]).mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "TR slice".to_owned(), })?;
    for i in period..price_array_length {
        atr[i] = (atr[i-1] * (period as f64 - 1.0) + tr[i]) / period as f64;
    }
    let atr: Array1<f64> = Array1::from_vec(atr);
    // +DM SMMA and -DM SMMA
    pdm_smma[period] = pdm.slice(s![1..period+1]).mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "+DM slice".to_owned(), })?;
    mdm_smma[period] = mdm.slice(s![1..period+1]).mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "-DM slice".to_owned(), })?;
    for i in (period+1)..price_array_length {
        pdm_smma[i] = (pdm_smma[i-1] * (period as f64 - 1.0) + pdm[i]) / period as f64;
        mdm_smma[i] = (mdm_smma[i-1] * (period as f64 - 1.0) + mdm[i]) / period as f64;
    }
    let pdm_smma: Array1<f64> = Array1::from_vec(pdm_smma);
    let mdm_smma: Array1<f64> = Array1::from_vec(mdm_smma);
    // +DI and -DI
    let pdi: Array1<f64> = 100.0 * &pdm_smma / &atr;
    let mdi: Array1<f64> = 100.0 * &mdm_smma / &atr;
    // |+DI - -DI|
    let abs_pdi_mdi: Array1<f64> = (&pdi - &mdi).map(| i | i.abs());
    // ADX
    adx[2*period] = abs_pdi_mdi.slice(s![period..(2*period + 1)]).mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "|+DI - -DI| slice".to_owned(), })?;
    for i in (2*period + 1)..price_array_length {
        adx[i] = (adx[i-1] * (period as f64 - 1.0) + abs_pdi_mdi[i]) / period as f64;
    }
    adx = 100.0 * adx / (&pdi + &mdi);
    Ok(ADX { period, benchmark, pdm, mdm, pdi, mdi, adx, benchmark_array: Array1::from_vec(vec![benchmark; price_array_length]) })
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// On-Balance Volume (OBV) data.
pub struct OBV {
    /// An array of on-balance volume readings
    pub obv: Array1<f64>,
}


/// On-Balance Volume (OBV) is an indicator that describes the relationship between price and volume in the market.
/// 
/// # Input
/// - `close_price`: An array of close prices
/// - `volume`: Volume of the stock
/// 
/// # Output
/// - An array of OBV readings
/// 
/// # Errors
/// - Returns an error if the lengths of close price and volume array do not match.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/On-balance_volume>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::technical_indicators::{OBV, obv};
/// 
/// let close_price: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
/// let volume: Array1<f64> = Array1::from_vec(vec![1000.0, 1200.0, 1100.0, 1250.0, 1260.0, 1110.0]);
/// let obv: OBV = obv(&close_price, &volume).unwrap();
/// 
/// assert!((obv.obv - Array1::from_vec(vec![0.0, 1200.0, 2300.0, 3550.0, 2290.0, 3400.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub fn obv(close_price: &Array1<f64>, volume: &Array1<f64>) -> Result<OBV, DigiFiError> {
    compare_len(&close_price.iter(), &volume.iter(), "close_price", "volume")?;
    let price_array_len = close_price.len();
    let mut obv: Vec<f64> = vec![0.0; price_array_len];
    for i in 1..price_array_len {
        if close_price[i-1] < close_price[i] {
            obv[i] = obv[i-1] + volume[i];
        } else if close_price[i] < close_price[i-1] {
            obv[i] = obv[i-1] - volume[i];
        } else {
            obv[i] = obv[i-1];
        }
    }
    Ok(OBV { obv: Array1::from_vec(obv) })
}


#[cfg(feature = "plotly")]
/// Plots moving average.
///
/// # Input
/// - `ma`: Moving average readings (e.g., SMA, EMA)
/// - `times`: Vector of times
/// - `plot`: Plot (e.g., candlesticj chart) to which the moving average will be added as a trace (if `None`, then the new plot will be created)
///
/// # Output
/// - Moving average plot
///
/// # Examples
///
/// ```rust,ignore
/// use ndarray::Array1;
/// use digifi::technical_indicators::sma;
///
/// #[cfg(feature = "plotly")]
/// fn test_plot_moving_average() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_moving_average;
///
///     // Data
///     let times: Vec<&str> = vec!["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
///         "2024-01-16"];
///     let times: Vec<String> = times.into_iter().map(|v| String::from(v) ).collect();
///     let price_array: Array1<f64> = Array1::from_vec(vec![149.92999267578125, 148.47000122070312, 144.57000732421875, 145.24000549316406, 149.10000610351562,
///         151.3699951171875, 153.72999572753906, 155.17999267578125, 154.6199951171875, 153.16000366210938]);
///
///     // SMA
///     let sma: Array1<f64> = sma(&price_array, 3).unwrap().sma;
///
///     // Moving average plot
///     let plot: Plot = plot_moving_average(&sma, &times, None);
///     plot.show();
/// }
/// ```
pub fn plot_moving_average(ma: &Array1<f64>, times: &Vec<String>, plot: Option<Plot>) -> Plot {
    let ma_trace: Box<dyn Trace> = Scatter::new(times.clone(), ma.to_vec()).name("Moving Average");
    let mut plot: Plot = match plot {
        Some(p) => { p },
        None => {
            let mut plot: Plot = Plot::new();
            let x_axis: Axis = Axis::new().title("Time");
            let y_axis: Axis = Axis::new().title("Price");
            let layout: Layout = Layout::new().title("<b>Moving Average</b>").x_axis(x_axis).y_axis(y_axis);
            plot.set_layout(layout);
            plot
        },
    };
    plot.add_trace(ma_trace);
    plot
}


#[cfg(feature = "plotly")]
/// Plots MACD.
///
/// # Input
/// - `macd`: MACD information (i.e., MACD, signal line and MACD histogram)
/// - `times`: Vector of times
///
/// # Output
/// - MACD plot
///
/// # Examples
///
/// ```rust,ignore
/// use ndarray::Array1;
/// use digifi::technical_indicators::{MACD, macd};
///
/// #[cfg(feature = "plotly")]
/// fn test_plot_macd() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_macd;
///
///     // Data
///     let times: Vec<&str> = vec!["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
///         "2024-01-16"];
///     let times: Vec<String> = times.into_iter().map(|v| String::from(v) ).collect();
///     let price_array: Array1<f64> = Array1::from_vec(vec![149.92999267578125, 148.47000122070312, 144.57000732421875, 145.24000549316406, 149.10000610351562,
///         151.3699951171875, 153.72999572753906, 155.17999267578125, 154.6199951171875, 153.16000366210938]);
///
///     // MACD
///     let macd_: MACD = macd(&price_array, 3, 4, 2, 2).unwrap();
///
///     // MACD plot
///     let plot: Plot = plot_macd(&macd_, &times);
///     plot.show();
/// }
/// ```
pub fn plot_macd(macd: &MACD, times: &Vec<String>) -> Plot {
    let mut plot: Plot = Plot::new();
    // MACD
    let macd_line: Line = Line::new().color(NamedColor::Blue);
    plot.add_trace(Scatter::new(times.clone(), macd.macd.to_vec()).name("MACD").line(macd_line));
    // Signal line
    let signal_line: Line = Line::new().color(NamedColor::Red);
    plot.add_trace(Scatter::new(times.clone(), macd.signal_line.to_vec()).name("MACD Signal Line").line(signal_line));
    // MACD histogram
    plot.add_trace(Bar::new(times.clone(), macd.macd_hist.to_vec()).name("MACD Histogram"));
    // Layout
    let x_axis: Axis = Axis::new().title("Time");
    let y_axis: Axis = Axis::new().title("MACD");
    let layout: Layout = Layout::new().title("<b>MACD</b>").x_axis(x_axis).y_axis(y_axis);
    plot.set_layout(layout);
    plot
}


#[cfg(feature = "plotly")]
/// Plots Bollinger bands.
///
/// # Input
/// - `bb`: Bollinger bands information (e.g., SMA, upper band, lower band)
/// - `times`: Vector of times
/// - `plot`: Plot (e.g., candlesticj chart) to which the Bollinger bands will be added as a trace (if `None`, then the new plot will be created)
///
/// # Output
/// - Bollinger bands plot
///
/// # Examples
///
/// ```rust,ignore
/// use ndarray::Array1;
/// use digifi::technical_indicators::{BollingerBands, bollinger_bands};
///
/// #[cfg(feature = "plotly")]
/// fn test_plot_bollinger_bands() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_bollinger_bands;
///
///     // Data
///     let times: Vec<&str> = vec!["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
///         "2024-01-16"];
///     let times: Vec<String> = times.into_iter().map(|v| String::from(v) ).collect();
///     let price_array: Array1<f64> = Array1::from_vec(vec![149.92999267578125, 148.47000122070312, 144.57000732421875, 145.24000549316406, 149.10000610351562,
///         151.3699951171875, 153.72999572753906, 155.17999267578125, 154.6199951171875, 153.16000366210938]);
///
///     // Bollinger bands
///     let bb: BollingerBands = bollinger_bands(&price_array, 3, 2).unwrap();
///
///     // Bollinger bands plot
///     let plot: Plot = plot_bollinger_bands(&bb, &times, None);
///     plot.show();
/// }
/// ```
pub fn plot_bollinger_bands(bb: &BollingerBands, times: &Vec<String>, plot: Option<Plot>) -> Plot {
    let ma_trace: Box<dyn Trace> = Scatter::new(times.clone(), bb.sma.to_vec()).name("Simple Moving Average");
    let upper_band: Box<dyn Trace> = Scatter::new(times.clone(), bb.upper_band.to_vec()).name("Upper Band");
    let lower_band: Box<dyn Trace> = Scatter::new(times.clone(), bb.lower_band.to_vec()).name("Lower Band");
    let mut plot: Plot = match plot {
        Some(p) => { p },
        None => {
            let mut plot: Plot = Plot::new();
            let x_axis: Axis = Axis::new().title("Time");
            let y_axis: Axis = Axis::new().title("Price");
            let layout: Layout = Layout::new().title("<b>Bollinger Bands</b>").x_axis(x_axis).y_axis(y_axis);
            plot.set_layout(layout);
            plot
        },
    };
    plot.add_trace(ma_trace);
    plot.add_trace(upper_band);
    plot.add_trace(lower_band);
    plot
}


#[cfg(feature = "plotly")]
/// Plots RSI.
///
/// # Input
/// - `rsi`: RSI information (i.e., RSI, oversold and overbought thresholds)
/// - `times`: Vector of times
///
/// # Output
/// - RSI plot
///
/// # Examples
///
/// ```rust,ignore
/// use ndarray::Array1;
/// use digifi::technical_indicators::{RSI, rsi};
///
/// #[cfg(feature = "plotly")]
/// fn test_plot_rsi() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_rsi;
///
///     // Data
///     let times: Vec<&str> = vec!["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
///         "2024-01-16"];
///     let times: Vec<String> = times.into_iter().map(|v| String::from(v) ).collect();
///     let price_array: Array1<f64> = Array1::from_vec(vec![149.92999267578125, 148.47000122070312, 144.57000732421875, 145.24000549316406, 149.10000610351562,
///         151.3699951171875, 153.72999572753906, 155.17999267578125, 154.6199951171875, 153.16000366210938]);
///
///     // RSI
///     let rsi_: RSI = rsi(&price_array, 3, 30.0, 70.0).unwrap();
///
///     // RSI plot
///     let plot: Plot = plot_rsi(&rsi_, &times);
///     plot.show();
/// }
/// ```
pub fn plot_rsi(rsi: &RSI, times: &Vec<String>) -> Plot {
    let mut plot: Plot = Plot::new();
    // RSI
    let rsi_line: Line = Line::new().color(NamedColor::Blue);
    plot.add_trace(Scatter::new(times.clone(), rsi.rsi.to_vec()).name("RSI").line(rsi_line));
    // Oversold
    let oversold_line: Line = Line::new().color(NamedColor::Green);
    plot.add_trace(Scatter::new(times.clone(), rsi.oversold.to_vec()).name("Oversold Threshold").line(oversold_line));
    // Overbought
    let overbought_line: Line = Line::new().color(NamedColor::Red);
    plot.add_trace(Scatter::new(times.clone(), rsi.overbought.to_vec()).name("Overbought Threshold").line(overbought_line));
    // Layout
    let x_axis: Axis = Axis::new().title("Time");
    let y_axis: Axis = Axis::new().title("RSI");
    let layout: Layout = Layout::new().title("<b>RSI</b>").x_axis(x_axis).y_axis(y_axis);
    plot.set_layout(layout);
    plot
}


#[cfg(feature = "plotly")]
/// Plots ADX.
///
/// # Input
/// - `adx`: ADX information (i.e., ADX, -DI, +DI and benchmark threshold)
/// - `times`: Vector of times
///
/// # Output
/// - ADX plot
///
/// # Examples
///
/// ```rust,ignore
/// use ndarray::Array1;
/// use digifi::technical_indicators::{ADX, adx};
///
/// #[cfg(feature = "plotly")]
/// fn test_plot_adx() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_adx;
///
///     // Data
///     let times: Vec<&str> = vec!["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
///         "2024-01-16"];
///     let times: Vec<String> = times.into_iter().map(|v| String::from(v) ).collect();
///     let close_price: Array1<f64> = Array1::from_vec(vec![149.92999267578125, 148.47000122070312, 144.57000732421875, 145.24000549316406, 149.10000610351562,
///         151.3699951171875, 153.72999572753906, 155.17999267578125, 154.6199951171875, 153.16000366210938]);
///     let high_price: Array1<f64> = Array1::from_vec(vec![152.3800048828125, 151.0500030517578, 147.3800048828125, 146.58999633789062, 149.39999389648438,
///         151.7100067138672, 154.4199981689453, 157.1699981689453, 156.1999969482422, 154.99000549316406]);
///     let low_price: Array1<f64> = Array1::from_vec(vec![148.38999938964844, 148.3300018310547, 144.0500030517578, 144.52999877929688, 146.14999389648438,
///         148.2100067138672, 151.8800048828125, 153.1199951171875, 154.00999450683594, 152.14999389648438]);
///
///     // ADX
///     let adx_: ADX = adx(&high_price, &low_price, &close_price, 2, 25.0).unwrap();
///
///     // ADX plot
///     let plot: Plot = plot_adx(&adx_, &times);
///     plot.show();
/// }
/// ```
pub fn plot_adx(adx: &ADX, times: &Vec<String>) -> Plot {
    let mut plot: Plot = Plot::new();
    // ADX
    let adx_line: Line = Line::new().color(NamedColor::Blue);
    plot.add_trace(Scatter::new(times.clone(), adx.adx.to_vec()).name("ADX").line(adx_line));
    // -DI
    let mdi_line: Line = Line::new().color(NamedColor::Red);
    plot.add_trace(Scatter::new(times.clone(), adx.mdi.to_vec()).name("-DI").line(mdi_line));
    // +DI
    let pdi_line: Line = Line::new().color(NamedColor::Green);
    plot.add_trace(Scatter::new(times.clone(), adx.pdi.to_vec()).name("+DI").line(pdi_line));
    // Benchmark
    let benchmark_line: Line = Line::new().color(NamedColor::Black);
    plot.add_trace(Scatter::new(times.clone(), adx.benchmark_array.to_vec()).name("Benchmark").line(benchmark_line));
    // Layout
    let x_axis: Axis = Axis::new().title("Time");
    let y_axis: Axis = Axis::new().title("ADX");
    let layout: Layout = Layout::new().title("<b>ADX</b>").x_axis(x_axis).y_axis(y_axis);
    plot.set_layout(layout);
    plot
}


#[cfg(feature = "plotly")]
/// Plots OBV.
///
/// # Input
/// - `obv`: An array of OBV readings
/// - `times`: Vector of times
///
/// # Output
/// - OBV plot
///
/// # Examples
///
/// ```rust,ignore
/// use ndarray::Array1;
/// use digifi::technical_indicators::obv;
///
/// #[cfg(feature = "plotly")]
/// fn test_plot_obv() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_obv;
///
///     // Data
///     let times: Vec<&str> = vec!["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
///         "2024-01-16"];
///     let times: Vec<String> = times.into_iter().map(|v| String::from(v) ).collect();
///     let close_price: Array1<f64> = Array1::from_vec(vec![149.92999267578125, 148.47000122070312, 144.57000732421875, 145.24000549316406, 149.10000610351562,
///         151.3699951171875, 153.72999572753906, 155.17999267578125, 154.6199951171875, 153.16000366210938]);
///     let volume: Array1<f64> = Array1::from_vec(vec![47339400.0, 49425500.0, 56039800.0, 45124800.0, 46757100.0, 43812600.0, 44421800.0, 49072700.0,
///         40460300.0, 41384600.0]);
///
///     // OBV
///     let obv: Array1<f64> = obv(&close_price, &volume).unwrap().obv;
///
///     // OBV plot
///     let plot: Plot = plot_obv(&obv, &times);
///     plot.show();
/// }
/// ```
pub fn plot_obv(obv: &Array1<f64>, times: &Vec<String>) -> Plot {
    let mut plot: Plot = Plot::new();
    plot.add_trace(Scatter::new(times.clone(), obv.to_vec()).name("OBV"));
    let x_axis: Axis = Axis::new().title("Time");
    let y_axis: Axis = Axis::new().title("OBV");
    let layout: Layout = Layout::new().title("<b>OBV</b>").x_axis(x_axis).y_axis(y_axis);
    plot.set_layout(layout);
    plot
}


#[cfg(test)]
mod tests {
    use ndarray::{Array1, s};
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_maximum_drawdown() -> () {
        use crate::technical_indicators::maximum_drawdown;
        let asset_price: Array1<f64> = Array1::from_vec(vec![100.0, 101.0, 104.0, 102.0, 103.0, 106.0]);
        let max_dd: f64 = maximum_drawdown(&asset_price);
        assert_eq!(max_dd, 100.0*2.0/104.0);
    }

    #[test]
    fn unit_test_sma() -> () {
        use crate::technical_indicators::{SMA, sma};
        let price_array: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
        let sma: SMA = sma(&price_array, 3).unwrap();
        assert!((sma.sma - Array1::from_vec(vec![f64::NAN, f64::NAN, 11.0, 12.0, 12.0, 13.0])).slice(s![2..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_ema() -> () {
        use crate::technical_indicators::{EMA, ema};
        let price_array: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
        let ema: EMA = ema(&price_array, 3, 2).unwrap();
        assert!((ema.ema - Array1::from_vec(vec![f64::NAN, f64::NAN, 11.0, 12.0, 11.5, 13.25])).slice(s![2..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_macd() -> () {
        use crate::technical_indicators::{ema, macd, MACD};
        let price_array: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
        let macd_info: MACD = macd(&price_array, 3, 4, 2, 2).unwrap();
        assert!((macd_info.macd - ema(&price_array, 3, 2).unwrap().ema + ema(&price_array, 4, 2).unwrap().ema).slice(s![3..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((macd_info.signal_line.slice(s![4..]).to_owned() - Array1::from_vec(vec![0.35, 0.43])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_bollinger_bands() -> () {
        use crate::technical_indicators::{sma, bollinger_bands, BollingerBands};
        let price_array: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
        let deviations: Array1<f64> = 2.0 * Array1::from_vec(vec![f64::NAN, f64::NAN, (2.0/3.0_f64).sqrt(), (2.0/3.0_f64).sqrt(), (2.0/3.0_f64).sqrt(), (8.0/3.0_f64).sqrt()]);
        let bb_info: BollingerBands = bollinger_bands(&price_array, 3, 2).unwrap();
        assert!((bb_info.upper_band - sma(&price_array, 3).unwrap().sma - &deviations).slice(s![2..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_rsi() -> () {
        use crate::technical_indicators::{rsi, RSI};
        let price_array: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
        let rsi_info: RSI = rsi(&price_array, 3, 30.0, 70.0).unwrap();
        assert!((rsi_info.u - Array1::from_vec(vec![0.0, 1.0, 1.0, 1.0, 0.0, 4.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((rsi_info.d - Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 2.0, 0.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((rsi_info.u_smma - Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN, 3.0/3.0, (1.0*2.0 + 0.0)/3.0, (2.0/3.0*2.0 + 4.0)/3.0])).slice(s![3..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((rsi_info.d_smma - Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN, 0.0, (0.0*2.0 + 2.0)/3.0, (2.0/3.0*2.0 + 0.0)/3.0])).slice(s![3..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((rsi_info.rs - Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, 1.0, 4.0])).slice(s![4..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((rsi_info.rsi - Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, 50.0, 80.0])).slice(s![4..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_adx() -> () {
        use crate::technical_indicators::{adx, ADX};
        let close_price: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
        let low_price: Array1<f64> = Array1::from_vec(vec![9.5, 10.7, 11.8, 12.6, 10.7, 14.9]);
        let high_price: Array1<f64> = Array1::from_vec(vec![10.2, 11.4, 12.3, 13.4, 11.3, 15.2]);
        let adx_info: ADX = adx(&high_price, &low_price, &close_price, 2, 25.0).unwrap();
        assert!((adx_info.pdm - Array1::from_vec(vec![0.0, 1.2, 0.9, 1.1, 0.0, 3.9])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((adx_info.mdm - Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.9, 0.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((adx_info.pdi - Array1::from_vec(vec![f64::NAN, f64::NAN, 105.0, 89.58333333333331, 30.71428571428571, 74.57983193277309])).slice(s![2..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((adx_info.mdi - Array1::from_vec(vec![f64::NAN, f64::NAN, 0.0, 0.0, 54.28571428571428, 15.966386554621852])).slice(s![2..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((adx_info.adx - Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, 85.55088702147526, 72.52191286414025])).slice(s![4..]).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_obv() -> () {
        use crate::technical_indicators::{OBV, obv};
        let close_price: Array1<f64> = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 11.0, 15.0]);
        let volume: Array1<f64> = Array1::from_vec(vec![1000.0, 1200.0, 1100.0, 1250.0, 1260.0, 1110.0]);
        let obv: OBV = obv(&close_price, &volume).unwrap();
        assert!((obv.obv - Array1::from_vec(vec![0.0, 1200.0, 2300.0, 3550.0, 2290.0, 3400.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_plot_moving_average() -> () {
        use plotly::Plot;
        use crate::technical_indicators::{sma, plot_moving_average};
        // Data
        let times: Vec<&str> = vec!["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
            "2024-01-16"];
        let times: Vec<String> = times.into_iter().map(|v| String::from(v) ).collect();
        let price_array: Array1<f64> = Array1::from_vec(vec![149.92999267578125, 148.47000122070312, 144.57000732421875, 145.24000549316406, 149.10000610351562,
            151.3699951171875, 153.72999572753906, 155.17999267578125, 154.6199951171875, 153.16000366210938]);
        // SMA
        let sma: Array1<f64> = sma(&price_array, 3).unwrap().sma;
        // Moving average plot
        let plot: Plot = plot_moving_average(&sma, &times, None);
        plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_plot_macd() -> () {
        use plotly::Plot;
        use crate::technical_indicators::{MACD, macd, plot_macd};
        // Data
        let times: Vec<&str> = vec!["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
            "2024-01-16"];
        let times: Vec<String> = times.into_iter().map(|v| String::from(v) ).collect();
        let price_array: Array1<f64> = Array1::from_vec(vec![149.92999267578125, 148.47000122070312, 144.57000732421875, 145.24000549316406, 149.10000610351562,
            151.3699951171875, 153.72999572753906, 155.17999267578125, 154.6199951171875, 153.16000366210938]);
        // MACD
        let macd: MACD = macd(&price_array, 3, 4, 2, 2).unwrap();
        // Moving average plot
        let plot: Plot = plot_macd(&macd, &times);
        plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_plot_bollinger_bands() -> () {
        use plotly::Plot;
        use crate::technical_indicators::{BollingerBands, bollinger_bands, plot_bollinger_bands};
        // Data
        let times: Vec<&str> = vec!["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
            "2024-01-16"];
        let times: Vec<String> = times.into_iter().map(|v| String::from(v) ).collect();
        let price_array: Array1<f64> = Array1::from_vec(vec![149.92999267578125, 148.47000122070312, 144.57000732421875, 145.24000549316406, 149.10000610351562,
            151.3699951171875, 153.72999572753906, 155.17999267578125, 154.6199951171875, 153.16000366210938]);
        // Bollinger bands
        let bb: BollingerBands = bollinger_bands(&price_array, 3, 2).unwrap();
        // Moving average plot
        let plot: Plot = plot_bollinger_bands(&bb, &times, None);
        plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_plot_rsi() -> () {
        use plotly::Plot;
        use crate::technical_indicators::{RSI, rsi, plot_rsi};
        // Data
        let times: Vec<&str> = vec!["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
            "2024-01-16"];
        let times: Vec<String> = times.into_iter().map(|v| String::from(v) ).collect();
        let price_array: Array1<f64> = Array1::from_vec(vec![149.92999267578125, 148.47000122070312, 144.57000732421875, 145.24000549316406, 149.10000610351562,
            151.3699951171875, 153.72999572753906, 155.17999267578125, 154.6199951171875, 153.16000366210938]);
        // RSI
        let rsi_: RSI = rsi(&price_array, 3, 30.0, 70.0).unwrap();
        // Moving average plot
        let plot: Plot = plot_rsi(&rsi_, &times);
        plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_plot_adx() -> () {
        use plotly::Plot;
        use crate::technical_indicators::{ADX, adx, plot_adx};
        // Data
        let times: Vec<&str> = vec!["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
            "2024-01-16"];
        let times: Vec<String> = times.into_iter().map(|v| String::from(v) ).collect();
        let close_price: Array1<f64> = Array1::from_vec(vec![149.92999267578125, 148.47000122070312, 144.57000732421875, 145.24000549316406, 149.10000610351562,
            151.3699951171875, 153.72999572753906, 155.17999267578125, 154.6199951171875, 153.16000366210938]);
        let high_price: Array1<f64> = Array1::from_vec(vec![152.3800048828125, 151.0500030517578, 147.3800048828125, 146.58999633789062, 149.39999389648438,
            151.7100067138672, 154.4199981689453, 157.1699981689453, 156.1999969482422, 154.99000549316406]);
        let low_price: Array1<f64> = Array1::from_vec(vec![148.38999938964844, 148.3300018310547, 144.0500030517578, 144.52999877929688, 146.14999389648438,
            148.2100067138672, 151.8800048828125, 153.1199951171875, 154.00999450683594, 152.14999389648438]);
        // ADX
        let adx_: ADX = adx(&high_price, &low_price, &close_price, 2, 25.0).unwrap();
        // Moving average plot
        let plot: Plot = plot_adx(&adx_, &times);
        plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_plot_obv() -> () {
        use plotly::Plot;
        use crate::technical_indicators::{obv, plot_obv};
        // Data
        let times: Vec<&str> = vec!["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
            "2024-01-16"];
        let times: Vec<String> = times.into_iter().map(|v| String::from(v) ).collect();
        let close_price: Array1<f64> = Array1::from_vec(vec![149.92999267578125, 148.47000122070312, 144.57000732421875, 145.24000549316406, 149.10000610351562,
            151.3699951171875, 153.72999572753906, 155.17999267578125, 154.6199951171875, 153.16000366210938]);
        let volume: Array1<f64> = Array1::from_vec(vec![47339400.0, 49425500.0, 56039800.0, 45124800.0, 46757100.0, 43812600.0, 44421800.0, 49072700.0,
            40460300.0, 41384600.0]);
        // OBV
        let obv: Array1<f64> = obv(&close_price, &volume).unwrap().obv;
        // Moving average plot
        let plot: Plot = plot_obv(&obv, &times);
        plot.show();
    }
}