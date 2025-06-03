use std::cmp;
use ndarray::{Array1, Array2, s};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;
use crate::utilities::LossFunction;
use crate::utilities::{
    compare_array_len,
    loss_functions::SSE,
    maths_utils::differencing,
};
use crate::statistics::linear_regression;


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Confidence interval for the augmented Dickey-Fuller test.
pub enum ADFConfidence {
    /// 0.1 Confidence interval
    Ten,
    /// 0.05 confidence interval
    Five,
    /// 0.025 confidence interval
    TwoHalf,
    /// 0.01 confidence interval
    One,
}


#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Type of augmented Dickey-Fuller test.
pub enum ADFType {
    /// Adds constant term to the regression
    Constant,
    /// Adds constant term and deterministic time trend to the regression
    ConstantAndTrend,
    /// Does not add additional constraints to the regression
    Simple,
}

impl ADFType {

    /// # Description
    /// Returns that critical value from the Dickey-Fuller table that is used as a confidence interval to test Dickey-Fuller statistic against.
    /// 
    /// # Input
    /// - `n`: Number of data points in the series
    /// - `interval`: Confidence interval
    pub fn get_critical_value(&self, n: usize, interval: &ADFConfidence) -> f64 {
        match self {
            ADFType::Simple => {
                if n <= 25 {
                    match interval {
                        ADFConfidence::Ten => -1.609,
                        ADFConfidence::Five => -1.955,
                        ADFConfidence::TwoHalf => -2.273,
                        ADFConfidence::One => -2.661,
                    }
                } else if n <= 50 {
                    match interval {
                        ADFConfidence::Ten => -1.612,
                        ADFConfidence::Five => -1.947,
                        ADFConfidence::TwoHalf => -2.246,
                        ADFConfidence::One => -2.612,
                    }
                } else if n <= 100 {
                    match interval {
                        ADFConfidence::Ten => -1.614,
                        ADFConfidence::Five => -1.944,
                        ADFConfidence::TwoHalf => -2.234,
                        ADFConfidence::One => -2.588,
                    }
                } else if n <= 250 {
                    match interval {
                        ADFConfidence::Ten => -1.616,
                        ADFConfidence::Five => -1.942,
                        ADFConfidence::TwoHalf => -2.227,
                        ADFConfidence::One => -2.575,
                    }
                } else if n <= 500 {
                    match interval {
                        ADFConfidence::Ten => -1.616,
                        ADFConfidence::Five => -1.942,
                        ADFConfidence::TwoHalf => -2.224,
                        ADFConfidence::One => -2.570,
                    }
                } else {
                    match interval {
                        ADFConfidence::Ten => -1.616,
                        ADFConfidence::Five => -1.941,
                        ADFConfidence::TwoHalf => -2.223,
                        ADFConfidence::One => -2.567,
                    }
                }
            },
            ADFType::Constant => {
                if n <= 25 {
                    match interval {
                        ADFConfidence::Ten => -2.633,
                        ADFConfidence::Five => -2.986,
                        ADFConfidence::TwoHalf => -3.318,
                        ADFConfidence::One => -3.724,
                    }
                } else if n <= 50 {
                    match interval {
                        ADFConfidence::Ten => -2.599,
                        ADFConfidence::Five => -2.921,
                        ADFConfidence::TwoHalf => -3.213,
                        ADFConfidence::One => -3.568,
                    }
                } else if n <= 100 {
                    match interval {
                        ADFConfidence::Ten => -2.582,
                        ADFConfidence::Five => -2.891,
                        ADFConfidence::TwoHalf => -3.164,
                        ADFConfidence::One => -3.498,
                    }
                } else if n <= 250 {
                    match interval {
                        ADFConfidence::Ten => -2.573,
                        ADFConfidence::Five => -2.873,
                        ADFConfidence::TwoHalf => -3.136,
                        ADFConfidence::One => -3.457,
                    }
                } else if n <= 500 {
                    match interval {
                        ADFConfidence::Ten => -2.570,
                        ADFConfidence::Five => -2.867,
                        ADFConfidence::TwoHalf => -3.127,
                        ADFConfidence::One => -3.443,
                    }
                } else {
                    match interval {
                        ADFConfidence::Ten => -2.568,
                        ADFConfidence::Five => -2.863,
                        ADFConfidence::TwoHalf => -3.120,
                        ADFConfidence::One => -3.434,
                    }
                }
            },
            ADFType::ConstantAndTrend => {
                if n <= 25 {
                    match interval {
                        ADFConfidence::Ten => -3.238,
                        ADFConfidence::Five => -3.589,
                        ADFConfidence::TwoHalf => -3.943,
                        ADFConfidence::One => -4.375,
                    }
                } else if n <= 50 {
                    match interval {
                        ADFConfidence::Ten => -3.181,
                        ADFConfidence::Five => -3.495,
                        ADFConfidence::TwoHalf => -3.791,
                        ADFConfidence::One => -4.152,
                    }
                } else if n <= 100 {
                    match interval {
                        ADFConfidence::Ten => -3.153,
                        ADFConfidence::Five => -3.452,
                        ADFConfidence::TwoHalf => -3.722,
                        ADFConfidence::One => -4.052,
                    }
                } else if n <= 250 {
                    match interval {
                        ADFConfidence::Ten => -3.137,
                        ADFConfidence::Five => -3.427,
                        ADFConfidence::TwoHalf => -3.683,
                        ADFConfidence::One => -3.995,
                    }
                } else if n <= 500 {
                    match interval {
                        ADFConfidence::Ten => -3.132,
                        ADFConfidence::Five => -3.419,
                        ADFConfidence::TwoHalf => -3.670,
                        ADFConfidence::One => -3.977,
                    }
                } else {
                    match interval {
                        ADFConfidence::Ten => -3.128,
                        ADFConfidence::Five => -3.413,
                        ADFConfidence::TwoHalf => -3.660,
                        ADFConfidence::One => -3.963,
                    }
                }
            },
        }
        
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Result of the augmented Dickey-Fuller test.
pub struct ADFResult {
    /// Result of the augmented Dickey-Fuller test
    pub unit_root_exists: bool,
    /// Dickey-Fuller tau statistic
    pub df_statistic: Option<f64>,
    /// Critical value that acts as a confidence interval
    pub critical_value: Option<f64>,
    /// Standard error of estimated gamma coefficient
    pub se_gamma: Option<f64>,
    /// Coefficient of `t-1` step
    pub gamma: f64,
    /// Coefficients of autoregressive components of the linear regression
    pub deltas: Vec<f64>,
    /// Coefficient of the time trend
    pub beta: Option<f64>,
    /// Constant term
    pub alpha: Option<f64>,
    /// Type of augmented Dickey-Fuller test that was performed.
    pub adf_type: ADFType,
}

impl ADFResult {

    /// # Description
    /// Constructs `ADFResult` from parameters of the linear regression.
    pub fn new(params: &Array1<f64>, unit_root_exists: bool, df_statistic: Option<f64>, critical_value: Option<f64>, se_gamma: Option<f64>, adf_type: &ADFType) -> Result<Self, DigiFiError> {
        let error_title: String = String::from("ADFResult From Params");
        match adf_type {
            ADFType::Simple => {
                let mut deltas: Vec<f64> = params.to_vec();
                let gamma: f64 = deltas.pop().ok_or(DigiFiError::Other { title: error_title.clone(), details: "Parameters of the linear regression were empty.".to_owned() })?;
                Ok(ADFResult { unit_root_exists, df_statistic, critical_value, se_gamma, gamma, deltas, beta: None, alpha: None, adf_type: adf_type.clone(), })
            },
            ADFType::Constant => {
                let mut deltas: Vec<f64> = params.to_vec();
                let gamma: f64 = deltas.pop().ok_or(DigiFiError::Other { title: error_title.clone(), details: "Parameters of the linear regression were empty.".to_owned() })?;
                let alpha: Option<f64> = deltas.pop();
                Ok(ADFResult { unit_root_exists, df_statistic, critical_value, se_gamma, gamma, deltas, beta: None, alpha, adf_type: adf_type.clone(), })
            },
            ADFType::ConstantAndTrend => {
                let mut deltas: Vec<f64> = params.to_vec();
                let gamma: f64 = deltas.pop().ok_or(DigiFiError::Other { title: error_title.clone(), details: "Parameters of the linear regression were empty.".to_owned() })?;
                let alpha: Option<f64> = deltas.pop();
                let beta: Option<f64> = deltas.pop();
                Ok(ADFResult { unit_root_exists, df_statistic, critical_value, se_gamma, gamma, deltas, beta, alpha, adf_type: adf_type.clone(), })
            },
        }
    }
}


/// # Description
/// Augmented Dickey–Fuller test (ADF) tests the null hypothesis that a unit root is present in a time series sample.
/// The alternative hypothesis depends on which version of the test is used, but is usually stationarity or trend-stationarity.
/// 
/// # Input
/// - `x`: Time series to perform the test on
/// - `lag`: Number of lagging differencing terms to use in the linear regression
/// - `adf_type`: Type of augmented Dickey-Fuller test to perform
/// - `ci`: Confidence interval in the Dickey-Fuller table that will be used to compare the Dickey-Fuller statistic against for the hypothesis test
/// 
/// # Output
/// - Parameters of linear regression and test result
/// 
/// # LaTeX Formula
/// - Full Model: \\Delta y_{t} = \\alpha + \\beta y_{t-1} + \\delta_{1}\\Delta y_{t-1} + ... + \\delta_{p-1}\\Delta_{t-p-1} + \\epsilon_{t}
/// - DF Statistic: DF_{\\tau} = \\frac{\\hat{\\gamma}}{SE(\\hat{\\gamma})}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test>
/// - Original Source: <https://doi.org/10.2307/2286348>
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::statistics::stat_tests::{ADFConfidence, ADFType, ADFResult, adf};
///
/// let x: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 6.0, 5.0, 6.0, 7.0, 8.0]);
/// let result: ADFResult = adf(&x, Some(1), &ADFType::Simple, &ADFConfidence::One).unwrap();
/// 
/// assert_eq!(result.unit_root_exists, false);
/// ```
pub fn adf(x: &Array1<f64>, lag: Option<usize>, adf_type: &ADFType, ci: &ADFConfidence) -> Result<ADFResult, DigiFiError> {
    let error_title: String = String::from("ADF");
    let x_len: usize = x.len();
    let lag: usize = match lag {
        Some(v) => {
            if x_len < v {
                return Err(DigiFiError::IndexOutOfRange { title: error_title.clone(), index: "lag".to_owned(), array: "v".to_owned(), });
            }
            v
        },
        None => cmp::max((12.0 * (x_len as f64 / 100.0).powf(0.25)) as usize, 0),
    };
    let diff: Array1<f64> = differencing(&x, 1)?;
    // Left side of linear regression model
    let lr_result: Array1<f64> = diff.slice(s![lag..]).iter().map(|v_| *v_ ).collect();
    // Right side of the linear regression model
    let mut lr_matrix: Vec<Vec<f64>> = Vec::<Vec<f64>>::new();
    match adf_type {
        ADFType::Simple => {
            // Delta terms
            for j in 0..lag {
                let upper_index: usize = diff.len() - 1 - j;
                lr_matrix.push(diff.slice(s![upper_index-lr_result.len()..upper_index]).into_iter().map(|v_| *v_ ).collect());
            }
            // Gamma term
            lr_matrix.push(x.slice(s![lag..(x_len-1)]).into_iter().map(|v_| *v_ ).collect());
        },
        ADFType::Constant => {
            // Delta terms
            for j in 0..lag {
                let upper_index: usize = diff.len() - 1 - j;
                lr_matrix.push(diff.slice(s![upper_index-lr_result.len()..upper_index]).into_iter().map(|v_| *v_ ).collect());
            }
            // Gamma term
            lr_matrix.push(x.slice(s![lag..(x_len-1)]).into_iter().map(|v_| *v_ ).collect());
            // Constant term (Alpha)
            lr_matrix.push(vec![1.0; lr_result.len()]);
        },
        ADFType::ConstantAndTrend => {
            // Delta terms
            for j in 0..lag {
                let upper_index: usize = diff.len() - 1 - j;
                lr_matrix.push(diff.slice(s![upper_index-lr_result.len()..upper_index]).into_iter().map(|v_| *v_ ).collect());
            }
            // Constant trend term (Beta)
            lr_matrix.push((x_len-lr_result.len()+1..=x_len).map(|v| v as f64 ).collect());
            // Constant term (Alpha)
            lr_matrix.push(vec![1.0; lr_result.len()]);
            // Gamma term
            lr_matrix.push(x.slice(s![lag..(x_len-1)]).into_iter().map(|v_| *v_ ).collect());
        },
    }
    // Linear regression
    let data_matrix: Array2<f64> = Array2::from_shape_vec((lr_matrix.len(), lr_matrix[0].len()), lr_matrix.into_iter().flatten().collect())?;
    let params: Array1<f64> = linear_regression(&data_matrix.t().to_owned(), &lr_result)?;
    // If gamma is positive no uni root exists as the process is not stationary
    let gamma: f64 = params[params.len()-1];
    if 0.0 < gamma {
        return ADFResult::new(&params, false, None, None, None, adf_type);
    }
    // Compute standard error for estimated gamma
    let prediction: Array1<f64> = data_matrix.t().dot(&params);
    let loss_function: SSE = SSE;
    let denominator: f64 = (lr_result.len() - params.len()) as f64; // Number of data points minus degrees of freedom
    let estimated_var: f64 = loss_function.loss_array(&lr_result, &prediction)? / denominator;
    let mean_of_y_t_minus_1: f64 = data_matrix.slice(s![data_matrix.dim().0-1, ..]).mean()
        .ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "y_t_minus_1".to_owned(), })?;
    let estimated_var_gamma_denominator: f64 = data_matrix.slice(s![data_matrix.dim().0-1, ..]).map(|v| (v - mean_of_y_t_minus_1).powi(2) ).sum();
    let se_gamma: f64 = (estimated_var / estimated_var_gamma_denominator).sqrt();
    // Calculate Dickey-Fuller statistic (i.e., gamma / SE(gamma))
    let df_statistic: f64 = gamma / se_gamma;
    let critical_value: f64 = adf_type.get_critical_value(x_len, ci);
    let unit_root_exists: bool = if df_statistic < critical_value { false } else { true };
    ADFResult::new(&params, unit_root_exists, Some(df_statistic), Some(critical_value), Some(se_gamma), adf_type)
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Result of the cointegration test.
pub struct CointegrationResult {
    pub cointegrated: bool,
    pub adf_results: Option<Vec<ADFResult>>,
}


/// # Description
/// Cointegration is a statistical property describing a long-term, stable relationship between two or more time series variables,
/// even if those variables themselves are individually non-stationary (i.e., they have trends). This means that despite their individual fluctuations,
/// the variables move together in the long run, anchored by an underlying equilibrium relationship.
/// 
/// # Input
/// - `x`: Time series that is first-order integrated (i.e., I(0))
/// - `y`: Time series that is first-order integrated (i.e., I(0))
/// - `ci`: Confidence interval in the Dickey-Fuller table that will be used to compare the Dickey-Fuller statistic against for the hypothesis test
/// 
/// # Output
/// - Cointegration result, which includes the results of the intermediate tests
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Cointegration>
/// - Original Source: <https://doi.org/10.2307/1913236>
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::statistics::stat_tests::{ADFConfidence, CointegrationResult, cointegration};
///
/// #[cfg(feature = "sample_data")]
/// fn test_capm() -> () {
///     use digifi::utilities::SampleData;
///
///     // Get test data
///     let sample: SampleData = SampleData::Portfolio; 
///     let (_, mut sample_data) = sample.load_sample_data();
///     let jpm: Array1<f64> = sample_data.remove("JPM").unwrap();
///     let gs: Array1<f64> = sample_data.remove("GS").unwrap();
///     
///     // Cointegration test
///     let cointegration_result: CointegrationResult = cointegration(&jpm, &gs, &ADFConfidence::One).unwrap();
///     
///     assert_eq!(cointegration_result.cointegrated, true);
/// }
/// ```
pub fn cointegration(x: &Array1<f64>, y: &Array1<f64>, ci: &ADFConfidence) -> Result<CointegrationResult, DigiFiError> {
    compare_array_len(x, y, "x", "y")?;
    let data_matrix: Array2<f64> = Array2::from_shape_vec((x.len(), 1), x.to_vec())?;
    let params: Array1<f64> = linear_regression(&data_matrix, y)?;
    let prediction: Array1<f64> = data_matrix.dot(&params);
    let u_t: Array1<f64> = y - &prediction;
    // Run ADF test with different configurations (i.e., different types of augmented Dickey-Fuller test)
    let adf_variations: Vec<ADFType> = vec![ADFType::Simple, ADFType::Constant, ADFType::ConstantAndTrend];
    let mut cointegrated: bool = false;
    let mut adf_results: Vec<ADFResult> = Vec::<ADFResult>::new();
    for adf_type in &adf_variations {
        let adf_result: ADFResult = adf(&u_t, None, adf_type, ci)?;
        let stationary_u_t: bool = !adf_result.unit_root_exists;
        adf_results.push(adf_result);
        // If unit root does not exists then `u_t` is stationary, so the processes `x` and `y` are cointegrated
        if stationary_u_t {
            cointegrated = true;
            break;
        }
    }
    Ok(CointegrationResult { cointegrated, adf_results: Some(adf_results), })
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;

    #[test]
    fn unit_test_adf() -> () {
        use crate::statistics::stat_tests::{ADFConfidence, ADFType, ADFResult, adf};
        let x: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 6.0, 5.0, 6.0, 7.0, 8.0]);
        let result: ADFResult = adf(&x, Some(1), &ADFType::Simple, &ADFConfidence::One).unwrap();
        assert_eq!(result.unit_root_exists, false);
    }

    #[cfg(all(test, feature = "sample_data"))]
    #[test]
    fn unit_test_cointegration() -> () {
        use crate::utilities::sample_data::SampleData;
        use crate::statistics::stat_tests::{ADFConfidence, CointegrationResult, cointegration};
        let sample: SampleData = SampleData::Portfolio; 
        let (_, mut sample_data) = sample.load_sample_data();
        let jpm: Array1<f64> = sample_data.remove("JPM").unwrap();
        let gs: Array1<f64> = sample_data.remove("GS").unwrap();
        let cointegration_result: CointegrationResult = cointegration(&jpm, &gs, &ADFConfidence::One).unwrap();
        assert_eq!(cointegration_result.cointegrated, true);
    }
}