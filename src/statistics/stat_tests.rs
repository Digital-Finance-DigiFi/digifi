// Re-Exports
pub use self::t_test::{TTestResult, TTestTwoSampleCase, t_test_two_sample, t_test_lr};
pub use self::f_test::{nested_f_statistic, FTestResult, f_test_anova};
pub use self::granger_causality::{
    GrangerCausalityTestType, GrangerCausalityRejectReason, GrangerCausalityResult, GrangerCausalitySettings,
    granger_causality_test, simple_granger_causality_test,
};

mod f_test;
mod t_test;
mod granger_causality;


use std::{cmp, fmt::Display};
use ndarray::{Array1, Array2, s};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::{LARGE_TEXT_BREAK, compare_len, FeatureCollection, data_transformations::differencing};
use crate::statistics::{
    linear_regression,
    linear_regression_analysis::{LinearRegressionSettings, LinearRegressionFeatureResult, LinearRegressionResult, LinearRegressionAnalysis}, 
};


#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Confidence interval for different tests.
pub enum ConfidenceLevel {
    /// 0.1 Confidence interval
    Ten,
    /// 0.05 confidence interval
    #[default]
    Five,
    /// 0.025 confidence interval
    TwoHalf,
    /// 0.01 confidence interval
    One,
}

impl ConfidenceLevel {
    pub fn get_p(&self) -> f64 {
        match self {
            ConfidenceLevel::Ten => 0.1,
            ConfidenceLevel::Five => 0.05,
            ConfidenceLevel::TwoHalf => 0.25,
            ConfidenceLevel::One => 0.01,
        }
    }
}

impl Display for ConfidenceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get_p().to_string())
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

    /// Returns that critical value from the Dickey-Fuller table that is used as a confidence interval to test Dickey-Fuller statistic against.
    /// 
    /// # Input
    /// - `n`: Number of data points in the series
    /// - `interval`: Confidence interval
    pub fn get_critical_value(&self, n: usize, confidence_level: &ConfidenceLevel) -> f64 {
        match self {
            ADFType::Simple => {
                if n <= 25 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -1.609,
                        ConfidenceLevel::Five => -1.955,
                        ConfidenceLevel::TwoHalf => -2.273,
                        ConfidenceLevel::One => -2.661,
                    }
                } else if n <= 50 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -1.612,
                        ConfidenceLevel::Five => -1.947,
                        ConfidenceLevel::TwoHalf => -2.246,
                        ConfidenceLevel::One => -2.612,
                    }
                } else if n <= 100 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -1.614,
                        ConfidenceLevel::Five => -1.944,
                        ConfidenceLevel::TwoHalf => -2.234,
                        ConfidenceLevel::One => -2.588,
                    }
                } else if n <= 250 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -1.616,
                        ConfidenceLevel::Five => -1.942,
                        ConfidenceLevel::TwoHalf => -2.227,
                        ConfidenceLevel::One => -2.575,
                    }
                } else if n <= 500 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -1.616,
                        ConfidenceLevel::Five => -1.942,
                        ConfidenceLevel::TwoHalf => -2.224,
                        ConfidenceLevel::One => -2.570,
                    }
                } else {
                    match confidence_level {
                        ConfidenceLevel::Ten => -1.616,
                        ConfidenceLevel::Five => -1.941,
                        ConfidenceLevel::TwoHalf => -2.223,
                        ConfidenceLevel::One => -2.567,
                    }
                }
            },
            ADFType::Constant => {
                if n <= 25 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -2.633,
                        ConfidenceLevel::Five => -2.986,
                        ConfidenceLevel::TwoHalf => -3.318,
                        ConfidenceLevel::One => -3.724,
                    }
                } else if n <= 50 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -2.599,
                        ConfidenceLevel::Five => -2.921,
                        ConfidenceLevel::TwoHalf => -3.213,
                        ConfidenceLevel::One => -3.568,
                    }
                } else if n <= 100 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -2.582,
                        ConfidenceLevel::Five => -2.891,
                        ConfidenceLevel::TwoHalf => -3.164,
                        ConfidenceLevel::One => -3.498,
                    }
                } else if n <= 250 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -2.573,
                        ConfidenceLevel::Five => -2.873,
                        ConfidenceLevel::TwoHalf => -3.136,
                        ConfidenceLevel::One => -3.457,
                    }
                } else if n <= 500 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -2.570,
                        ConfidenceLevel::Five => -2.867,
                        ConfidenceLevel::TwoHalf => -3.127,
                        ConfidenceLevel::One => -3.443,
                    }
                } else {
                    match confidence_level {
                        ConfidenceLevel::Ten => -2.568,
                        ConfidenceLevel::Five => -2.863,
                        ConfidenceLevel::TwoHalf => -3.120,
                        ConfidenceLevel::One => -3.434,
                    }
                }
            },
            ADFType::ConstantAndTrend => {
                if n <= 25 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -3.238,
                        ConfidenceLevel::Five => -3.589,
                        ConfidenceLevel::TwoHalf => -3.943,
                        ConfidenceLevel::One => -4.375,
                    }
                } else if n <= 50 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -3.181,
                        ConfidenceLevel::Five => -3.495,
                        ConfidenceLevel::TwoHalf => -3.791,
                        ConfidenceLevel::One => -4.152,
                    }
                } else if n <= 100 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -3.153,
                        ConfidenceLevel::Five => -3.452,
                        ConfidenceLevel::TwoHalf => -3.722,
                        ConfidenceLevel::One => -4.052,
                    }
                } else if n <= 250 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -3.137,
                        ConfidenceLevel::Five => -3.427,
                        ConfidenceLevel::TwoHalf => -3.683,
                        ConfidenceLevel::One => -3.995,
                    }
                } else if n <= 500 {
                    match confidence_level {
                        ConfidenceLevel::Ten => -3.132,
                        ConfidenceLevel::Five => -3.419,
                        ConfidenceLevel::TwoHalf => -3.670,
                        ConfidenceLevel::One => -3.977,
                    }
                } else {
                    match confidence_level {
                        ConfidenceLevel::Ten => -3.128,
                        ConfidenceLevel::Five => -3.413,
                        ConfidenceLevel::TwoHalf => -3.660,
                        ConfidenceLevel::One => -3.963,
                    }
                }
            },
        }
        
    }
}

impl Display for ADFType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ADFType::Simple => write!(f, "Simple (No constant or trend terms)"),
            ADFType::Constant => write!(f, "Constant (Constant term in the ADF test regression)"),
            ADFType::ConstantAndTrend => write!(f, "Constant and Trend (Constant and trend terms in the ADF test regression)"),
        }
    }
}


#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

    /// Constructs `ADFResult` from parameters of the linear regression.
    pub fn build(params: &Array1<f64>, unit_root_exists: bool, df_statistic: Option<f64>, critical_value: Option<f64>, se_gamma: Option<f64>, adf_type: &ADFType) -> Result<Self, DigiFiError> {
        match adf_type {
            ADFType::Simple => {
                let mut deltas: Vec<f64> = params.to_vec();
                let gamma: f64 = deltas.pop().ok_or(DigiFiError::Other {
                    title: Self::error_title(),
                    details: "Parameters of the linear regression were empty.".to_owned(),
                })?;
                Ok(ADFResult { unit_root_exists, df_statistic, critical_value, se_gamma, gamma, deltas, beta: None, alpha: None, adf_type: adf_type.clone(), })
            },
            ADFType::Constant => {
                let mut deltas: Vec<f64> = params.to_vec();
                let gamma: f64 = deltas.pop().ok_or(DigiFiError::Other {
                    title: Self::error_title(),
                    details: "Parameters of the linear regression were empty.".to_owned(),
                })?;
                let alpha: Option<f64> = deltas.pop();
                Ok(ADFResult { unit_root_exists, df_statistic, critical_value, se_gamma, gamma, deltas, beta: None, alpha, adf_type: adf_type.clone(), })
            },
            ADFType::ConstantAndTrend => {
                let mut deltas: Vec<f64> = params.to_vec();
                let gamma: f64 = deltas.pop().ok_or(DigiFiError::Other {
                    title: Self::error_title(),
                    details: "Parameters of the linear regression were empty.".to_owned(),
                })?;
                let alpha: Option<f64> = deltas.pop();
                let beta: Option<f64> = deltas.pop();
                Ok(ADFResult { unit_root_exists, df_statistic, critical_value, se_gamma, gamma, deltas, beta, alpha, adf_type: adf_type.clone(), })
            },
        }
    }
}

impl ErrorTitle for ADFResult {
    fn error_title() -> String {
        String::from("ADFResult Build")
    }
}

impl Display for ADFResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = LARGE_TEXT_BREAK.to_owned()
            + "Augmented Dickey-Fuller Test Result\n"
            + LARGE_TEXT_BREAK
            + &format!("Unit Root Exists: {}\n", self.unit_root_exists)
            + &format!("df-statistic: {}\n", self.df_statistic.unwrap_or(f64::NAN))
            + &format!("Critical Value: {}\n", self.critical_value.unwrap_or(f64::NAN))
            + &format!("Gamma Term: {}\n", self.gamma)
            + &format!("Standard Error of Gamma: {}\n", self.se_gamma.unwrap_or(f64::NAN))
            + "Autoregressive Terms (Delta): " + &self.deltas.iter().map(|v| v.to_string() ).collect::<Vec<String>>().join(", ") + "\n"
            + &format!("Trend Term (Beta): {}\n", self.beta.unwrap_or(f64::NAN))
            + &format!("Constant Term (Alpha): {}\n", self.alpha.unwrap_or(f64::NAN))
            + &format!("ADF Test Type: {}\n", self.adf_type)
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}


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
/// - Full Model: \\Delta y_{t} = \\alpha + \\beta t \\gamma y_{t-1} + \\delta_{1}\\Delta y_{t-1} + ... + \\delta_{p-1}\\Delta y_{t-p-1} + \\epsilon_{t}
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
/// use digifi::statistics::stat_tests::{ConfidenceLevel, ADFType, ADFResult, adf};
///
/// let x: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 6.0, 5.0, 6.0, 7.0, 8.0]);
/// let result: ADFResult = adf(&x, Some(1), &ADFType::Simple, Some(ConfidenceLevel::One)).unwrap();
/// 
/// assert_eq!(result.unit_root_exists, false);
/// ```
pub fn adf(x: &Array1<f64>, lag: Option<usize>, adf_type: &ADFType, cl: Option<ConfidenceLevel>) -> Result<ADFResult, DigiFiError> {
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
    let lr_result_len: usize = lr_result.len();
    // Right side of the linear regression model
    let mut fc: FeatureCollection = FeatureCollection::new();
    // Delta terms
    for j in 0..lag {
        let upper_index: usize = diff.len() - 1 - j;
        let feature_name: String = format!("Delta Term (Upper Index: {})", upper_index);
        fc.add_feature(diff.slice(s![(upper_index - lr_result_len)..upper_index]).iter().map(|v_| *v_ ), &feature_name)?;
    }
    // Constant trend term (Beta)
    if let ADFType::ConstantAndTrend = adf_type {
        let (range_start, range_end) = ((x_len - lr_result_len + 1) as i16, x_len as i16);
        fc.add_feature((range_start..=range_end).map(|v| v as f64 ), &format!("Constant Trend Term (Beta)"))?;    }
    // Constant term (Alpha)
    if let ADFType::Constant | ADFType::ConstantAndTrend = adf_type {
        fc.add_constant = true;
    }
    // Gamma term
    let gamma_label: String = String::from("Gamma Term");
    fc.add_feature(x.slice(s![lag..(x_len - 1)]).iter().map(|v_| *v_ ), &gamma_label)?;
    // Linear regression
    let mut lra_settings: LinearRegressionSettings = LinearRegressionSettings::disable_all();
    lra_settings.enable_se = true;
    let lra: LinearRegressionAnalysis = LinearRegressionAnalysis::new(lra_settings);
    let lra_result: LinearRegressionResult = lra.run(&mut fc, &lr_result)?;
    let gamma: &LinearRegressionFeatureResult = &lra_result.coefficients[fc.get_feature_index(&gamma_label)?];
    // If gamma is positive no unit root exists as the process is not stationary
    if 0.0 < gamma.coefficient {
        return ADFResult::build(&lra_result.all_coefficients, false, None, None, None, adf_type);
    }
    let se_gamma: f64 = gamma.standard_error.ok_or(DigiFiError::NotFound { title: error_title, data: "standard error of gamma".to_owned(), })?;
    // Calculate Dickey-Fuller statistic (i.e., gamma / SE(gamma))
    let df_statistic: f64 = gamma.coefficient / se_gamma;
    let cl: ConfidenceLevel = match cl { Some(cl) => cl, None => ConfidenceLevel::default(), };
    let critical_value: f64 = adf_type.get_critical_value(x_len, &cl);
    let unit_root_exists: bool = if df_statistic < critical_value { false } else { true };
    ADFResult::build(&lra_result.all_coefficients, unit_root_exists, Some(df_statistic), Some(critical_value), Some(se_gamma), adf_type)
}


#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Result of the cointegration test.
pub struct CointegrationResult {
    pub cointegrated: bool,
    pub adf_results: Option<Vec<ADFResult>>,
}

impl Display for CointegrationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = match &self.adf_results {
                Some(tests) => {
                    tests.iter().map(|t| t.to_string() ).collect::<Vec<String>>().join("\n\n")
                },
                None => "".to_owned(),
            } + "\n\n"
            + LARGE_TEXT_BREAK
            + "Cointegration Test Result\n"
            + LARGE_TEXT_BREAK
            + &format!("Cointegrated: {}\n", self.cointegrated)
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}


/// Cointegration is a statistical property describing a long-term, stable relationship between two or more time series variables,
/// even if those variables themselves are individually non-stationary (i.e., they have trends). This means that despite their individual fluctuations,
/// the variables move together in the long run, anchored by an underlying equilibrium relationship.
/// 
/// # Input
/// - `x`: Time series that is first-order integrated (i.e., I(0))
/// - `y`: Time series that is first-order integrated (i.e., I(0))
/// - `cl`: Confidence level in the Dickey-Fuller table that will be used to compare the Dickey-Fuller statistic against for the hypothesis test
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
/// use digifi::statistics::{ConfidenceLevel, CointegrationResult, cointegration};
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
///     let cointegration_result: CointegrationResult = cointegration(&jpm, &gs, Some(ConfidenceLevel::One)).unwrap();
///     
///     assert_eq!(cointegration_result.cointegrated, true);
/// }
/// ```
pub fn cointegration(x: &Array1<f64>, y: &Array1<f64>, cl: Option<ConfidenceLevel>) -> Result<CointegrationResult, DigiFiError> {
    compare_len(&x.iter(), &y.iter(), "x", "y")?;
    let data_matrix: Array2<f64> = Array2::from_shape_vec((x.len(), 1), x.to_vec())?;
    let params: Array1<f64> = linear_regression(&data_matrix, y)?;
    let prediction: Array1<f64> = data_matrix.dot(&params);
    let u_t: Array1<f64> = y - &prediction;
    // Run ADF test with different configurations (i.e., different types of augmented Dickey-Fuller test)
    let adf_variations: Vec<ADFType> = vec![ADFType::Simple, ADFType::Constant, ADFType::ConstantAndTrend];
    let mut cointegrated: bool = false;
    let mut adf_results: Vec<ADFResult> = Vec::<ADFResult>::new();
    for adf_type in &adf_variations {
        let adf_result: ADFResult = adf(&u_t, None, adf_type, cl)?;
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
        use crate::statistics::stat_tests::{ConfidenceLevel, ADFType, ADFResult, adf};
        let x: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 6.0, 5.0, 6.0, 7.0, 8.0]);
        let result: ADFResult = adf(&x, Some(1), &ADFType::Simple, Some(ConfidenceLevel::One)).unwrap();
        assert_eq!(result.unit_root_exists, false);
    }

    #[cfg(all(test, feature = "sample_data"))]
    #[test]
    fn unit_test_cointegration() -> () {
        use crate::utilities::sample_data::SampleData;
        use crate::statistics::stat_tests::{ConfidenceLevel, CointegrationResult, cointegration};
        let sample: SampleData = SampleData::Portfolio; 
        let (_, mut sample_data) = sample.load_sample_data();
        let jpm: Array1<f64> = sample_data.remove("JPM").unwrap();
        let gs: Array1<f64> = sample_data.remove("GS").unwrap();
        let cointegration_result: CointegrationResult = cointegration(&jpm, &gs, Some(ConfidenceLevel::One)).unwrap();
        assert_eq!(cointegration_result.cointegrated, true);
    }
}