use std::cmp;
use ndarray::{Array1, Array2, arr1, s};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;
use crate::utilities::{compare_array_len, maths_utils::differencing};
use crate::statistics::{
    ProbabilityDistribution, linear_regression, se_lr_coefficient,
    continuous_distributions::StudentsTDistribution,
};


#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
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
/// let result: ADFResult = adf(&x, Some(1), &ADFType::Simple, &ConfidenceLevel::One).unwrap();
/// 
/// assert_eq!(result.unit_root_exists, false);
/// ```
pub fn adf(x: &Array1<f64>, lag: Option<usize>, adf_type: &ADFType, cl: &ConfidenceLevel) -> Result<ADFResult, DigiFiError> {
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
    // If gamma is positive no unit root exists as the process is not stationary
    let gamma: f64 = params[params.len()-1];
    if 0.0 < gamma {
        return ADFResult::new(&params, false, None, None, None, adf_type);
    }
    // Compute standard error for estimated gamma
    let prediction: Array1<f64> = data_matrix.t().dot(&params);
    let se_gamma: f64 = se_lr_coefficient(&lr_result, &prediction, &data_matrix.slice(s![data_matrix.dim().0-1, ..]).to_owned(), params.len())?;
    // Calculate Dickey-Fuller statistic (i.e., gamma / SE(gamma))
    let df_statistic: f64 = gamma / se_gamma;
    let critical_value: f64 = adf_type.get_critical_value(x_len, cl);
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
///     let cointegration_result: CointegrationResult = cointegration(&jpm, &gs, &ConfidenceLevel::One).unwrap();
///     
///     assert_eq!(cointegration_result.cointegrated, true);
/// }
/// ```
pub fn cointegration(x: &Array1<f64>, y: &Array1<f64>, cl: &ConfidenceLevel) -> Result<CointegrationResult, DigiFiError> {
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


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Result of the t-test.
pub struct TTestResult {
    pub reject_h0: bool,
    pub p_value: f64,
    pub p_cl: f64,
}


/// # Description
/// Two sample t-test. It is used to test whether two samples have an equal mean. The null hypothesis is that the populations of both samp-les have equal mean.
/// 
/// # Input
/// - `sample_1`: Sample from the first population
/// - `sample_2`: Sample from the second population
/// - `cl`: Confidence level that should be used for the hypothesis test (Note: if `None` is provided, then defaults to the `ConfidenceLevel` deafult value)
/// 
/// # Output
/// - t-test result with additional information about the test
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Student's_t-test#Slope_of_a_regression_line>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::statistics::{ConfidenceLevel, TTestResult, t_test_two_sample};
///
/// #[cfg(feature = "sample_data")]
/// fn test_capm() -> () {
///     use digifi::utilities::SampleData;
///     
///     let sample: SampleData = SampleData::Portfolio; 
///     let (_, mut sample_data) = sample.load_sample_data();
///     let jpm: Array1<f64> = sample_data.remove("JPM").unwrap();
///     let gs: Array1<f64> = sample_data.remove("GS").unwrap();
/// 
///     // Test whether jpm and gs have equal mean
///     let test_result: TTestResult = t_test_two_sample(&jpm, &gs, Some(ConfidenceLevel::Five)).unwrap();
///     assert_eq!(test_result.reject_h0, true);
/// }
/// ```
pub fn t_test_two_sample(sample_1: &Array1<f64>, sample_2: &Array1<f64>, cl: Option<ConfidenceLevel>) -> Result<TTestResult, DigiFiError> {
    let error_title: String = String::from("T-test (Two Sample)");
    // Compute t score
    let dof: f64 = sample_1.len() as f64;
    let mean_1: f64 = sample_1.mean().ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "series_1".to_owned(), })?;
    let mean_2: f64 = sample_2.mean().ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "series_2".to_owned(), })?;
    let pooled_std: f64 = ((sample_1.var(1.0) + sample_2.var(1.0)) / 2.0).sqrt();
    let t_score: f64 = (mean_1 - mean_2) / (pooled_std * (2.0/dof).sqrt());
    // Obtain confidence interval value
    let dist: StudentsTDistribution = StudentsTDistribution::new(dof)?;
    let p_value: f64 = 1.0 - dist.cdf(&arr1(&[t_score]))?[0];
    let p_cl: f64 = match cl { Some(v) => 1.0 - v.get_p(), None => 1.0 - ConfidenceLevel::default().get_p() };
    let reject_h0: bool = if p_cl < p_value { true } else { false };
    Ok(TTestResult { reject_h0, p_value, p_cl })
}


/// # Description
/// T-test for the coefficient of a regression model. It is used to test whether an assumption about the value of the coefficient is
/// supported by the empirical data.
/// 
/// # Input
/// - `beta`: Coefficient estimated by the linear regression
/// - `beta_0`: Null Hypothesis for the coefficient beta's value (Note: `None` is evaluated to `0.0`)
/// - `y`: Observed response values
/// - `y_prediction`: Values predicted by the linear regression
/// - `x`: Feature in linear regression that corresponds to the coefficient beta
/// - `ddof`: Delta degrees of freedom of the linear regression model (i.e., degrees of freedom = N - ddof)
/// - `cl`: Confidence level that should be used for the hypothesis test (Note: if `None` is provided, then defaults to the `ConfidenceLevel` deafult value)
/// 
/// # Output
/// - t-test result with additional information about the test
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Student's_t-test#Slope_of_a_regression_line>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::corporate_finance::{CAPMParams, CAPMSolutionType, CAPMType, CAPM};
/// use digifi::statistics::{ConfidenceLevel, TTestResult, t_test_lr};
///
/// #[cfg(feature = "sample_data")]
/// fn test_capm() -> () {
///     use digifi::utilities::SampleData;
///     
///     // CAPM model
///     let sample: SampleData = SampleData::CAPM; 
///     let (_, mut sample_data) = sample.load_sample_data();
///     let capm_type: CAPMType = CAPMType::FiveFactorFamaFrench {
///         smb: sample_data.remove("SMB").unwrap(), hml: sample_data.remove("HML").unwrap(),
///         rmw: sample_data.remove("RMW").unwrap(), cma: sample_data.remove("CMA").unwrap(),
///     };
///     let market_return: Array1<f64> = sample_data.remove("Market").unwrap();
///     let rf: Array1<f64> = sample_data.remove("RF").unwrap();
///     let solution_type: CAPMSolutionType = CAPMSolutionType::LinearRegression;
///     let capm: CAPM = CAPM::new(market_return.clone(), rf.clone(), capm_type, solution_type).unwrap();
///     let y: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
///     let params: CAPMParams = capm.get_parameters(&y).unwrap();
/// 
///     // t-test
///     let x: Array1<f64> = market_return - rf;
///     let y_prediction: Array1<f64> = capm.predict_asset_return(
///         params.alpha.unwrap(), params.beta, params.beta_s.unwrap_or(0.0), params.beta_h.unwrap_or(0.0),
///         params.beta_r.unwrap_or(0.0), params.beta_c.unwrap_or(0.0)
///     ).unwrap();
///     let ddof: usize = y.len() - 6;
/// 
///     // Test for parameter beta with hull hypothesis that beta is 0.0
///     let test_result: TTestResult = t_test_lr(params.beta, None, &y, &y_prediction, &x, ddof, Some(ConfidenceLevel::Five)).unwrap();
///     assert_eq!(test_result.reject_h0, true);
/// 
///     // Test for parameter beta with hull hypothesis that beta is 1.0
///     let test_result: TTestResult = t_test_lr(params.beta, Some(1.0), &y, &y_prediction, &x, ddof, Some(ConfidenceLevel::Five)).unwrap();
///     assert_eq!(test_result.reject_h0, false);
/// }
/// ```
pub fn t_test_lr(beta: f64, beta_0: Option<f64>, y: &Array1<f64>, y_prediction: &Array1<f64>, x: &Array1<f64>, ddof: usize, cl: Option<ConfidenceLevel>) -> Result<TTestResult, DigiFiError> {
    // Compute t score
    let beta_0: f64 = match beta_0 { Some(v) => v, None => 0.0, };
    let se_beta: f64 = se_lr_coefficient(y, y_prediction, x, ddof)?;
    let t_score: f64 = (beta - beta_0) / se_beta;
    // Obtain confidence interval value
    let dist: StudentsTDistribution = StudentsTDistribution::new(ddof as f64)?;
    let p_value: f64 = 1.0 - dist.cdf(&arr1(&[t_score]))?[0];
    let p_cl: f64 = match cl { Some(v) => v.get_p(), None => ConfidenceLevel::default().get_p() };
    let reject_h0: bool = if p_value < p_cl { true } else { false };
    Ok(TTestResult { reject_h0, p_value, p_cl })
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;

    #[test]
    fn unit_test_adf() -> () {
        use crate::statistics::stat_tests::{ConfidenceLevel, ADFType, ADFResult, adf};
        let x: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 6.0, 5.0, 6.0, 7.0, 8.0]);
        let result: ADFResult = adf(&x, Some(1), &ADFType::Simple, &ConfidenceLevel::One).unwrap();
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
        let cointegration_result: CointegrationResult = cointegration(&jpm, &gs, &ConfidenceLevel::One).unwrap();
        assert_eq!(cointegration_result.cointegrated, true);
    }

    #[test]
    fn unit_test_t_test_two_sample() -> () {
        use crate::utilities::sample_data::SampleData;
        use crate::statistics::stat_tests::{ConfidenceLevel, TTestResult, t_test_two_sample};
        let sample: SampleData = SampleData::Portfolio; 
        let (_, mut sample_data) = sample.load_sample_data();
        let jpm: Array1<f64> = sample_data.remove("JPM").unwrap();
        let gs: Array1<f64> = sample_data.remove("GS").unwrap();
        // Test whether jpm and gs have equal mean
        let test_result: TTestResult = t_test_two_sample(&jpm, &gs, Some(ConfidenceLevel::Five)).unwrap();
        assert_eq!(test_result.reject_h0, true);
    }

    #[cfg(all(test, feature = "sample_data"))]
    #[test]
    fn unit_test_t_test_lr() -> () {
        use crate::utilities::sample_data::SampleData;
        use crate::corporate_finance::capm::{CAPMParams, CAPMSolutionType, CAPMType, CAPM};
        use crate::statistics::stat_tests::{ConfidenceLevel, TTestResult, t_test_lr};
        // CAPM model
        let sample: SampleData = SampleData::CAPM; 
        let (_, mut sample_data) = sample.load_sample_data();
        let capm_type: CAPMType = CAPMType::FiveFactorFamaFrench {
            smb: sample_data.remove("SMB").unwrap(), hml: sample_data.remove("HML").unwrap(),
            rmw: sample_data.remove("RMW").unwrap(), cma: sample_data.remove("CMA").unwrap(),
        };
        let market_return: Array1<f64> = sample_data.remove("Market").unwrap();
        let rf: Array1<f64> = sample_data.remove("RF").unwrap();
        let solution_type: CAPMSolutionType = CAPMSolutionType::LinearRegression;
        let capm: CAPM = CAPM::new(market_return.clone(), rf.clone(), capm_type, solution_type).unwrap();
        let y: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
        let params: CAPMParams = capm.get_parameters(&y).unwrap();
        // t-test
        let x: Array1<f64> = market_return - rf;
        let y_prediction: Array1<f64> = capm.predict_asset_return(
            params.alpha.unwrap(), params.beta, params.beta_s.unwrap_or(0.0), params.beta_h.unwrap_or(0.0),
            params.beta_r.unwrap_or(0.0), params.beta_c.unwrap_or(0.0)
        ).unwrap();
        let ddof: usize = y.len() - 6;
        // Test for parameter beta with hull hypothesis that beta is 0.0
        let test_result: TTestResult = t_test_lr(params.beta, None, &y, &y_prediction, &x, ddof, Some(ConfidenceLevel::Five)).unwrap();
        assert_eq!(test_result.reject_h0, true);
        // Test for parameter beta with hull hypothesis that beta is 1.0
        let test_result: TTestResult = t_test_lr(params.beta, Some(1.0), &y, &y_prediction, &x, ddof, Some(ConfidenceLevel::Five)).unwrap();
        assert_eq!(test_result.reject_h0, false);
    }
}