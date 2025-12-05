use std::fmt::Display;
use ndarray::{Array1, s};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::{FeatureCollection, LARGE_TEXT_BREAK, compare_len};
use crate::statistics::{
    ProbabilityDistribution, akaike_information_criterion_log,
    continuous_distributions::FDistribution,
    stat_tests::{ConfidenceLevel, f_test::nested_f_statistic},
    linear_regression_analysis::{LinearRegressionResult, LinearRegressionSettings, LinearRegressionAnalysis},
    arima::{ARResult, AROrderMethod, ARSettings, AR},
};


#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Type of Granger causality test.
pub enum GrangerCausalityTestType {
    #[default]
    /// Simple Test (`X` and `Y` are considered to be the order of `max_lag` and no t-test filtering of autoregressive terms of `X` is applied)
    Simple,
    /// Full Test (Optimal maximum lags for autoregressive terms of `X` and `Y` are found and t-test filtering of autoregressive terms of `X` is applied)
    Full,
}

impl Display for GrangerCausalityTestType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrangerCausalityTestType::Simple => write!(f, "Simple Test"),
            GrangerCausalityTestType::Full => write!(f, "Full Test"),
        }
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GrangerCausalityRejectReason {
    InfiniteAIC,
    TTestFilteredX,
    FTestRejected,
}

impl Display for GrangerCausalityRejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrangerCausalityRejectReason::InfiniteAIC => write!(f, "Could not select a model with autoregressive terms of X as tall models have infinite Akaike information criterion."),
            GrangerCausalityRejectReason::TTestFilteredX => write!(f, "All autoregressive terms of X have been filtered out as insignificant according to the t-test."),
            GrangerCausalityRejectReason::FTestRejected => write!(f, "Autoregressive terms of X do not add explanatory power according to the F-test."),
        }
    }
}


#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Result of the Granger causality test.
pub struct GrangerCausalityResult {
    /// Type of Granger causality test performed
    pub test_type: GrangerCausalityTestType,
    /// Whether to reject null hypothesis
    pub reject_h0: bool,
    /// Maximum lag that is applied to both `X` and `Y` (i.e., the maximum order of autoregressive terms of `X` and `Y`).
    /// Applicable only for Simple Granger causality test.
    pub max_lag: Option<usize>,
    /// Reason for failing to reject the null hypothesis.
    /// Applicable only for Full Granger causality test.
    pub failed_reject_h0_reason: Option<GrangerCausalityRejectReason>,
    /// Maximum lag used in computing partial autocorrelations.
    /// Applicable only for Full Granger causality test.
    pub pac_max_lag: Option<usize>,
    /// Confidence level for the partial autocorrelations of `Y`.
    /// Applicable only for Full Granger causality test.
    pub pac_cl: Option<f64>,
    /// Critical value for partial autocorrelations of `Y`, obtained from `pac_ci`.
    /// Applicable only for Full Granger causality test.
    pub pac_crit: Option<f64>,
    /// Actual order (i.e., maximum lag) of the autoregressive terms of `Y`.
    /// Applicable only for Full Granger causality test.
    pub y_order: Option<usize>,
    /// Minimized Akaike information criterion that produces the maximum lag of `X`.
    /// Applicable only for Full Granger causality test.
    pub min_aic: Option<f64>,
    /// Maximum lag of the autoregressive terms of `X`.
    /// Applicable only for Full Granger causality test.
    /// 
    /// Note: This may not correspond to the order of `X` in the final Granger causality dependency as some autoregressive terms of `X`
    /// may be filtered out by the t-test.
    /// Applicable only for Full Granger causality test.
    pub x_max_lag: Option<usize>,
    /// Residual sum of squares (RSS) of plain autoregressive model of `Y` (i.e., model independent of `X`)
    pub f_test_nested_rss: Option<f64>,
    /// Number of parameters in the plain autoregressive model of `Y` (i.e., model independent of `X`)
    pub f_test_nested_d: Option<usize>,
    /// Residual sum of squares (RSS) of augmented autoregressive model of `Y` (i.e., model dependent on autoregressive terms of `X`)
    pub f_test_parent_rss: Option<f64>,
    /// Number of parameters in the augmented autoregressive model of `Y` (i.e., model dependent on autoregressive terms of `X`)
    pub f_test_parent_d: Option<usize>,
    /// F-statistic of the F-test between plain and augmented autoregressive models
    pub f_test_f_statistic: Option<f64>,
    /// Numerator degrees of freedom used in the F-test (i.e., `parent_d - nested_d`)
    pub f_test_dof_1: Option<usize>,
    /// Denominator degrees of freedom used in the F-test (i.e., `n - parent_d`, where `n` is the total number of data points inside the feature collection)
    pub f_test_dof_2: Option<usize>,
    /// p-value of the F-test
    pub f_test_p_value: Option<f64>,
    /// Confidence level for F-test (Quoted as probability to compare `p_value` against)
    pub f_test_p_cl: Option<f64>,
    /// Whether to reject the null hypothesis that any of the lagged terms `X` add explanatory power to the model of `Y`
    pub f_test_reject_h0: Option<bool>,
    /// Result of the nested model (i.e., AR model of `Y`)
    pub nested_model_result: Option<ARResult>,
    /// Result of the parent model (i.e., linear regression of `Y` in terms of autoregressive terms of `Y` and `X`)
    pub parent_model_result: Option<LinearRegressionResult>,
}

impl Display for GrangerCausalityResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let test_based_parameters: String = match self.test_type {
            GrangerCausalityTestType::Simple => {
                format!("Maximum Lag (i.e., maximum order for both `X` and `Y`): {}\n", match &self.max_lag { Some(v) => v.to_string(), None => "".to_owned(), })
            },
            GrangerCausalityTestType::Full => {
                format!("Maximum Lag (Partial Autocorrelation): {}\n", match &self.pac_max_lag { Some(v) => v.to_string(), None => "".to_owned(), })
                    + &format!("Confidence Level (Partial Autocorrelation): {}\n", self.pac_cl.unwrap_or(f64::NAN))
                    + &format!("Critical Value (Partial Autocorrelation): {}\n", self.pac_crit.unwrap_or(f64::NAN))
                    + &format!("Order of Y (i.e., order of AR model of Y): {}\n", match &self.y_order { Some(v) => v.to_string(), None => "".to_owned(), })
                    + LARGE_TEXT_BREAK
                    + &format!("Minimized AIC for Selecting the Maximum Lag of X: {}\n", self.min_aic.unwrap_or(f64::NAN))
                    + &format!("Maximum Lag of X (i.e., upper limit for the order of X): {}\n", match &self.x_max_lag { Some(v) => v.to_string(), None => "".to_owned(), })
            },
        };
        let result: String = LARGE_TEXT_BREAK.to_owned()
            + "Granger Causality Test Result\n"
            + LARGE_TEXT_BREAK
            + &format!("Test Type: {}\n", self.test_type)
            + LARGE_TEXT_BREAK
            + &test_based_parameters
            + LARGE_TEXT_BREAK
            + &format!("Nested RSS (i.e., RSS of AR model of Y): {}\n", self.f_test_nested_rss.unwrap_or(f64::NAN))
            + &format!("Nested D (i.e., number of parameters in AR model of Y): {}\n", match &self.f_test_nested_d { Some(v) => v.to_string(), None => "".to_owned(), })
            + &format!("Parent RSS (i.e., RSS of full Granger linear regression model): {}\n", self.f_test_parent_rss.unwrap_or(f64::NAN))
            + &format!("Parent D (i.e., number of parameters in full Granger linear regression model): {}\n", match &self.f_test_parent_d { Some(v) => v.to_string(), None => "".to_owned(), })
            + &format!("Numerator Degrees of Freedom (F-test): {}\n", match &self.f_test_dof_1 { Some(v) => v.to_string(), None => "".to_owned(), })
            + &format!("Denominator Degrees of Freedom (F-test): {}\n", match &self.f_test_dof_2 { Some(v) => v.to_string(), None => "".to_owned(), })
            + &format!("Reject Null Hypothesis (F-test): {}\n", self.f_test_reject_h0.unwrap_or_default())
            + &format!("F-statistic (F-test): {}\n", self.f_test_f_statistic.unwrap_or(f64::NAN))
            + &format!("p-value (F-test): {}\n", self.f_test_p_value.unwrap_or(f64::NAN))
            + &format!("Confidence Level (F-test): {}\n", self.f_test_p_cl.unwrap_or(f64::NAN))
            + LARGE_TEXT_BREAK
            + &format!("Reason for Failing to Reject Null Hypothesis: {}\n", match &self.failed_reject_h0_reason { Some(v) => v.to_string(), None => "".to_owned(), })
            + &format!("Reject Null Hypothesis: {}\n", self.reject_h0)
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}


#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Settings of the Granger causality test.
pub struct GrangerCausalitySettings {
    /// Maximum lag used in computing partial autocorrelations of `Y`
    pub pac_max_lag: Option<usize>,
    /// Confidence interval for partial autocorrelation. This interval is used to find the order of the autoregressive model of `Y`
    /// such that the order is the lag after which the partial autocorrelations are all within the confidence interval.
    /// 
    /// Note: This confidence interval is used to compute the symmetrical critical values (i.e., ± critical values) which act as the cut-off points
    /// for the order of the autoregressive model of `Y`.
    pub pac_ci: Option<f64>,
    /// Maximum lag of `X` used in finding the optimal linear regression model of `Y` in terms of autoregressive terms of `Y` and `X`.
    /// 
    /// Note: If no maximum lag provided, the order of AR model of `Y` will be used as the maximum lag for autoregressive terms of `X`.
    pub x_max_lag:  Option<usize>,
    /// Confidence level used to filter out the autoregressive terms of `X`
    pub t_test_cl: Option<ConfidenceLevel>,
    /// Confidence level that should be used for the F-test (Note: if `None` is provided, then defaults to the `ConfidenceLevel` deafult value)
    pub f_test_cl: Option<ConfidenceLevel>,
    /// Whether to return the result of the nested model (i.e., AR model of `Y`)
    pub return_nested_result: bool,
    /// Whether to return the result of the parent model (i.e., linear regression of `Y` in terms of autoregressive terms of `Y` and `X`)
    pub return_parent_result: bool,
}

impl Display for GrangerCausalitySettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = LARGE_TEXT_BREAK.to_owned()
            + "Linear Regression Settings\n"
            + LARGE_TEXT_BREAK
            + &format!("Maximum Lag (Partial Autocorrelation): {}\n", match &self.pac_max_lag { Some(v) => v.to_string(), None => "".to_owned(), })
            + &format!("Confidence Interval (Partial Autocorrelation): {}\n", match &self.pac_ci { Some(v) => v.to_string(), None => "".to_owned(), })
            + &format!("Maximum Lag of X: {}\n", match &self.x_max_lag { Some(v) => v.to_string(), None => "".to_owned(), })
            + &format!("t-Test Condifence Level: {}\n", self.t_test_cl.unwrap_or_default())
            + &format!("F-test Confidence Level: {}\n", self.f_test_cl.unwrap_or_default())
            + &format!("Return Nested Result (i.e., AR model of Y): {}\n", self.return_nested_result)
            + &format!("Return Parent Result (i.e., Linear regression model of Y in terms of autoregressive terms of Y and X): {}\n", self.return_parent_result)
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}


#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Granger causality test and its utility functions.
struct GrangerCausalityTest {
    /// Granger causality test settings
    settings: GrangerCausalitySettings,
}

impl GrangerCausalityTest {

    /// Constructs a new instance of `GrangerCausalityTest`.
    /// 
    /// # Input
    /// - `settings`: Settings of the Granger causality test
    fn new(settings: GrangerCausalitySettings) -> Self {
        GrangerCausalityTest { settings, }
    }

    fn validate(&self, y: &Array1<f64>, x: &Array1<f64>) -> Result<(), DigiFiError> {
        compare_len(&y.iter(), &x.iter(), "Y", "X")?;
        let n_div_two: usize = (y.len() as u32).div_euclid(2) as usize;
        match self.settings.x_max_lag {
            Some(x_max_lag) if n_div_two < x_max_lag => {
                return Err(DigiFiError::ValidationError {
                    title: Self::error_title(),
                    details: "Maximum lag `x_max_lag` must be smaller than `n // 2`.".to_owned(),
                });
            },
            _ => (),
        }
        Ok(())
    }

    /// Performs F-test over the remaining lagged auteregressive terms left from `X` and comparea to plain `Y` autoregressive model.
    fn conjoined_f_test(
        &self, nested_lra_result: &LinearRegressionResult, parent_lra_result: &LinearRegressionResult, n_data_points: usize,
    ) -> Result<(f64, usize, f64, usize, f64, usize, usize, f64, f64, bool), DigiFiError> {
        let nested_rss: f64 = nested_lra_result.sse.unwrap_or(0.0);
        let nested_d: usize = nested_lra_result.coefficients.len() + 1;
        let parent_rss: f64 = parent_lra_result.sse.unwrap_or(0.0);
        let parent_d: usize = parent_lra_result.coefficients.len() + 1;
        let f_statistic: f64 = nested_f_statistic(nested_rss, parent_rss, n_data_points, nested_d, parent_d)?;
        let dof_1: usize = parent_d - nested_d;
        let dof_2: usize = n_data_points - parent_d;
        let p_value: f64 = 1.0 - FDistribution::build(dof_1, dof_2)?.cdf(f_statistic)?;
        let p_cl: f64 = match self.settings.f_test_cl { Some(v) => v.get_p(), None => ConfidenceLevel::default().get_p() };
        let reject_h0: bool = if p_value < p_cl { true } else { false };
        Ok((nested_rss, nested_d, parent_rss, parent_d, f_statistic, dof_1, dof_2, p_value, p_cl, reject_h0))
    }

    /// Produces a name for the autoregressive terms of `Y` that will be used as its label in the feature collection.
    fn y_name(lag: usize) -> String {
        format!("Y(t-{})", lag)
    }

    /// Produces a name for the autoregressive terms of `X` that will be used as its label in the feature collection.
    fn x_name(lag: usize) -> String {
        format!("X(t-{})", lag)
    }

    /// Default settings that will be reused internally for all linear regression analysis instances.
    fn lr_settings(&self, max_likelihood: bool, t_test: bool) -> LinearRegressionSettings {
        let mut settings: LinearRegressionSettings = LinearRegressionSettings::disable_all();
        settings.enable_sse = true;
        settings.enable_max_likelihood = max_likelihood;
        settings.enable_t_test = t_test;
        settings.t_test_cl = self.settings.t_test_cl;
        settings
    }

    /// Adjusts the length of Y to fit the autoregressive models (i.e., shrinks Y).
    fn adjusted_y(y: &Array1<f64>, start_index: usize) -> Array1<f64> {
        Array1::from_iter(y.slice(s![start_index..]).iter().map(|v| *v ))
    }

    /// Creates a feature collection of autoregressive terms of `Y`, where each feature is trunkated by the order of `max(y_order, x_max_lag)`.
    fn create_feature_collection(&self, y: &Array1<f64>, y_order: usize) -> Result<FeatureCollection, DigiFiError> {
        let y_len: usize = y.len();
        let mut fc: FeatureCollection = FeatureCollection::new();
        fc.add_constant = true;
        let fc_order: usize = match self.settings.x_max_lag {
            Some(v) => v.max(y_order),
            None => y_order,
        };
        for lag in (1..=y_order).into_iter() {
            fc.add_feature(y.slice(s![(fc_order - lag)..(y_len - lag)]).into_iter(), &Self::y_name(lag))?;
        }
        Ok(fc)
    }

    /// Trains the linear regression model for the provided feature collection and `Y`.
    fn lr(&self, fc: &FeatureCollection, y: &Array1<f64>, t_test: bool) -> Result<LinearRegressionResult, DigiFiError> {
        let lra: LinearRegressionAnalysis = LinearRegressionAnalysis::new(self.lr_settings(true, t_test));
        lra.run(fc, y)
    }

    /// Finds the order of the autoregressive terms of `X` that minimizes Akaike information criterion.
    fn optimize_aic_x_lr(&self, fc: &mut FeatureCollection, x: &Array1<f64>, y: &Array1<f64>, x_max_lag: usize, y_order: usize, fc_order: usize) -> Result<(usize, f64), DigiFiError> {
        let x_len: usize = y.len();
        let n_parameters: usize = y_order + 1;
        let mut min_aic: f64 = f64::INFINITY;
        let mut x_order: usize = 0;
        // Adjust the length of Y based on the current feature size
        let y_slice: Array1<f64> = Self::adjusted_y(y, fc_order);
        for lag in (1..=x_max_lag).into_iter() {
            let x_slice = x.slice(s![(fc_order - lag)..(x_len - lag)]).into_iter();
            fc.add_feature(x_slice, &Self::x_name(lag))?;
            let lr_result: LinearRegressionResult = self.lr(&fc, &y_slice, false)?;
            let aic: f64 = akaike_information_criterion_log(lr_result.max_log_likelihood.unwrap_or_default(), n_parameters + lag);
            if aic < min_aic {
                min_aic = aic;
                x_order = lag;
            }
        }
        Ok((x_order, min_aic))
    }

    /// Runs the Granger causality test.
    fn run(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<GrangerCausalityResult, DigiFiError> {
        self.validate(y, x)?;
        let mut result: GrangerCausalityResult = GrangerCausalityResult::default();
        result.test_type = GrangerCausalityTestType::Full;
        // Run autoregression model for `Y`
        let ar_settings: ARSettings = ARSettings {
            order_method: AROrderMethod::PAC { max_lag: self.settings.pac_max_lag, cl: self.settings.pac_ci },
            return_fc: false, add_constant: true, additional_settings: Some(self.lr_settings(false, false)),
        };
        let ar: AR = AR::build(ar_settings)?;
        let (ar_result, _) = ar.run(y)?;
        let y_order: usize = ar_result.order;
        result.pac_max_lag = ar_result.pac_max_lag;
        result.pac_cl = ar_result.pac_cl;
        result.pac_crit = ar_result.pac_crit;
        result.y_order = Some(y_order);
        // Create a feature collection of autoregressive terms of `Y`, where each feature is shrunk by the order of `max(y_order, x_max_lag)`
        let mut fc: FeatureCollection = self.create_feature_collection(&y, y_order)?;
        // Determine the maximum number of autoregressive terms for `X` (Akaike information criterion)
        let x_max_lag: usize = self.settings.x_max_lag.unwrap_or(y_order);
        let fc_order: usize = y_order.max(x_max_lag);
        let (x_order, min_aic) = self.optimize_aic_x_lr(&mut fc, x, y, x_max_lag, y_order, fc_order)?;
        result.min_aic = Some(min_aic);
        result.x_max_lag = Some(x_max_lag);
        if x_max_lag == 0 {
            // Failed to reject h0 as models with any autoregressive terms of `X` result in infinite AIC value
            result.failed_reject_h0_reason = Some(GrangerCausalityRejectReason::InfiniteAIC);
            if self.settings.return_nested_result { result.nested_model_result = Some(ar_result); }
            return Ok(result)
        }
        // Format the feature collection to contain only the terms from `X` that minimize the AIC
        let over_x_max_lag: usize = x_order + 1;
        if over_x_max_lag <= x_max_lag {
            for lag in (over_x_max_lag..=x_max_lag).into_iter() {
               fc.remove_feature(&Self::x_name(lag))?;
            }
        }
        // Filter out lagged values of `X` by t-test
        let mut filtered: bool = false;
        let mut contains_x: bool = false;
        let mut parent_lra_result: LinearRegressionResult = LinearRegressionResult::default();
        let y_slice: Array1<f64> = Self::adjusted_y(y, fc_order);
        // Run LRA iteratively while removing lagged `X` features that fall outside of the t-test confidence level
        while !filtered {
            filtered = true;
            contains_x = false;
            parent_lra_result = self.lr(&fc, &y_slice, true)?;
            for c in parent_lra_result.coefficients.iter() {
                let coefficient_name: String = c.coefficient_name.clone().ok_or(DigiFiError::Other {
                    title: Self::error_title(),
                    details: "Nameless feature exists in the feature collection.".to_owned(),
                })?;
                // Check if the coefficient is the autoregressive term of `X`
                if coefficient_name.contains("X(t-") {
                    contains_x = true;
                    let reject_h0: bool = c.t_test_reject_h0.unwrap_or_default();
                    if !reject_h0 {
                       fc.remove_feature(&coefficient_name)?;
                    }
                    // Only if all coefficients are significant (i.e., h0 rejected for all) the model is considered to be filtered
                    filtered = filtered && reject_h0;
                }
            }
            if filtered {
                break
            }
        }
        // Check at least some autoregressive terms of `X` are remaining, otherwise `reject_h0 = false`
        if !contains_x {
            result.failed_reject_h0_reason = Some(GrangerCausalityRejectReason::TTestFilteredX);
            if self.settings.return_nested_result { result.nested_model_result = Some(ar_result); }
            return Ok(result);
        }
        // Perform F-test over the remaining lagged auteregressive terms left from `X` and compare to plain `Y` autoregressive model
        let n_data_points: usize = fc.feature_size().unwrap_or_default();
        let (nested_rss, nested_d, parent_rss, parent_d, f_statistic, dof_1, dof_2, p_value, p_cl, f_test_reject_h0) = self.conjoined_f_test(&ar_result.model_result, &parent_lra_result, n_data_points)?;
        result.reject_h0 = f_test_reject_h0;
        if !f_test_reject_h0 { result.failed_reject_h0_reason = Some(GrangerCausalityRejectReason::FTestRejected); }
        result.f_test_nested_rss = Some(nested_rss);
        result.f_test_nested_d = Some(nested_d);
        result.f_test_parent_rss = Some(parent_rss);
        result.f_test_parent_d = Some(parent_d);
        result.f_test_f_statistic = Some(f_statistic);
        result.f_test_dof_1 = Some(dof_1);
        result.f_test_dof_2 = Some(dof_2);
        result.f_test_p_value = Some(p_value);
        result.f_test_p_cl = Some(p_cl);
        result.f_test_reject_h0 = Some(f_test_reject_h0);
        if self.settings.return_nested_result { result.nested_model_result = Some(ar_result); }
        if self.settings.return_parent_result { result.parent_model_result = Some(parent_lra_result); }
        Ok(result)
    }
}

impl ErrorTitle for GrangerCausalityTest {
    fn error_title() -> String {
        String::from("Granger Causality Test")
    }
}


/// The Granger causality test is a statistical hypothesis test for determining whether one time series is useful in forecasting another.
/// The null hypothesis is that the time series `X` does not Granger-cause time series `Y`.
/// 
/// # Input
/// - `x`: Independent variable
/// - `y`: Variable that is potentially being forecasted (i.e., Granger-caused) by `X`
/// - `settings`: Settings of the Granger causality test
/// 
/// # Output
/// - Granger causality test result with additional information about the test
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Granger_causality>
/// - Original Source: <https://doi.org/10.2307/1912791>
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::LinearRegressionResult;
/// use digifi::statistics::{GrangerCausalityRejectReason, GrangerCausalityResult, GrangerCausalitySettings, granger_causality_test};
/// 
/// #[cfg(feature = "sample_data")]
/// fn test_granger_causality_test() -> () {
///     use digifi::utilities::SampleData;
/// 
///     // Sample data
///     let sample: SampleData = SampleData::CAPM; 
///     let (_, mut sample_data) = sample.load_sample_data();
///     let market_premium: Array1<f64> = sample_data.remove("Market").unwrap() - sample_data.remove("RF").unwrap();
///     let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
///
///     // Granger causalityy test
///     let settings: GrangerCausalitySettings = GrangerCausalitySettings {
///         pac_max_lag: Some(20), pac_ci: Some(0.90), x_max_lag: None,
///         t_test_cl: None, f_test_cl: None,
///         return_nested_result: true, return_parent_result: true,
///     };
///     let gcr: GrangerCausalityResult = granger_causality_test(&market_premium, &risk_premium, settings).unwrap();
///
///     // Nested model tests (i.e., tests of AR model of Y)
///     assert_eq!(gcr.y_order.unwrap(), 16);
///     let nested_result: &LinearRegressionResult = &gcr.nested_model_result.as_ref().unwrap().model_result;
///     assert!((nested_result.intercept.unwrap() - 0.017544077338390595).abs() < TEST_ACCURACY);
///     let coefs: Vec<f64> = vec![
///         0.07144487, 0.07683892, 0.11094491, -0.11023364, -0.14995811, -0.05011439, 0.18793717,
///         0.26099817, -0.33252268, -0.14385452, 0.19426531, 0.12370539, -0.00569986, -0.04653284, -0.21241471, 0.31252425,
///     ];
///     let _ = nested_result.coefficients.iter().zip(coefs.iter()).map(|(ar_c, c)|  {
///         assert!((ar_c.coefficient - c).abs() < TEST_ACCURACY);
///     } );
///
///     // Parent model tests (i.e., tests of the complete Granger linear model)
///     assert_eq!(gcr.reject_h0, false);
///     assert!(match gcr.failed_reject_h0_reason.unwrap() { GrangerCausalityRejectReason::TTestFilteredX => true, _ => false,});
/// }
/// ```
pub fn granger_causality_test(x: &Array1<f64>, y: &Array1<f64>, settings: GrangerCausalitySettings) -> Result<GrangerCausalityResult, DigiFiError> {
    let gct: GrangerCausalityTest = GrangerCausalityTest::new(settings);
    gct.run(x, y)
}


/// The Granger causality test is a statistical hypothesis test for determining whether one time series is useful in forecasting another.
/// The null hypothesis is that the time series `X` does not Granger-cause time series `Y`.
/// 
/// Note: This implementation of Granger causality test assumes that time series `X` and `Y` both have the same maximum lag, and there is no t-test
/// filtering that is done to remove the non-significant autoregressive terms of `X` from the parent Granger linear regression model.
/// 
/// # Input
/// - `x`: Independent variable
/// - `y`: Variable that is potentially being forecasted (i.e., Granger-caused) by `X`
/// - `max_lag`: Maximum lag that is applied to both `X` and `Y` (i.e., the maximum order of autoregressive terms of `X` and `Y`)
/// - `f_test_cl`: Confidence level that should be used for the F-test (Note: if `None` is provided, then defaults to the `ConfidenceLevel` deafult value)
/// 
/// # Output
/// - Granger causality test result with additional information about the test
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Granger_causality>
/// - Original Source: <https://doi.org/10.2307/1912791>
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{GrangerCausalityRejectReason, GrangerCausalityResult, simple_granger_causality_test};
/// 
/// #[cfg(feature = "sample_data")]
/// fn test_simple_granger_causality_test() -> () {
///     use digifi::utilities::SampleData;
/// 
///     // Sample data
///     let sample: SampleData = SampleData::CAPM; 
///     let (_, mut sample_data) = sample.load_sample_data();
///     let market_premium: Array1<f64> = sample_data.remove("Market").unwrap() - sample_data.remove("RF").unwrap();
///     let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
/// 
///     // Simple Granger causality test
///     let gcr: GrangerCausalityResult = simple_granger_causality_test(&market_premium, &risk_premium, 16, None).unwrap();
/// 
///     // The results were found using grangercausalitytests from statsmodels.tsa.stattools
///     // Nested model tests (i.e., tests of AR model of `Y`)
///     assert!((gcr.nested_model_result.as_ref().unwrap().model_result.intercept.unwrap() - 0.01754408).abs() < TEST_ACCURACY);
///     let nested_coefs: Vec<f64> = vec![
///         0.07144487, 0.07683892, 0.11094491, -0.11023364, -0.14995811, -0.05011439, 0.18793717, 0.26099817, -0.33252268, -0.14385452, 0.19426531,
///         0.12370539, -0.00569986, -0.04653284, -0.21241471,  0.31252425,
///     ];
///     let _ = gcr.nested_model_result.as_ref().unwrap().model_result.coefficients.iter().zip(nested_coefs.iter()).map(|(ar_c, c)|  {
///         assert!((ar_c.coefficient - c).abs() < TEST_ACCURACY);
///     } );
///
///     // Parent model tests (i.e., tests of the complete Granger linear model)
///     assert!((gcr.parent_model_result.as_ref().unwrap().intercept.unwrap() - 1.14274372e-02).abs() < TEST_ACCURACY);
///     let nested_coefs: Vec<f64> = vec![
///         4.19223242e-01, -2.34404524e-01,  2.72340900e-01,  1.03013448e+00, -1.81718789e-01, -4.27993121e-01,  8.40314944e-01,  6.19071691e-01,
///         -6.60135239e-01, -2.97185005e-01, -2.48296270e-03, -3.07798660e-01, 3.65214587e-01,  6.54557206e-01,  6.82388733e-01,  2.57055272e-01,
///         -1.16703714e+00,  3.84263021e-01, -6.80947028e-01, -1.98267527e+00, -8.58625697e-02,  1.61253913e-01, -9.51342081e-01,  2.87793511e-01,
///         5.41451441e-01,  3.76870986e-01,  5.03655333e-01,  4.78529632e-01, -1.41274516e-01, -7.91401835e-01, -2.74771397e+00, -8.14687698e-01,
///     ];
///     let _ = gcr.parent_model_result.as_ref().unwrap().coefficients.iter().zip(nested_coefs.iter()).map(|(ar_c, c)|  {
///         assert!((ar_c.coefficient - c).abs() < TEST_ACCURACY);
///     } );
///
///     // F-test tests
///     assert!((gcr.f_test_f_statistic.unwrap() - 0.4512).abs() < 10_000.0 * TEST_ACCURACY);
///     assert_eq!(gcr.f_test_dof_1.unwrap(), 16);
///     assert_eq!(gcr.f_test_dof_2.unwrap(), 7);
///     assert!((gcr.f_test_p_value.unwrap() - 0.9111).abs() < 10_000.0 * TEST_ACCURACY);
///     assert_eq!(gcr.reject_h0, false);
///     assert!(match gcr.failed_reject_h0_reason.unwrap() { GrangerCausalityRejectReason::FTestRejected => true, _ => false,});
/// }
/// ```
pub fn simple_granger_causality_test(x: &Array1<f64>, y: &Array1<f64>, max_lag: usize, f_test_cl: Option<ConfidenceLevel>) -> Result<GrangerCausalityResult, DigiFiError> {
    let error_title: String = String::from("Simple Granger Causality Test");
    let mut result: GrangerCausalityResult = GrangerCausalityResult::default();
    result.max_lag = Some(max_lag);
    // Parameters
    let y_len: usize = y.len();
    let mut settings: LinearRegressionSettings = LinearRegressionSettings::disable_all();
    settings.enable_sse = true;
    // Validation
    compare_len(&y.iter(), &x.iter(), "Y", "X")?;
    if max_lag < 1 {
        return Err(DigiFiError::ParameterConstraint { title: error_title, constraint: "The parameter `max_lag` must be bigger or equal to `1`.".to_owned(), });
    }
    let n_div_two: usize = (y_len as u32).div_euclid(2) as usize;
    if n_div_two < max_lag {
        return Err(DigiFiError::ValidationError { title: error_title, details: "Maximum lag `max_lag` must be smaller than `n // 2`.".to_owned(), });
    }
    // Aurogeressive model of `Y`
    let ar_settings: ARSettings = ARSettings {
        order_method: AROrderMethod::Manual { order: max_lag },
        return_fc: true, add_constant: true, additional_settings: Some(settings.clone()),
    };
    let ar: AR = AR::build(ar_settings)?;
    let (nested_model_result, fc) = ar.run(y)?;
    let mut fc: FeatureCollection = fc.ok_or(DigiFiError::NotFound { title: error_title, data: "Feature collection is not found.".to_owned(), })?;
    // Populate feature collection with autoregressive terms of `X`
    for lag in (1..=max_lag).into_iter() {
        fc.add_feature(x.slice(s![(max_lag - lag)..(y_len - lag)]).into_iter(), &format!("X(t-{})", lag))?;
    }
    // Linear regression with autoregressive terms of `Y` and `X`
    let y_slice: Array1<f64> = Array1::from_iter(y.slice(s![max_lag..]).into_iter().map(|v| *v ));
    let lr: LinearRegressionAnalysis = LinearRegressionAnalysis::new(settings);
    let parent_model_result: LinearRegressionResult = lr.run(&fc, &y_slice)?;
    // F-test
    let n_data_points: usize = fc.feature_size().unwrap_or_default();
    let nested_rss: f64 = nested_model_result.model_result.sse.unwrap_or(0.0);
    let nested_d: usize = nested_model_result.model_result.coefficients.len() + 1;
    let parent_rss: f64 = parent_model_result.sse.unwrap_or(0.0);
    let parent_d: usize = parent_model_result.coefficients.len() + 1;
    let f_statistic: f64 = nested_f_statistic(nested_rss, parent_rss, n_data_points, nested_d, parent_d)?;
    let dof_1: usize = parent_d - nested_d;
    let dof_2: usize = n_data_points - parent_d;
    let p_value: f64 = 1.0 - FDistribution::build(dof_1, dof_2)?.cdf(f_statistic)?;
    let p_cl: f64 = match f_test_cl { Some(v) => v.get_p(), None => ConfidenceLevel::default().get_p() };
    let f_test_reject_h0: bool = if p_value < p_cl { true } else { false };
    if !f_test_reject_h0 { result.failed_reject_h0_reason = Some(GrangerCausalityRejectReason::FTestRejected); }
    // Result
    result.reject_h0 = f_test_reject_h0;
    result.f_test_nested_rss = Some(nested_rss);
    result.f_test_nested_d = Some(nested_d);
    result.f_test_parent_rss = Some(parent_rss);
    result.f_test_parent_d = Some(parent_d);
    result.f_test_f_statistic = Some(f_statistic);
    result.f_test_dof_1 = Some(dof_1);
    result.f_test_dof_2 = Some(dof_2);
    result.f_test_p_value = Some(p_value);
    result.f_test_p_cl = Some(p_cl);
    result.f_test_reject_h0 = Some(f_test_reject_h0);
    result.nested_model_result = Some(nested_model_result);
    result.parent_model_result = Some(parent_model_result);
    Ok(result)
}


#[cfg(all(test, feature = "sample_data"))]
mod tests {
    use ndarray::Array1;
    use crate::utilities::{TEST_ACCURACY, sample_data::SampleData};
    use crate::statistics::stat_tests::granger_causality::{
        GrangerCausalityRejectReason, GrangerCausalityResult,
    };

    #[test]
    fn unit_test_granger_causality_test() -> () {
        use crate::statistics::linear_regression_analysis::LinearRegressionResult;
        use crate::statistics::stat_tests::granger_causality::{GrangerCausalitySettings, granger_causality_test};
        // Sample data
        let sample: SampleData = SampleData::CAPM; 
        let (_, mut sample_data) = sample.load_sample_data();
        let market_premium: Array1<f64> = sample_data.remove("Market").unwrap() - sample_data.remove("RF").unwrap();
        let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
        // Granger causalityy test
        let settings: GrangerCausalitySettings = GrangerCausalitySettings {
            pac_max_lag: Some(20), pac_ci: Some(0.90), x_max_lag: None,
            t_test_cl: None, f_test_cl: None,
            return_nested_result: true, return_parent_result: true,
        };
        let gcr: GrangerCausalityResult = granger_causality_test(&market_premium, &risk_premium, settings).unwrap();
        // Nested model tests (i.e., tests of AR model of Y)
        assert_eq!(gcr.y_order.unwrap(), 16);
        let nested_result: &LinearRegressionResult = &gcr.nested_model_result.as_ref().unwrap().model_result;
        assert!((nested_result.intercept.unwrap() - 0.017544077338390595).abs() < TEST_ACCURACY);
        let coefs: Vec<f64> = vec![
            0.07144487, 0.07683892, 0.11094491, -0.11023364, -0.14995811, -0.05011439, 0.18793717,
            0.26099817, -0.33252268, -0.14385452, 0.19426531, 0.12370539, -0.00569986, -0.04653284, -0.21241471, 0.31252425,
        ];
        let _ = nested_result.coefficients.iter().zip(coefs.iter()).map(|(ar_c, c)|  {
            assert!((ar_c.coefficient - c).abs() < TEST_ACCURACY);
        } );
        // Parent model tests (i.e., tests of the complete Granger linear model)
        assert_eq!(gcr.reject_h0, false);
        assert!(match gcr.failed_reject_h0_reason.unwrap() { GrangerCausalityRejectReason::TTestFilteredX => true, _ => false,});
    }

    #[test]
    fn unit_test_simple_granger_causality_test() -> () {
        use crate::statistics::stat_tests::granger_causality::simple_granger_causality_test;
        // Sample data
        let sample: SampleData = SampleData::CAPM; 
        let (_, mut sample_data) = sample.load_sample_data();
        let market_premium: Array1<f64> = sample_data.remove("Market").unwrap() - sample_data.remove("RF").unwrap();
        let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
        // Simple Granger causality test
        let gcr: GrangerCausalityResult = simple_granger_causality_test(&market_premium, &risk_premium, 16, None).unwrap();
        // The results were found using grangercausalitytests from statsmodels.tsa.stattools
        // Nested model tests (i.e., tests of AR model of `Y`)
        assert!((gcr.nested_model_result.as_ref().unwrap().model_result.intercept.unwrap() - 0.01754408).abs() < TEST_ACCURACY);
        let nested_coefs: Vec<f64> = vec![
            0.07144487, 0.07683892, 0.11094491, -0.11023364, -0.14995811, -0.05011439, 0.18793717, 0.26099817, -0.33252268, -0.14385452, 0.19426531,
            0.12370539, -0.00569986, -0.04653284, -0.21241471,  0.31252425,
        ];
        let _ = gcr.nested_model_result.as_ref().unwrap().model_result.coefficients.iter().zip(nested_coefs.iter()).map(|(ar_c, c)|  {
            assert!((ar_c.coefficient - c).abs() < TEST_ACCURACY);
        } );
        // Parent model tests (i.e., tests of the complete Granger linear model)
        assert!((gcr.parent_model_result.as_ref().unwrap().intercept.unwrap() - 1.14274372e-02).abs() < TEST_ACCURACY);
        let nested_coefs: Vec<f64> = vec![
            4.19223242e-01, -2.34404524e-01,  2.72340900e-01,  1.03013448e+00, -1.81718789e-01, -4.27993121e-01,  8.40314944e-01,  6.19071691e-01,
            -6.60135239e-01, -2.97185005e-01, -2.48296270e-03, -3.07798660e-01, 3.65214587e-01,  6.54557206e-01,  6.82388733e-01,  2.57055272e-01,
            -1.16703714e+00,  3.84263021e-01, -6.80947028e-01, -1.98267527e+00, -8.58625697e-02,  1.61253913e-01, -9.51342081e-01,  2.87793511e-01,
            5.41451441e-01,  3.76870986e-01,  5.03655333e-01,  4.78529632e-01, -1.41274516e-01, -7.91401835e-01, -2.74771397e+00, -8.14687698e-01,
        ];
        let _ = gcr.parent_model_result.as_ref().unwrap().coefficients.iter().zip(nested_coefs.iter()).map(|(ar_c, c)|  {
            assert!((ar_c.coefficient - c).abs() < TEST_ACCURACY);
        } );
        // F-test tests
        assert!((gcr.f_test_f_statistic.unwrap() - 0.4512).abs() < 10_000.0 * TEST_ACCURACY);
        assert_eq!(gcr.f_test_dof_1.unwrap(), 16);
        assert_eq!(gcr.f_test_dof_2.unwrap(), 7);
        assert!((gcr.f_test_p_value.unwrap() - 0.9111).abs() < 10_000.0 * TEST_ACCURACY);
        assert_eq!(gcr.reject_h0, false);
        assert!(match gcr.failed_reject_h0_reason.unwrap() { GrangerCausalityRejectReason::FTestRejected => true, _ => false,});
    }
}