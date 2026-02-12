use std::fmt::Display;
use ndarray::{Array1, s};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::{
    LARGE_TEXT_BREAK, FeatureCollection,
    maths_utils::erfinv,
};
use crate::statistics::{
    akaike_information_criterion_log, bayesian_information_criterion_log,
    autocorrelation::{PartialAutocorrelation, partial_autocorrelation},
    stat_tests::ConfidenceLevel,
    linear_regression_analysis::{LinearRegressionResult, LinearRegressionSettings, LinearRegressionAnalysis},
};


#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ARResult {
    /// Order of the AR model
    pub order: usize,
    /// Result of the AR model fitting
    pub model_result: LinearRegressionResult,
    /// Maximum lag used in computing partial autocorrelations
    /// 
    /// Note: This will only be computed if `PAC` method is used for finding the order of the AR model
    pub pac_max_lag: Option<usize>,
    /// Confidence interval for the partial autocorrelations of `Y`
    /// 
    /// Note: This will only be computed if `PAC` method is used for finding the order of the AR model
    pub pac_cl: Option<f64>,
    /// Critical value for partial autocorrelations of `Y`, obtained from `pac_cl`
    /// 
    /// Note: This will only be computed if `PAC` method is used for finding the order of the AR model
    pub pac_crit: Option<f64>,
    /// Residual sums of squares for every fitted model when maximizing an information criterion of choice.
    /// 
    /// Note: This will only be computed if `AIC` or `BIC` methods are used for finding the order of the AR model.
    pub ic_rss: Option<Vec<f64>>,
    /// Maximum log-likelihoods for every fitted model when maximizing an information criterion of choice.
    /// 
    /// Note: This will only be computed if `AIC` or `BIC` methods are used for finding the order of the AR model.
    pub ic_max_log_likelihood: Option<Vec<f64>>,
    /// Maximum likelihoods for every fitted model when maximizing an information criterion of choice.
    /// 
    /// Note: This will only be computed if `AIC` or `BIC` methods are used for finding the order of the AR model.
    pub ic_max_likelihood: Option<Vec<f64>>,
    /// Information criterion values for every fitted model when maximizing an information criterion of choice.
    /// 
    /// Note: This will only be computed if `AIC` or `BIC` methods are used for finding the order of the AR model.
    pub ic_values: Option<Vec<f64>>,
}

impl Display for ARResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = LARGE_TEXT_BREAK.to_owned()
            + "AR Model Result\n"
            + LARGE_TEXT_BREAK
            + &format!("Order: {}\n", self.order)
            + &self.model_result.to_string()
            + &format!("Partial Autocorrelation - Maximum Lag: {}\n", match &self.pac_max_lag { Some(v) => v.to_string(), None => "".to_owned(), })
            + &format!("Partial Autocorrelation - Confidence Level: {}\n", self.pac_cl.unwrap_or(f64::NAN))
            + &format!("Partial Autocorrelation - Critical Value: {}\n", self.pac_crit.unwrap_or(f64::NAN))
            + &format!("Information Criterion - RSS's: {:?}\n", match &self.ic_rss { Some(v) => format!("{:?}", v), None => "".to_owned(), })
            + &format!("Information Criterion - Maximum Log-Likelihoods: {:?}\n", match &self.ic_max_log_likelihood { Some(v) => format!("{:?}", v), None => "".to_owned(), })
            + &format!("Information Criterion - Maximum Likelihoods: {:?}\n", match &self.ic_max_likelihood { Some(v) => format!("{:?}", v), None => "".to_owned(), })
            + &format!("Information Criterion - Values: {:?}\n", match &self.ic_values { Some(v) => format!("{:?}", v), None => "".to_owned(), })
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Method for determining the order (i.e., the number of lag terms) of the AR model.
pub enum AROrderMethod {
    /// Order of AR model that is set manually
    Manual {
        /// Order of the AR model
        order: usize,
    },
    /// Partial Autocorrelation
    PAC {
        /// Maximum lag used in computing partial autocorrelations
        max_lag: Option<usize>,
        /// Confidence level for partial autocorrelation. This level is used to find the order of the AR model of `Y`
        /// such that the order is the lag after which the partial autocorrelations are all within the confidence interval.
        /// 
        /// Note: This confidence level is used to compute the symmetrical critical values (i.e., ± critical values) which act as the cut-off points
        /// for the order of the autoregressive model of `Y`.
        cl: Option<f64>,
    },
    /// Akaike Information Criterion
    AIC {
        /// Maximum lag used in the AR model when optimizing Akaike information criterion
        /// 
        /// Note: This may not be the actual order of the model, it is the maximum for what the order of the model may be.
        max_lag: Option<usize>,
    },
    /// Bayesion Information Criterion
    BIC {
        /// Maximum lag used in the AR model when optimizing Akaike information criterion
        /// 
        /// Note: This may not be the actual order of the model, it is the maximum for what the order of the model may be.
        max_lag: Option<usize>,
    },
}

impl AROrderMethod {
    /// Returns the the maximum lag of the AR model. If no order or maximum lag provided, will default to
    /// `min(10 * n.log10(), n // 2 - 1)`
    /// 
    /// Note: In the case of `PAC`, `AIC` and `BIC` the maximum lag corresponds to the potential maximum lag of the model,
    /// but once the model is fitted and optimized, its order might be smaller than the maximum lag returned by this function.
    /// To set the exact order of the AR model use `Manual` enum variant.
    ///  
    /// 
    /// # Input
    /// - `n`: Number of data points in time series `Y`
    /// 
    /// # Output
    /// - Maximum lag to be used in the AR model
    /// 
    /// # Errors
    /// - Returns an error if the maximum lag is bigger of equal to `n // 2` (i.e., `n` divided by `2` without the remainder)
    pub fn max_lag(&self, n: usize) -> Result<usize, DigiFiError> {
        let n_div_two: u32 = (n as u32).div_euclid(2);
        let max_lag: usize = match &self {
            Self::Manual { order } => *order,
            Self::PAC { max_lag, .. } if max_lag.is_some() => { max_lag.unwrap_or_default() },
            Self::AIC { max_lag } if max_lag.is_some() => { max_lag.unwrap_or_default() },
            Self::BIC { max_lag } if max_lag.is_some() => { max_lag.unwrap_or_default() },
            _ => (10 * n.ilog10()).min(n_div_two - 1) as usize,
        };
        if (n_div_two as usize) < max_lag {
            return Err(DigiFiError::ValidationError {
                title: Self::error_title(),
                details: "Maximum lag `max_lag` must be smaller than `n // 2`.".to_owned(),
            });
        }
        Ok(max_lag)
    }

    /// Returns the order (i.e., the maximum lag) for partial autocorrelation.
    /// 
    /// # Input
    /// - `y`: Time series
    /// 
    /// # Errors
    /// - Returns an error if the instance of the enum is not `PAC`.
    pub fn pac_order(&self, y: &Array1<f64>) -> Result<(usize, f64, f64, usize), DigiFiError> {
        match self {
            AROrderMethod::PAC { .. } => (),
            _ => return Err(DigiFiError::Other {
                title: Self::error_title(),
                details: "The method `pac_order` is only supported for the `PAC` variant of the enum.".to_owned() }),
        }
        let y_len: usize = y.len();
        let pac_max_lag: usize = self.max_lag(y_len)?;
        let pac: PartialAutocorrelation = partial_autocorrelation(y, pac_max_lag)?;
        let pac_ci: f64 = self.pac_cl()?;
        // The values of the sample partial autocorrelation sequence for lags greater than `p` (i.e., in the AR(p) model) follow a `N(0,1/N)` distribution,
        // where `N` is the length of the time series. For a 95% confidence interval, the critical value is `sqrt(2) * erfinv(ci)`
        // and the confidence interval is `Δ = ± 1.96 / sqrt(N)`
        let pac_crit: f64 = (2.0 / y_len as f64).sqrt() * erfinv(pac_ci, None);
        let order: usize = pac.partial_autocorrelations.iter().enumerate().fold(0, |order, (i, pac)| {
            if pac_crit <= pac.abs() { i } else { order }
        } );
        if order == 0 {
            return Err(DigiFiError::ValidationError {
                title: Self::error_title(),
                details: "The maximum lag of `Y` is `0`, so no AR model can be constructed for time series `Y` with current settings.".to_owned(),
            });
        }
        Ok((pac_max_lag, pac_ci, pac_crit, order))
    }

    /// Returns the confidence level to be used for the partial autocorrelation computation.
    /// 
    /// # Errors
    /// - Returns an error if the instance of the enum is not `PAC`.
    /// - Returns an error if the confidence level `pac_cl` is outside of the range `[0, 1]`.
    pub fn pac_cl(&self) -> Result<f64, DigiFiError> {
        match self {
            Self::PAC { cl, .. } => {
                match cl {
                    Some(cl) => {
                        if *cl < 0.0 || 1.0 < *cl {
                            return Err(DigiFiError::ParameterConstraint {
                                title: Self::error_title(),
                                constraint: "The confidence level `pac_cl` must be within the range `[0, 1]`.".to_owned(),
                            })
                        }
                        Ok(*cl)
                    },
                    None => Ok(1.0 - ConfidenceLevel::default().get_p()),
                }
            },
            _ => Err(DigiFiError::Other {
                title: Self::error_title(),
                details: "The method `pac_cl` is only supported for the `PAC` variant of the enum.".to_owned(),
            }),
        }
    }

    /// Returns the value for the specified information criterion.
    /// 
    /// # Input
    /// - `ll`: Maximized value of the log-likelihood function for the model
    /// - `k`: Number of estimated parameters in the model
    /// - `n`: Number of data points, number of observations, or the sample size
    /// 
    /// # Ouput
    /// - The value of the specified information criterion
    /// 
    /// # Errors
    /// - Returns an error if the instance of the enum is not `AIC` or `BIC`.
    pub fn ic_value(&self, ll: f64, k: usize, n: usize) -> Result<f64, DigiFiError> {
        match self {
            AROrderMethod::AIC { .. } => Ok(akaike_information_criterion_log(ll, k)),
            AROrderMethod::BIC { .. } => Ok(bayesian_information_criterion_log(ll, k, n)),
            _ => Err(DigiFiError::Other {
                title: Self::error_title(),
                details: "The method `ic_value` is only supported for the `AIC` or `BIC` variants of the enum.".to_owned(),
            })
        }
    }
}

impl Default for AROrderMethod {
    fn default() -> Self {
        Self::PAC { max_lag: None, cl: None, }
    }
}

impl Display for AROrderMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Manual { order } => write!(f, "Manual (Order: {})", order),
            Self::PAC { max_lag, cl } => {
                write!(f, "Partial Autocorrelation (Max Lag: {:?}, Confidence Level: {:?})", max_lag, cl)
            },
            Self::AIC { max_lag } => write!(f, "Akaike Information Criterion (Max Lag: {:?})", max_lag),
            Self::BIC { max_lag } => write!(f, "Bayesian Information Criterion (Max Lag: {:?})", max_lag),
        }
    }
}

impl ErrorTitle for AROrderMethod {
    fn error_title() -> String {
        String::from("AR Model - Order Method")
    }
}


#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Settings of the AR model.
pub struct ARSettings {
    /// Method for computing the order of the AR model
    pub order_method: AROrderMethod,
    /// Whether to return the feature collection that was constructed to train the model
    pub return_fc: bool,
    /// Whether to add a constant term to the AR model
    pub add_constant: bool,
    /// Additional settings for the AR model
    pub additional_settings: Option<LinearRegressionSettings>,
}

impl Display for ARSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = LARGE_TEXT_BREAK.to_owned()
            + "AR Model Settings\n"
            + LARGE_TEXT_BREAK
            + &format!("Method for Computing  the Order: {}\n", &self.order_method)
            + &format!("Return Feature Collection: {}\n", self.return_fc)
            + &format!("Add Constant Term: {}\n", self.add_constant)
            + LARGE_TEXT_BREAK
            + "Additional Settings\n"
            + &format!("{}", match &self.additional_settings {
                Some(s) => s.to_string(),
                None => AR::new_lr_settings().to_string(),
            })
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}


#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Autoregressive model (AR model).
/// 
/// # Examples
/// 
/// 1. AR model (Model order setting: Manual)
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{AROrderMethod, ARSettings, AR};
/// 
/// #[cfg(feature = "sample_data")]
/// fn test_ar_manual() -> () {
///     use digifi::utilities::SampleData;
/// 
///     // Sample data
///     let sample: SampleData = SampleData::CAPM;
///     let (_, mut sample_data) = sample.load_sample_data();
///     let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
/// 
///     // AR model (Model order setting: Manual)
///     let order: usize = 3;
///     let settings: ARSettings = ARSettings {
///         order_method: AROrderMethod::Manual { order, }, return_fc: true, add_constant: true, additional_settings: None, 
///     };
///     let ar: AR = AR::build(settings).unwrap();
///     let (ar_result, fc) = ar.run(&risk_premium).unwrap();
/// 
///     // The results were found using LinearRegression from sklearn
///     // Order test
///     assert_eq!(ar_result.order, order);
///     assert_eq!(fc.unwrap().len(), order);
///     // Coefficients test
///     assert!((ar_result.model_result.intercept.unwrap() - 0.034719354878670124).abs() < TEST_ACCURACY);
///     assert!((ar_result.model_result.coefficients[0].coefficient - 0.05102568).abs() < TEST_ACCURACY);
///     assert!((ar_result.model_result.coefficients[1].coefficient - -0.25217789).abs() < TEST_ACCURACY);
///     assert!((ar_result.model_result.coefficients[2].coefficient - 0.0873814).abs() < TEST_ACCURACY);
/// }
/// ```
/// 
/// 2. AR model (Model order setting: Partial Autocorrelation)
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{AROrderMethod, ARSettings, AR};
/// 
/// #[cfg(feature = "sample_data")]
/// fn test_ar_pac() -> () {
///     use digifi::utilities::SampleData;
/// 
///     // Sample data
///     let sample: SampleData = SampleData::CAPM;
///     let (_, mut sample_data) = sample.load_sample_data();
///     let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
/// 
///     // AR model (Model order setting: Partial Autocorrelation)
///     let max_lag: usize = 20;
///     let settings: ARSettings = ARSettings {
///         order_method: AROrderMethod::PAC { max_lag: Some(max_lag), cl: Some(0.9), }, return_fc: true, add_constant: true, additional_settings: None, 
///     };
///     let ar: AR = AR::build(settings).unwrap();
///     let (ar_result, _) = ar.run(&risk_premium).unwrap();
/// 
///     // The results were found using LinearRegression from sklearn
///     // Order test
///     assert!(ar_result.order <= max_lag);
///     assert_eq!(ar_result.order, 16);
///     // Coefficients test
///     assert!((ar_result.model_result.intercept.unwrap() - 0.017544077338390595).abs() < TEST_ACCURACY);
///     let coefs: Vec<f64> = vec![0.07144487, 0.07683892, 0.11094491, -0.11023364, -0.14995811, -0.05011439, 0.18793717,
///         0.26099817, -0.33252268, -0.14385452, 0.19426531, 0.12370539, -0.00569986, -0.04653284, -0.21241471, 0.31252425,
///     ];
///     let _ = ar_result.model_result.coefficients.iter().zip(coefs.iter()).map(|(ar_c, c)|  {
///         assert!((ar_c.coefficient - c).abs() < TEST_ACCURACY);
///     } );
/// }
/// ```
/// 
/// 3. AR model (Model order setting: Akaike Information Criterion)
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{AROrderMethod, ARSettings, AR};
/// 
/// #[cfg(feature = "sample_data")]
/// fn test_ar_aic() -> () {
///     use digifi::utilities::SampleData;
/// 
///     // Sample data
///     let sample: SampleData = SampleData::CAPM;
///     let (_, mut sample_data) = sample.load_sample_data();
///     let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
/// 
///     // AR model (Model order setting: Akaike Information Criterion)
///     let max_lag: usize = 20;
///     let settings: ARSettings = ARSettings {
///         order_method: AROrderMethod::AIC { max_lag: Some(max_lag) }, return_fc: true, add_constant: true, additional_settings: None, 
///     };
///     let ar: AR = AR::build(settings).unwrap();
///     let (ar_result, _) = ar.run(&risk_premium).unwrap();
/// 
///     // The results were found using LinearRegression from sklearn
///     // Order test
///     assert!(ar_result.order <= max_lag);
///     assert_eq!(ar_result.order, 1);
///     let (argmin, _) = ar_result.ic_values
///         .as_ref().unwrap()
///         .iter().enumerate().fold((0, f64::INFINITY), |(mut index, mut min), (i, value)| {
///             if *value < min {
///                 min = *value;
///                 index = i;
///             }
///             (index, min)
///         } );
///     assert_eq!(argmin, 0); // Index with smallest AIC value
///     // Coefficients test
///     assert!((ar_result.model_result.intercept.unwrap() - 0.03128657003559562).abs() < TEST_ACCURACY);
///     assert!((ar_result.model_result.coefficients[0].coefficient - 0.02728593).abs() < TEST_ACCURACY);
/// }
/// ```
/// 
/// 4. AR model (Model order setting: Bayesian Information Criterion)
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{AROrderMethod, ARSettings, AR};
/// 
/// #[cfg(feature = "sample_data")]
/// fn test_ar_aic() -> () {
///     use digifi::utilities::SampleData;
/// 
///     // Sample data
///     let sample: SampleData = SampleData::CAPM;
///     let (_, mut sample_data) = sample.load_sample_data();
///     let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
/// 
///     // AR model (Model order setting: Bayesian Information Criterion)
///     let max_lag: usize = 20;
///     let settings: ARSettings = ARSettings {
///         order_method: AROrderMethod::BIC { max_lag: Some(max_lag) }, return_fc: true, add_constant: true, additional_settings: None, 
///     };
///     let ar: AR = AR::build(settings).unwrap();
///     let (ar_result, _) = ar.run(&risk_premium).unwrap();
/// 
///     // The results were found using LinearRegression from sklearn
///     // Order test
///     assert!(ar_result.order <= max_lag);
///     assert_eq!(ar_result.order, 1);
///     let (argmin, _) = ar_result.ic_values
///         .as_ref().unwrap()
///         .iter().enumerate().fold((0, f64::INFINITY), |(mut index, mut min), (i, value)| {
///             if *value < min {
///                 min = *value;
///                 index = i;
///             }
///             (index, min)
///         } );
///     assert_eq!(argmin, 0); // Index with smallest AIC value
///     // Coefficients test
///     assert!((ar_result.model_result.intercept.unwrap() - 0.03128657003559562).abs() < TEST_ACCURACY);
///     assert!((ar_result.model_result.coefficients[0].coefficient - 0.02728593).abs() < TEST_ACCURACY);
/// }
/// ```
pub struct AR {
    /// AR model settings
    settings: ARSettings,
}

impl AR {

    /// Constructs a new instance of `AR`.
    /// 
    /// # Input
    /// - `settings`: Settings of the AR model (i.e., autoregressive model)
    pub fn build(settings: ARSettings) -> Result<Self, DigiFiError> {
        // Validation
        match (&settings.additional_settings, &settings.order_method) {
            (Some(s), AROrderMethod::AIC { .. }) | (Some(s), AROrderMethod::BIC { .. }) => {
                if !s.enable_sse {
                    return Err(DigiFiError::ParameterConstraint {
                        title: Self::error_title(),
                        constraint: "The `enable_sse` setting must be set to `true` in the additional settings if methods `AIC` or `BIC` are used.".to_owned(),
                    });
                }
                if !s.enable_max_likelihood {
                    return Err(DigiFiError::ParameterConstraint {
                        title: Self::error_title(),
                        constraint: "The `enable_max_likelihood` setting must be set to `true` in the additional settings if methods `AIC` or `BIC` are used.".to_owned(),
                    })
                }
            },
            _ => (),
        }
        Ok(Self { settings, })
    }

    /// Validates the order of the AR model.
    fn validate_order(order: usize) -> Result<(), DigiFiError> {
        if order == 0 {
            return Err(DigiFiError::ValidationError {
                title: Self::error_title(),
                details: "The maximum lag of `Y` is `0`, so no AR model can be constructed for time series `Y` with current settings.".to_owned(),
            });
        }
        Ok(())
    }

    /// Wraps feature collection in an `Option` enum.
    fn wrap_fc(&self, fc: FeatureCollection) -> Option<FeatureCollection> {
        if self.settings.return_fc { Some(fc) } else { None }
    }

    /// Creates the new linear regression analysis settings with 
    fn new_lr_settings() -> LinearRegressionSettings {
        let mut lr_settings: LinearRegressionSettings = LinearRegressionSettings::disable_all();
        lr_settings.enable_sse = true;
        lr_settings.enable_max_likelihood = true;
        lr_settings
    }

    /// Trains the linear regression model for the provided feature collection and `Y`.
    /// 
    /// Note: Uses settings for linear regression analysis provided in `ARSettings` under `additional_settings`.
    fn lr(&self, fc: &FeatureCollection, y: &Array1<f64>, order: usize) -> Result<LinearRegressionResult, DigiFiError> {
        let settings: LinearRegressionSettings = match &self.settings.additional_settings {
            Some(s) => s.clone(),
            None => Self::new_lr_settings(),
        };
        let lra: LinearRegressionAnalysis = LinearRegressionAnalysis::new(settings);
        lra.run(&fc, &y.slice(s![order..y.len()]).to_owned())
    }

    /// Constructs a feature collection for the specified order and trains the AR model.
    fn complete_ar(&self, y: &Array1<f64>, order: usize) -> Result<(FeatureCollection, LinearRegressionResult), DigiFiError> {
        // Create feature collection from time series `Y` with different lags
        let y_len: usize = y.len();
        let mut fc: FeatureCollection = FeatureCollection::new();
        fc.add_constant = self.settings.add_constant;
        for lag in (1..=order).into_iter() {
            fc.add_feature(y.slice(s![(order - lag)..(y_len - lag)]).into_iter(), &format!("Y(t-{})", lag))?;
        }
        // Create and run AR(order) model for `Y`
        let lra_result: LinearRegressionResult = self.lr(&fc, y, order)?;
        Ok((fc, lra_result))
    }

    /// Trains the AR model to maximize the provided information criterion.
    fn ic_iter_ar(&self, y: &Array1<f64>) -> Result<(ARResult, Option<FeatureCollection>), DigiFiError> {
        let mut result: ARResult = ARResult::default();
        let y_len: usize = y.len();
        let max_lag: usize = self.settings.order_method.max_lag(y_len)?;
        Self::validate_order(max_lag)?;
        let max_lag_minus_one: usize = max_lag - 1;
        let n_constant_term: usize = if self.settings.add_constant { 1 } else { 0 };
        let mut min_ic: f64 = f64::INFINITY;
        let mut model_order: usize = 1;
        let mut ic_rss: Vec<f64> = Vec::with_capacity(max_lag_minus_one);
        let mut ic_max_log_likelihood: Vec<f64> = Vec::with_capacity(max_lag_minus_one);
        let mut ic_max_likelihood: Vec<f64> = Vec::with_capacity(max_lag_minus_one);
        let mut ic_values: Vec<f64> = Vec::with_capacity(max_lag_minus_one);
        // Iterator over models with different lag to find the model that maximizes the information criterion
        for order in (1..=max_lag).into_iter() {
            // Feature collection must not be recycled due to different lengths of feeatures for every `order`
            let (_, lra_result) = self.complete_ar(y, order)?;
            let max_log_likelihood: f64 = lra_result.max_log_likelihood.unwrap_or_default();
            let k: usize = order + n_constant_term;
            let ic: f64 = self.settings.order_method.ic_value(max_log_likelihood, k, y_len)?;
            if ic < min_ic {
                min_ic = ic;
                model_order = order;
            }
            ic_rss.push(lra_result.sse.unwrap_or_default());
            ic_max_log_likelihood.push(max_log_likelihood);
            ic_max_likelihood.push(lra_result.max_likelihood.unwrap_or_default());
            ic_values.push(ic);
        }
        // Construct final version of the feature collection and refit the model to it
        let (fc, lra_result) = self.complete_ar(y, model_order)?;
        result.order = model_order;
        result.model_result = lra_result;
        result.ic_rss = Some(ic_rss);
        result.ic_max_log_likelihood = Some(ic_max_log_likelihood);
        result.ic_max_likelihood = Some(ic_max_likelihood);
        result.ic_values = Some(ic_values);
        Ok((result, self.wrap_fc(fc)))
    }

    /// Trains and runs the AR model.
    /// 
    /// # Input
    /// - `y`: Time series
    pub fn run(&self, y: &Array1<f64>) -> Result<(ARResult, Option<FeatureCollection>), DigiFiError> {
        let y_len: usize = y.len();
        let mut result: ARResult = ARResult::default();
        match self.settings.order_method {
            AROrderMethod::Manual { .. } => {
                // Determine the order of the AR model of `Y` (Manual)
                let order: usize = self.settings.order_method.max_lag(y_len)?;
                Self::validate_order(order)?;
                // Run the AR model for `Y`
                let (fc, lra_result) = self.complete_ar(y, order)?;
                result.order = order;
                result.model_result = lra_result;
                Ok((result, self.wrap_fc(fc)))
            },
            AROrderMethod::PAC { .. } => {
                // Determine the order of the AR model of `Y` (Partial autocorrelation)
                let (pac_max_lag, pac_cl, pac_crit, order) = self.settings.order_method.pac_order(y)?;
                Self::validate_order(order)?;
                // Run the AR model for `Y`
                let (fc, lra_result) = self.complete_ar(y, order)?;
                result.order = order;
                result.model_result = lra_result;
                result.pac_max_lag = Some(pac_max_lag);
                result.pac_cl = Some(pac_cl);
                result.pac_crit = Some(pac_crit);
                Ok((result, self.wrap_fc(fc)))
            },
            AROrderMethod::AIC { .. } => self.ic_iter_ar(y),
            AROrderMethod::BIC { .. } => self.ic_iter_ar(y),
        }
    }
}

impl ErrorTitle for AR {
    fn error_title() -> String {
        String::from("AR Model")
    }
}


#[cfg(all(test, feature = "sample_data"))]
mod tests {
    use ndarray::Array1;
    use crate::utilities::{TEST_ACCURACY, sample_data::SampleData};
    use crate::statistics::arima::ar::{AROrderMethod, ARSettings, AR};
    
    #[test]
    fn unit_test_ar_manual() -> () {
        // Sample data
        let sample: SampleData = SampleData::CAPM;
        let (_, mut sample_data) = sample.load_sample_data();
        let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
        // AR model (Model order setting: Manual)
        let order: usize = 3;
        let settings: ARSettings = ARSettings {
            order_method: AROrderMethod::Manual { order, }, return_fc: true, add_constant: true, additional_settings: None, 
        };
        let ar: AR = AR::build(settings).unwrap();
        let (ar_result, fc) = ar.run(&risk_premium).unwrap();
        // The results were found using LinearRegression from sklearn
        // Order test
        assert_eq!(ar_result.order, order);
        assert_eq!(fc.unwrap().len(), order);
        // Coefficients test
        assert!((ar_result.model_result.intercept.unwrap() - 0.034719354878670124).abs() < TEST_ACCURACY);
        assert!((ar_result.model_result.coefficients[0].coefficient - 0.05102568).abs() < TEST_ACCURACY);
        assert!((ar_result.model_result.coefficients[1].coefficient - -0.25217789).abs() < TEST_ACCURACY);
        assert!((ar_result.model_result.coefficients[2].coefficient - 0.0873814).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_ar_pac() -> () {
        // Sample data
        let sample: SampleData = SampleData::CAPM;
        let (_, mut sample_data) = sample.load_sample_data();
        let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
        // AR model (Model order setting: Partial Autocorrelation)
        let max_lag: usize = 20;
        let settings: ARSettings = ARSettings {
            order_method: AROrderMethod::PAC { max_lag: Some(max_lag), cl: Some(0.9), }, return_fc: true, add_constant: true, additional_settings: None, 
        };
        let ar: AR = AR::build(settings).unwrap();
        let (ar_result, _) = ar.run(&risk_premium).unwrap();
        // The results were found using LinearRegression from sklearn
        // Order test
        assert!(ar_result.order <= max_lag);
        assert_eq!(ar_result.order, 16);
        // Coefficients test
        assert!((ar_result.model_result.intercept.unwrap() - 0.017544077338390595).abs() < TEST_ACCURACY);
        let coefs: Vec<f64> = vec![0.07144487, 0.07683892, 0.11094491, -0.11023364, -0.14995811, -0.05011439, 0.18793717,
            0.26099817, -0.33252268, -0.14385452, 0.19426531, 0.12370539, -0.00569986, -0.04653284, -0.21241471, 0.31252425,
        ];
        let _ = ar_result.model_result.coefficients.iter().zip(coefs.iter()).map(|(ar_c, c)|  {
            assert!((ar_c.coefficient - c).abs() < TEST_ACCURACY);
        } );
    }

    #[test]
    fn unit_test_ar_aic() -> () {
        // Sample data
        let sample: SampleData = SampleData::CAPM;
        let (_, mut sample_data) = sample.load_sample_data();
        let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
        // AR model (Model order setting: Akaike Information Criterion)
        let max_lag: usize = 20;
        let settings: ARSettings = ARSettings {
            order_method: AROrderMethod::AIC { max_lag: Some(max_lag) }, return_fc: true, add_constant: true, additional_settings: None, 
        };
        let ar: AR = AR::build(settings).unwrap();
        let (ar_result, _) = ar.run(&risk_premium).unwrap();
        // The results were found using LinearRegression from sklearn
        // Order test
        assert!(ar_result.order <= max_lag);
        assert_eq!(ar_result.order, 1);
        let (argmin, _) = ar_result.ic_values
            .as_ref().unwrap()
            .iter().enumerate().fold((0, f64::INFINITY), |(mut index, mut min), (i, value)| {
                if *value < min {
                    min = *value;
                    index = i;
                }
                (index, min)
            } );
        assert_eq!(argmin, 0); // Index with smallest AIC value
        // Coefficients test
        assert!((ar_result.model_result.intercept.unwrap() - 0.03128657003559562).abs() < TEST_ACCURACY);
        assert!((ar_result.model_result.coefficients[0].coefficient - 0.02728593).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_ar_bic() -> () {
        // Sample data
        let sample: SampleData = SampleData::CAPM;
        let (_, mut sample_data) = sample.load_sample_data();
        let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
        // AR model (Model order setting: Bayesian Information Criterion)
        let max_lag: usize = 20;
        let settings: ARSettings = ARSettings {
            order_method: AROrderMethod::BIC { max_lag: Some(max_lag) }, return_fc: true, add_constant: true, additional_settings: None, 
        };
        let ar: AR = AR::build(settings).unwrap();
        let (ar_result, _) = ar.run(&risk_premium).unwrap();
        // The results were found using LinearRegression from sklearn
        // Order test
        assert!(ar_result.order <= max_lag);
        assert_eq!(ar_result.order, 1);
        let (argmin, _) = ar_result.ic_values
            .as_ref().unwrap()
            .iter().enumerate().fold((0, f64::INFINITY), |(mut index, mut min), (i, value)| {
                if *value < min {
                    min = *value;
                    index = i;
                }
                (index, min)
            } );
        assert_eq!(argmin, 0); // Index with smallest AIC value
        // Coefficients test
        assert!((ar_result.model_result.intercept.unwrap() - 0.03128657003559562).abs() < TEST_ACCURACY);
        assert!((ar_result.model_result.coefficients[0].coefficient - 0.02728593).abs() < TEST_ACCURACY);
    }
}