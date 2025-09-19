use std::fmt::Display;
use ndarray::{Array1, Array2};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::{
    LARGE_TEXT_BREAK, SMALL_TEXT_BREAK, FeatureCollection,
    loss_functions::{LossFunction, SSE},
};
use crate::statistics::{self, stat_tests};


#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// The data relevant to the specific linear regression feature.
pub struct LinearRegressionFeatureResult {
    /// Optional name of the coefficient or associated feature
    pub coefficient_name: Option<String>,
    /// Value of the coefficient
    pub coefficient: f64,
    /// Standard error of the feature.
    pub standard_error: Option<f64>,
    /// Covariance of the feature (i.e., `Xi`) against the model's dependent variable (i.e., `Y`)
    pub covariance: Option<f64>,
    /// Pearson correlation of the feature (i.e., `Xi`) agains the model's dependent variable (i.e., `Y`)
    pub pearson_correlation: Option<f64>,
    /// Null hypothesis for coefficient of the feature in the linear regression model
    pub t_test_h0: Option<f64>,
    /// T-test result
    pub t_test_reject_h0: Option<bool>,
    /// t score of the t-test
    pub t_test_t_score: Option<f64>,
    /// Degrees of freedom used in the t-test
    pub t_test_dof: Option<f64>,
    /// p-value of the t-test
    pub t_test_p_value: Option<f64>,
    /// Confidence level of the t-test
    pub t_test_cl: Option<f64>,
    /// Cointegration test between the feature (i.e., `Xi`) anf the model's dependent variable (i.e., `Y`)
    pub cointegrated: Option<bool>,
    /// Confidence level of the cointegration test
    pub cointegration_cl: Option<f64>,
    /// Variance inflation factor
    pub vif: Option<f64>,
}

impl Display for LinearRegressionFeatureResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = format!("\tCoefficient {}\n", match &self.coefficient_name { Some(v) => v.to_string(), None => "".to_owned(), })
            + &format!("\tCoefficient: {}\n", self.coefficient)
            + &format!("\tStandard Error: {}\n", self.standard_error.unwrap_or(f64::NAN))
            + &format!("\tCovariance: {}\n", self.covariance.unwrap_or(f64::NAN))
            + &format!("\tPearson Correlation: {}\n", self.pearson_correlation.unwrap_or(f64::NAN))
            + &format!("\tVariance Inflation Factor: {}\n", self.vif.unwrap_or(f64::NAN))
            + SMALL_TEXT_BREAK
            + &format!("\tt-Test Degrees of Freedom: {}", self.t_test_dof.unwrap_or(f64::NAN))
            + &format!("\tt-Test Null Hypothesis: {}\n", self.t_test_h0.unwrap_or(f64::NAN))
            + &format!("\tt-Test Reject Null Hypothesis: {}\n", match self.t_test_reject_h0 { Some(b) => b.to_string(), None => "".to_owned(), })
            + &format!("\tt-score: {}\n", self.t_test_t_score.unwrap_or(f64::NAN))
            + &format!("\tp-value: {}\n", self.t_test_p_value.unwrap_or(f64::NAN))
            + &format!("\tConfidence Level: {}\n", self.t_test_cl.unwrap_or(f64::NAN))
            + SMALL_TEXT_BREAK
            + &format!("\tCointegrated: {}\n", match self.cointegrated { Some(b) => b.to_string(), None => "".to_owned(), })
            + &format!("\tConfidence Level: {}\n", self.cointegration_cl.unwrap_or(f64::NAN));
        write!(f, "{}", result)
    }
}


#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// The data resulting from linear regression analysis.
pub struct LinearRegressionResult {
    /// Trained values of the linear regression moel (in the same order as the features that were input into the model)
    pub all_coefficients: Array1<f64>,
    /// Intercept of the linear regression model
    pub intercept: Option<f64>,
    /// Coefficients of the linear regression model and their metadata
    pub coefficients: Vec<LinearRegressionFeatureResult>,
    /// Delta degrees of freedom of the linear regression model (i.e., degrees of freedom = N - ddof)
    pub ddof: usize,
    /// Residual sum of squares (sum of squared residuals (i.e., SSR) or the sum of squared estimate of errors (i.e., SSE))
    pub sse: Option<f64>,
    /// Coefficient of determination, the ratio of exaplained variance to all variance
    pub r_squared: Option<f64>,
    /// Adjusted R-squared for the upward bias in the R-squared due to estimated values of the parameters used
    pub adjusted_r_squared: Option<f64>,
}

impl Display for LinearRegressionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = LARGE_TEXT_BREAK.to_owned()
            + "Linear Regression Analysis Result\n"
            + LARGE_TEXT_BREAK
            + &format!("\tIntercept: {}\n", self.intercept.unwrap_or(f64::NAN))
            + LARGE_TEXT_BREAK
            + &self.coefficients.iter().map(|f| f.to_string() ).collect::<Vec<String>>().join(LARGE_TEXT_BREAK)
            + LARGE_TEXT_BREAK
            + &format!("Delta Degrees of Freedom: {}\n", self.ddof)
            + &format!("Sum of Squared Estimate of Errors: {}\n", self.sse.unwrap_or(f64::NAN))
            + &format!("Coefficient of Determination: {}\n", self.r_squared.unwrap_or(f64::NAN))
            + &format!("Adjusted Coefficient of Determination: {}\n", self.adjusted_r_squared.unwrap_or(f64::NAN))
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}


#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Settings of the linear regression analysis.
pub struct LinearRegressionSettings {
    /// Whether to compute a residual sum of squares
    pub enable_sse: bool,
    /// Whether to compute R-squared
    pub enable_r_squared: bool,
    /// Whether to compute adjusted R-squared
    pub enable_adjusted_r_squared: bool,
    /// Whetehr to compute standard error for individual features
    pub enable_se: bool,
    /// Whether to compute covariance for individual features
    pub enable_cov: bool,
    /// Whether to compute Pearson correlation for individual features
    pub enable_pearson_corr: bool,
    /// Whether to run t-test for individual features
    pub enable_t_test: bool,
    /// Confidence level for the t-test
    pub t_test_cl: Option<stat_tests::ConfidenceLevel>,
    /// Null hypotheses for the t-test
    pub t_test_h0s: Option<Vec<f64>>,
    /// Whether to run cointegration test for individual features
    pub enable_cointegration: bool,
    /// Confidence level for the cointegration test 
    pub cointegration_cl: Option<stat_tests::ConfidenceLevel>,
    /// Whether to compute VIF for individual features
    pub enable_vif: bool,
}

impl LinearRegressionSettings {

    /// Returns linear regression settings with all settings enabled.
    /// 
    /// Note: This is a short cut to construct the settings for the full linear regression analysis.
    pub fn enable_all() -> Self {
        LinearRegressionSettings {
            enable_sse: true, enable_r_squared: true, enable_adjusted_r_squared: true, enable_se: true, enable_cov: true,
            enable_pearson_corr: true,
            enable_t_test: true, t_test_cl: Some(stat_tests::ConfidenceLevel::default()), t_test_h0s: None,
            enable_cointegration: true, cointegration_cl: Some(stat_tests::ConfidenceLevel::default()),
            enable_vif: true,
        }
    }

    /// Returns linear regression settings with all settings disabled.
    /// 
    /// Note: This is a short cut to construct the settings for the minimalistic linear regression analysis.
    pub fn disable_all() -> Self {
        LinearRegressionSettings {
            enable_sse: false, enable_r_squared: false, enable_adjusted_r_squared: false, enable_se: false, enable_cov: false,
            enable_pearson_corr: false,
            enable_t_test: false, t_test_cl: Some(stat_tests::ConfidenceLevel::default()), t_test_h0s: None,
            enable_cointegration: false, cointegration_cl: Some(stat_tests::ConfidenceLevel::default()),
            enable_vif: false,
        }
    }
}

impl Display for LinearRegressionSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = LARGE_TEXT_BREAK.to_owned()
            + "Linear Regression Settings\n"
            + LARGE_TEXT_BREAK
            + &format!("Enable SSE: {}\n", self.enable_sse)
            + &format!("Enable Coefficient of Determination: {}\n", self.enable_r_squared)
            + &format!("Enable Adjusted COefficient of Determination: {}\n", self.enable_adjusted_r_squared)
            + &format!("Enable SE: {}\n", self.enable_se)
            + &format!("Enable Covariance: {}\n", self.enable_cov)
            + &format!("Enable Pearson Correlation: {}\n", self.enable_pearson_corr)
            + &format!("Enable t-Test: {}\n", self.enable_t_test)
            + &format!("t-Test Confidence Level: {}\n", self.t_test_cl.unwrap_or_default())
            + "t-Test Null Hypotheses: " + &(match &self.t_test_h0s {
                Some(v) => {
                    v.iter().map(|v| v.to_string() ).collect::<Vec<String>>().join(", ")
                },
                None => "".to_owned(),
            }) + "\n"
            + &format!("Enable Cointegration: {}\n", self.enable_cointegration)
            + &format!("Cointegration Confidence Level: {}\n", self.cointegration_cl.unwrap_or_default())
            + &format!("Enable Variance Inflation Factor: {}\n", self.enable_vif)
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}


#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Set of tools for constructing a linear regression model and performing a further analysis of it.
/// 
/// Note: This tool combines other functions and methods from a library underr one implementation that is easy ton set up.
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, FeatureCollection};
/// use digifi::statistics::{LinearRegressionResult, LinearRegressionSettings, LinearRegressionAnalysis};
///
/// #[cfg(feature = "sample_data")]
/// fn test_t_test_two_sample() -> () {
///     use digifi::utilities::SampleData;
///     
///     // Sample data
///     let sample: SampleData = SampleData::CAPM; 
///     let (_, mut sample_data) = sample.load_sample_data();
///     let market_premium: Array1<f64> = sample_data.remove("Market").unwrap() - sample_data.remove("RF").unwrap();
///     let mut fc: FeatureCollection = FeatureCollection::new();
///     fc.add_constant = true;
///     fc.add_feature(market_premium.into_iter(), "Market Premium").unwrap();
///     fc.add_feature(sample_data.remove("SMB").unwrap().into_iter(), "SMB").unwrap();
///     fc.add_feature(sample_data.remove("HML").unwrap().into_iter(), "HML").unwrap();
///     fc.add_feature(sample_data.remove("RMW").unwrap().into_iter(), "RMW").unwrap();
///     fc.add_feature(sample_data.remove("CMA").unwrap().into_iter(), "CMA").unwrap();
///     let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
/// 
///     // Linear regression analysis
///     let lra: LinearRegressionAnalysis = LinearRegressionAnalysis::new(LinearRegressionSettings::enable_all());
///     let lr_result: LinearRegressionResult = lra.run(&mut fc, &risk_premium).unwrap();
/// 
///     // The results were found using LinearRegression from sklearn
///     assert!((lr_result.intercept.unwrap() - 0.01353015).abs() < TEST_ACCURACY);
///     assert!((lr_result.coefficients[0].coefficient - 1.37731033).abs() < TEST_ACCURACY);
///     assert!((lr_result.coefficients[1].coefficient - -0.38490771).abs() < TEST_ACCURACY);
///     assert!((lr_result.coefficients[2].coefficient - -0.58771487).abs() < TEST_ACCURACY);
///     assert!((lr_result.coefficients[3].coefficient - 0.11692186).abs() < TEST_ACCURACY);
///     assert!((lr_result.coefficients[4].coefficient - 0.4192746).abs() < TEST_ACCURACY);
/// }
/// ```
pub struct LinearRegressionAnalysis {
    /// Linear regression analysis settings
    pub settings: LinearRegressionSettings,
}

impl LinearRegressionAnalysis {

    /// Constructs a new instance of `LinearRegressionAnalysis`.
    /// 
    /// # Input
    /// - `settings`: Settings of the linear regression analysis (these determine what analysis and tests will be computed)
    pub fn new(settings: LinearRegressionSettings) -> Self {
        LinearRegressionAnalysis { settings, }
    }

    /// Validates the input data for the linear regression analysis.
    fn validate_input(&self, fc: &FeatureCollection, y: &Array1<f64>) -> Result<(), DigiFiError> {
        let y_len: usize = y.len();
        let fc_len: usize = fc.len();
        // Input validation
        if y.is_empty() {
            return Err(DigiFiError::ValidationError { title: Self::error_title(), details: "Array `y` must not be empty.".to_owned() ,});
        }
        match fc.feature_size() {
            Some(feature_len) => {
                if  feature_len != y_len {
                    return Err(DigiFiError::UnmatchingLength { array_1: "feature".to_owned(), array_2: "y".to_owned(), });
                }
            },
            None => {
                return Err(DigiFiError::ValidationError { title: Self::error_title(), details: "The feature collection is empty.".to_owned(), });
            },
        }
        if y_len < fc.ddof() {
            return Err(DigiFiError::Other { title: Self::error_title(), details: "There are fewer data points in `y` array than `ddof`.".to_owned() });
        }
        if self.settings.enable_t_test {
            match &self.settings.t_test_h0s {
                Some(v) => if v.len() != fc_len {
                    return Err(DigiFiError::UnmatchingLength { array_1: "feature collection".to_owned(), array_2: "t_test_h0s".to_owned(), });
                },
                None => (),
            }
        }
        Ok(())
    }

    /// Runs linear regression model along with the configured tests.
    /// 
    /// # Input
    /// - `fc`: Feature collection with features of the model (i.e., collection of series that represent different features)
    /// - `y`: Observed response values
    /// 
    /// # Errors
    /// - Returns an error if the length of any of the features `Xi` does not match the length of vector `Y`.
    pub fn run(&self, fc: &FeatureCollection, y: &Array1<f64>) -> Result<LinearRegressionResult, DigiFiError> {
        self.validate_input(fc, y)?;
        let y_len: usize = y.len();
        let fc_len: usize = fc.len();
        // Linear regression model
        let x_matrix: Array2<f64> = fc.get_matrix()?;
        let all_coefficients: Array1<f64> = statistics::linear_regression(&x_matrix, y)?;
        let y_prediction: Array1<f64> = x_matrix.dot(&all_coefficients);
        // Conjoined statistics
        let ddof: usize = fc.ddof();
        let mut coefficients: Vec<LinearRegressionFeatureResult> = vec![];
        for i in 0..fc_len {
            let x: &Array1<f64> = &fc.features[i];
            let mut coefficient: LinearRegressionFeatureResult = LinearRegressionFeatureResult::default();
            coefficient.coefficient_name = Some(fc.feature_names[i].clone());
            coefficient.coefficient = all_coefficients[i];
            if self.settings.enable_se { coefficient.standard_error = Some(statistics::se_lr_coefficient(y, &y_prediction, x, ddof)?); }
            if self.settings.enable_cov { coefficient.covariance = Some(statistics::covariance(y, x, 0)?); }
            if self.settings.enable_pearson_corr { coefficient.pearson_correlation = Some(statistics::pearson_correlation(y, x, 1)?); }
            if self.settings.enable_t_test {
                let beta_0: Option<f64> = match &self.settings.t_test_h0s { Some(v) => Some(v[i]), None => None, };
                let t_test_result: stat_tests::TTestResult = stat_tests::t_test_lr(coefficient.coefficient, beta_0, y, &y_prediction, x, ddof, self.settings.t_test_cl)?;
                coefficient.t_test_h0 = beta_0;
                coefficient.t_test_reject_h0 = Some(t_test_result.reject_h0);
                coefficient.t_test_t_score = Some(t_test_result.t_score);
                coefficient.t_test_dof = Some(t_test_result.dof);
                coefficient.t_test_p_value = Some(t_test_result.p_value);
                coefficient.t_test_cl = Some(t_test_result.p_cl);
            }
            if self.settings.enable_cointegration {
                let result: stat_tests::CointegrationResult = stat_tests::cointegration(x, y, self.settings.cointegration_cl)?;
                coefficient.cointegrated = Some(result.cointegrated);
                coefficient.cointegration_cl = self.settings.cointegration_cl.clone().map(|v| v.get_p() );
            }
            if self.settings.enable_vif {
                let mut xis: FeatureCollection = fc.features.iter().zip(fc.feature_names.iter()).enumerate()
                    .filter(|(index, _)| index != &i )
                    .fold(FeatureCollection::new(), |mut fc_, (_, (array, label))| {
                        // Note: Safe unwrap since all arrays are in the same original collection
                        fc_.add_feature(array.iter(), label).unwrap();
                        fc_
                    } );
                coefficient.vif = statistics::variance_inflation_factor(&mut xis, x)?;

            }
            coefficients.push(coefficient);
        }
        // Global statistics
        let intercept: Option<f64> = if fc.add_constant { all_coefficients.last().copied() } else { None };
        let sse: Option<f64> = if self.settings.enable_sse { Some(SSE.loss_iter(y.iter(), y_prediction.iter())?) } else { None };
        let r_squared: Option<f64> = if self.settings.enable_r_squared { Some(statistics::r_squared(y, &y_prediction)?) } else { None };
        let adjusted_r_squared: Option<f64> = if self.settings.enable_adjusted_r_squared { Some(statistics::adjusted_r_squared(y, &y_prediction, y_len, fc_len)?) } else { None };
        Ok(LinearRegressionResult {
            all_coefficients,
            intercept, coefficients,
            ddof, sse, r_squared, adjusted_r_squared,
        })
    }
}

impl ErrorTitle for LinearRegressionAnalysis {
    fn error_title() -> String {
        String::from("Linear Regression Analysis")
    }
}

impl Display for LinearRegressionAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.settings.to_string())
    }
}


#[cfg(all(test, feature = "sample_data"))]
mod tests {
    use ndarray::Array1;
    use crate::utilities::{TEST_ACCURACY, FeatureCollection, sample_data::SampleData};
    use crate::statistics::{LinearRegressionResult, LinearRegressionSettings, LinearRegressionAnalysis};

    #[test]
    fn unit_test_linear_regression_analysis() -> () {
        // Sample data
        let sample: SampleData = SampleData::CAPM; 
        let (_, mut sample_data) = sample.load_sample_data();
        let market_premium: Array1<f64> = sample_data.remove("Market").unwrap() - sample_data.remove("RF").unwrap();
        let mut fc: FeatureCollection = FeatureCollection::new();
        fc.add_constant = true;
        fc.add_feature(market_premium.into_iter(), "Market Premium").unwrap();
        fc.add_feature(sample_data.remove("SMB").unwrap().into_iter(), "SMB").unwrap();
        fc.add_feature(sample_data.remove("HML").unwrap().into_iter(), "HML").unwrap();
        fc.add_feature(sample_data.remove("RMW").unwrap().into_iter(), "RMW").unwrap();
        fc.add_feature(sample_data.remove("CMA").unwrap().into_iter(), "CMA").unwrap();
        let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
        // Linear regression analysis
        let lra: LinearRegressionAnalysis = LinearRegressionAnalysis::new(LinearRegressionSettings::enable_all());
        let lr_result: LinearRegressionResult = lra.run(&mut fc, &risk_premium).unwrap();
        // The results were found using LinearRegression from sklearn
        assert!((lr_result.intercept.unwrap() - 0.01353015).abs() < TEST_ACCURACY);
        assert!((lr_result.coefficients[0].coefficient - 1.37731033).abs() < TEST_ACCURACY);
        assert!((lr_result.coefficients[1].coefficient - -0.38490771).abs() < TEST_ACCURACY);
        assert!((lr_result.coefficients[2].coefficient - -0.58771487).abs() < TEST_ACCURACY);
        assert!((lr_result.coefficients[3].coefficient - 0.11692186).abs() < TEST_ACCURACY);
        assert!((lr_result.coefficients[4].coefficient - 0.4192746).abs() < TEST_ACCURACY);
    }
}