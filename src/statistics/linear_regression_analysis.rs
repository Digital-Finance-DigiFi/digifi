use ndarray::{Array1, Array2};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;
use crate::utilities::loss_functions::{LossFunction, SSE};
use crate::statistics::{self, stat_tests};


#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Thw data relevant to the specific linear regression feature.
pub struct LinearRegressionFeatureResult {
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


#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
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


#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Settings of the linear regression analysis.
pub struct LinearRegressionSettings {
    /// Whether to add a constant (i.e., intercept) to the linear regression model
    pub add_constant: bool,
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

    /// # Description
    /// Returns linear regression settings with all settings enabled.
    /// 
    /// Note: This is a short cut to construct the settings for the full linear regression analysis.
    pub fn enable_all() -> Self {
        LinearRegressionSettings {
            add_constant: true, enable_sse: true, enable_r_squared: true, enable_adjusted_r_squared: true, enable_se: true, enable_cov: true, enable_pearson_corr: true,
            enable_t_test: true, t_test_cl: Some(stat_tests::ConfidenceLevel::default()), t_test_h0s: None,
            enable_cointegration: true, cointegration_cl: Some(stat_tests::ConfidenceLevel::default()),
            enable_vif: true,
        }
    }

    /// # Description
    /// Returns linear regression settings with all settings disabled.
    /// 
    /// Note: This is a short cut to construct the settings for the minimalistic linear regression analysis.
    pub fn disable_all() -> Self {
        LinearRegressionSettings {
            add_constant: false, enable_sse: false, enable_r_squared: false, enable_adjusted_r_squared: false, enable_se: false, enable_cov: false, enable_pearson_corr: false,
            enable_t_test: false, t_test_cl: Some(stat_tests::ConfidenceLevel::default()), t_test_h0s: None,
            enable_cointegration: false, cointegration_cl: Some(stat_tests::ConfidenceLevel::default()),
            enable_vif: false,
        }
    }
}


#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Set of tools for constructing a linear regression model and performing a further analysis of it.
/// 
/// Note: This tool combines other functions and methods from a library underr one implementation that is easy ton set up.
pub struct LinearRegressionAnalysis {
    /// Linear regression analysis settings
    pub settings: LinearRegressionSettings,
}

impl LinearRegressionAnalysis {

    /// # Description
    /// Constructs a new instance of `LinearRegressionAnalysis`.
    /// 
    /// # Input
    /// - `settings`: Settings of the linear regression analysis (these determine what analysis and tests will be computed)
    pub fn new(settings: LinearRegressionSettings) -> Self {
        LinearRegressionAnalysis { settings, }
    }

    /// # Description
    /// Validates the input data for the linear regression analysis.
    fn validate_input(&self, x: &Vec<Array1<f64>>, y: &Array1<f64>) -> Result<(), DigiFiError> {
        let error_title: String = String::from("Linear Regression Analysis");
        let y_len: usize = y.len();
        let x_len: usize = x.len();
        // Input validation
        if y.is_empty() {
            return Err(DigiFiError::ValidationError { title: error_title.clone(), details: "Array `y` must not be empty.".to_owned() ,});
        }
        for v in x {
            if v.len() != y_len {
                return Err(DigiFiError::UnmatchingLength { array_1: "x".to_owned(), array_2: "y".to_owned(), });
            }
        }
        if y_len < x_len || (self.settings.add_constant && y_len < x_len + 1) {
            return Err(DigiFiError::Other { title: error_title.clone(), details: "There are fewer data points in `y` array than `ddof`.".to_owned() });
        }
        if self.settings.enable_t_test {
            match &self.settings.t_test_h0s {
                Some(v) => if v.len() != x_len {
                    return Err(DigiFiError::UnmatchingLength { array_1: "x".to_owned(), array_2: "t_test_h0s".to_owned(), });
                },
                None => (),
            }
        }
        Ok(())
    }

    /// # Description
    /// Runs linear regression model along with the configured tests.
    /// 
    /// # Input
    /// - `x`: Vector with features of the model
    /// - `y`: Observed response values
    /// 
    /// # Errors
    /// - Returns an error if the length of any of the features `Xi` does not match the length of vector `Y`.
    pub fn run(&self, x: &Vec<Array1<f64>>, y: &Array1<f64>) -> Result<LinearRegressionResult, DigiFiError> {
        let y_len: usize = y.len();
        let x_len: usize = x.len();
        self.validate_input(x, y)?;
        // Linear regression model
        let (mut shape, mut x_matrix) = ((x_len, y_len), x.iter().fold(vec![], |mut prev, next| { prev.append(&mut next.to_vec()); prev } ));
        let mut ddof: usize = y_len - x_len;
        if self.settings.add_constant {
            shape.0 += 1;
            x_matrix.append(&mut vec![1.0; y.len()]);
            ddof += 1;
        }
        let x_matrix: Array2<f64> = Array2::from_shape_vec(shape, x_matrix)?.reversed_axes();
        let all_coefficients: Array1<f64> = statistics::linear_regression(&x_matrix, y)?;
        let y_prediction: Array1<f64> = x_matrix.dot(&all_coefficients);
        // Conjoined statistics
        let mut coefficients: Vec<LinearRegressionFeatureResult> = vec![];
        for i in 0..x_len {
            let mut coefficient: LinearRegressionFeatureResult = LinearRegressionFeatureResult::default();
            coefficient.coefficient = all_coefficients[i];
            if self.settings.enable_se { coefficient.standard_error = Some(statistics::se_lr_coefficient(y, &y_prediction, &x[i], ddof)?); }
            if self.settings.enable_cov { coefficient.covariance = Some(statistics::covariance(y, &x[i], 0)?); }
            if self.settings.enable_pearson_corr { coefficient.pearson_correlation = Some(statistics::pearson_correlation(y, &x[i], 1)?); }
            if self.settings.enable_t_test {
                let beta_0: Option<f64> = match &self.settings.t_test_h0s { Some(v) => Some(v[i]), None => None, };
                let t_test_result: stat_tests::TTestResult = stat_tests::t_test_lr(coefficient.coefficient, beta_0, y, &y_prediction, &x[i], ddof, self.settings.t_test_cl.clone())?;
                coefficient.t_test_h0 = beta_0;
                coefficient.t_test_reject_h0 = Some(t_test_result.reject_h0);
                coefficient.t_test_p_value = Some(t_test_result.p_value);
                coefficient.t_test_cl = Some(t_test_result.p_cl);
            }
            if self.settings.enable_cointegration {
                let result: stat_tests::CointegrationResult = stat_tests::cointegration(&x[i], y, self.settings.cointegration_cl.clone())?;
                coefficient.cointegrated = Some(result.cointegrated);
                coefficient.cointegration_cl = self.settings.cointegration_cl.clone().map(|v| v.get_p() );
            }
            if self.settings.enable_vif {
                let xis: Vec<Array1<f64>> = x.iter().enumerate().filter(|(index, _)| index != &i ).map(|(_, e)| e.clone() ).collect();
                coefficient.vif = statistics::variance_inflation_factor(&xis, &x[i])?;

            }
            coefficients.push(coefficient);
        }
        // Global statistics
        let intercept: Option<f64> = if self.settings.add_constant { all_coefficients.last().copied() } else { None };
        let sse: Option<f64> = if self.settings.enable_sse { Some(SSE.loss_array(y, &y_prediction)?) } else { None };
        let r_squared: Option<f64> = if self.settings.enable_r_squared { Some(statistics::r_squared(y, &y_prediction)?) } else { None };
        let adjusted_r_squared: Option<f64> = if self.settings.enable_adjusted_r_squared {Some(statistics::adjusted_r_squared(y, &y_prediction, y_len, x_len)?)} else { None };
        Ok(LinearRegressionResult {
            all_coefficients,
            intercept, coefficients,
            ddof, sse, r_squared, adjusted_r_squared,
        })
    }
}


#[cfg(all(test, feature = "sample_data"))]
mod tests {
    use ndarray::Array1;
    use crate::utilities::{TEST_ACCURACY, sample_data::SampleData};
    use crate::statistics::{LinearRegressionResult, LinearRegressionSettings, LinearRegressionAnalysis};

    #[test]
    fn unit_test_linear_regression_analysis() -> () {
        // Get sample data
        let sample: SampleData = SampleData::CAPM; 
        let (_, mut sample_data) = sample.load_sample_data();
        let market_premium: Array1<f64> = sample_data.remove("Market").unwrap() - sample_data.remove("RF").unwrap();
        let x: Vec<Array1<f64>> = vec![
            market_premium,
            sample_data.remove("SMB").unwrap(), sample_data.remove("HML").unwrap(), sample_data.remove("RMW").unwrap(), sample_data.remove("CMA").unwrap(),
        ];
        let risk_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
        // Linear regression analysis
        let settings: LinearRegressionSettings = LinearRegressionSettings::enable_all();
        let lra: LinearRegressionAnalysis = LinearRegressionAnalysis::new(settings);
        let lr_result: LinearRegressionResult = lra.run(&x, &risk_premium).unwrap();
        println!("{:?}", &lr_result);
        // The results were found using LinearRegression from sklearn
        assert!((lr_result.intercept.unwrap() - 0.01353015).abs() < TEST_ACCURACY);
        assert!((lr_result.coefficients[0].coefficient - 1.37731033).abs() < TEST_ACCURACY);
        assert!((lr_result.coefficients[1].coefficient - -0.38490771).abs() < TEST_ACCURACY);
        assert!((lr_result.coefficients[2].coefficient - -0.58771487).abs() < TEST_ACCURACY);
        assert!((lr_result.coefficients[3].coefficient - 0.11692186).abs() < TEST_ACCURACY);
        assert!((lr_result.coefficients[4].coefficient - 0.4192746).abs() < TEST_ACCURACY);
    }
}