use std::fmt::Display;
use ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;
use crate::utilities::{LARGE_TEXT_BREAK, SMALL_TEXT_BREAK, compare_len, FeatureCollection};
use crate::statistics::{
    ProbabilityDistribution, continuous_distributions::NormalDistribution, stat_tests::ConfidenceLevel,
    linear_regression_analysis::{LinearRegressionSettings, LinearRegressionFeatureResult, LinearRegressionResult, LinearRegressionAnalysis},
};


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Result of the first or second analysis step.
pub struct BKMediationAnalysisStep {
    step_number: usize,
    /// Intercept (Alpha) of the linear regression
    pub alpha: f64,
    /// Slope (Beta) of the linear regression
    pub beta: f64,
    /// Standard error of the slope coefficient
    pub beta_se: f64,
    /// Delta degrees of freedom
    pub ddof: usize,
    /// Degrees of freedom used in the test
    pub dof: f64,
    /// Whether to reject the null hypothesis
    /// 
    /// Note: Rejecting null hypothesis means that there is significant relationship
    pub reject_h0: bool,
    /// t-statistic of the t-test
    pub t_score: f64,
    /// p-value of the t-test
    pub p_value: f64,
    /// Confidence level for t-test (Quoted as probability to compare `p_value` against)
    pub p_cl: f64,
}

impl Display for BKMediationAnalysisStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = format!("\tStep {}\n", self.step_number)
            + SMALL_TEXT_BREAK
            + &format!("\tIntercept (Alpha): {}\n", self.alpha)
            + &format!("\tSlope (Beta): {}\n", self.beta)
            + &format!("\tStandard Error of Slope Coefficient (SE beta): {}\n", self.beta_se)
            + SMALL_TEXT_BREAK
            + &format!("\tDegrees of Freedom: {} (DDoF: {})\n", self.dof, self.ddof)
            + &format!("\tReject Null Hypothesis: {}\n", self.reject_h0)
            + &format!("\tt-statistic: {}\n", self.t_score)
            + &format!("\tp-value: {}\n", self.p_value)
            + &format!("\tConfidence Level: {}\n", self.p_cl);

        write!(f, "{}", result)
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Result of the final analysis test.
pub struct BKMediationAnalysisFinalStep {
    /// Intercept (Alpha) of the linear regression
    pub alpha: f64,
    /// Slope (Beta) of the independent variable
    pub indep_beta: f64,
    /// Standard error of the independent variable slope coefficient
    pub indep_beta_se: f64,
    /// Slope (Beta) of the mediator variable
    pub me_beta: f64,
    /// Standard error of the mediator variable slope coefficient
    pub me_beta_se: f64,
    /// Delta degrees of freedom
    pub ddof: usize,
    /// Degrees of freedom used in the test
    pub dof: f64,
    /// Whether to reject the null hypothesis for the independent variable's slope
    pub indep_reject_h0: bool,
    /// t-statistic of the t-test for the independent variable's slope
    pub indep_t_score: f64,
    /// p-value of the t-test for the independent variable's slope
    pub indep_p_value: f64,
    /// Whether to reject the null hypothesis for the mediator variable's slope
    pub me_reject_h0: bool,
    /// t-statistic of the t-test for the mediator variable's slope
    pub me_t_score: f64,
    /// p-value of the t-test for the mediator variable's slope
    pub me_p_value: f64,
    /// Confidence level for t-tests (Quoted as probability to compare `p_value` against)
    pub p_cl: f64,
}

impl Display for BKMediationAnalysisFinalStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = String::from("\tStep 3\n")
            + SMALL_TEXT_BREAK
            + &format!("\tIntercept (Alpha): {}\n", self.alpha)
            + &format!("\tSlope of Independent Variable (Beta): {}\n", self.indep_beta)
            + &format!("\tStandard Error of Independent Variable Slope Coefficient (SE beta): {}\n", self.indep_beta_se)
            + &format!("\tSlope of Mediator Variable (Beta): {}\n", self.me_beta)
            + &format!("\tStandard Error of Mediator Variable Slope Coefficient (SE beta): {}\n", self.me_beta_se)
            + SMALL_TEXT_BREAK
            + &format!("\tDegrees of Freedom: {} (DDoF: {})\n", self.dof, self.ddof)
            + SMALL_TEXT_BREAK
            + &format!("\tReject Null Hypothesis (Independent Variable): {}\n", self.indep_reject_h0)
            + &format!("\tt-statistic (Independent Variable): {}\n", self.indep_t_score)
            + &format!("\tp-value (Independent Variable): {}\n", self.indep_p_value)
            + &format!("\tConfidence Level: {}\n", self.p_cl)
            + &format!("\tReject Null Hypothesis (Mediator Variable): {}\n", self.me_reject_h0)
            + &format!("\tt-statistic (Mediator Variable): {}\n", self.me_t_score)
            + &format!("\tp-value (Mediator Variable): {}\n", self.me_p_value)
            + &format!("\tConfidence Level: {}\n", self.p_cl);
        write!(f, "{}", result)
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BKMediationAnalysisResult {
    /// Result of Step 1 - Verification that the independent variable is a significant predictor of the dependent variable
    pub step_1_result: BKMediationAnalysisStep,
    /// Result of Step 2 - Verification that the dependent variable is a significant predictor of the mediator variable
    pub step_2_result: BKMediationAnalysisStep,
    /// Result of Step 3 - Verification that the mediator variable is a significant predictor of the dependent variable,
    /// and that the strength of the coefficient of the previously significant independent variable in step one is now greatly reduced,
    /// if not rendered nonsignificant
    pub step_3_result: BKMediationAnalysisFinalStep,
    /// Whether the mediator `me` should be used alongside `x` in the linear regression that predicts `y`.
    pub is_mediated: bool,
}

impl Display for BKMediationAnalysisResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = LARGE_TEXT_BREAK.to_owned()
            + "Baron and Kenny's Mediation Analysis Result\n"
            + LARGE_TEXT_BREAK
            + &format!("{}", self.step_1_result.to_string())
            + LARGE_TEXT_BREAK
            + &format!("{}", self.step_2_result.to_string())
            + LARGE_TEXT_BREAK
            + &format!("{}", self.step_3_result.to_string())
            + LARGE_TEXT_BREAK
            + &format!("The effect of mediation is significant: {}\n", self.is_mediated)
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Baron and Kenny's mediation analysis. This analysis technique helps to determine whether the mediator variable should be introduced into
/// a linear regression model that will provide statistically more significant result than just a linear regression model between
/// the independent and dependent variables.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Mediation_(statistics)#Baron_and_Kenny's_(1986)_steps_for_mediation_analysis>
/// - Original Source: <https://doi.org/10.1037%2F0022-3514.51.6.1173>
/// 
/// # Examples
/// ```rust
/// use ndarray::Array1;
/// use digifi::statistics::{ConfidenceLevel, BKMediationAnalysisResult, BaronKennyMeriationAnalysis};
/// 
/// #[cfg(feature = "sample_data")]
/// fn test_bk_analysis() -> () {
///     use digifi::utilities::SampleData;
/// 
///     // Sample data
///     let sample: SampleData = SampleData::CAPM; 
///     let (_, mut sample_data) = sample.load_sample_data();
///     let smb: Array1<f64> = sample_data.remove("SMB").unwrap();
///     let market_premium: Array1<f64> = sample_data.get("Market").unwrap() - sample_data.get("RF").unwrap();
///     let asset_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
/// 
///     // Test whether market premium is a mediator variable in the model where smb is used to predict asset premium
///     let bk_analysis: BaronKennyMeriationAnalysis = BaronKennyMeriationAnalysis::new(Some(ConfidenceLevel::Five));
///     let bkr: BKMediationAnalysisResult = bk_analysis.run(&asset_premium, &smb, &market_premium).unwrap();
///     assert_eq!(bkr.is_mediated, false);
/// }
/// ```
pub struct BaronKennyMeriationAnalysis {
    /// Confidence level for significance testing
    t_test_confidence_level: Option<ConfidenceLevel>,
}

impl BaronKennyMeriationAnalysis {
    /// Creates a new instance of `BaronKennyMeriationAnalysis`.
    /// 
    /// # Input
    /// - `cl`: Confidence level for significance testing
    pub fn new(cl: Option<ConfidenceLevel>) -> Self {
        Self { t_test_confidence_level: cl, }
    }

    fn validate(&self, y: &Array1<f64>, x: &Array1<f64>, me: &Array1<f64>) -> Result<(), DigiFiError> {
        compare_len(&y.iter(), &x.iter(), "y", "x")?;
        compare_len(&y.iter(), &me.iter(), "y", "me")
    }

    fn feature_name_x(&self) -> &str { "x" }

    fn feature_name_mediator(&self) -> &str { "Mediator" }

    fn linear_regression(&self, y: &Array1<f64>, fc: &FeatureCollection) -> Result<LinearRegressionResult, DigiFiError> {
        let mut settings: LinearRegressionSettings = LinearRegressionSettings::disable_all();
        settings.enable_se = true;
        settings.enable_t_test = true;
        settings.t_test_cl = self.t_test_confidence_level;
        LinearRegressionAnalysis::new(settings).run(&fc, y)
    }

    fn unstack_linear_regression_step(&self, fc: &FeatureCollection, result: LinearRegressionResult, step_number: usize) -> Result<BKMediationAnalysisStep, DigiFiError> {
        let beta: &LinearRegressionFeatureResult = &result.coefficients[fc.get_feature_index(self.feature_name_x())?];
        Ok(BKMediationAnalysisStep {
            step_number, alpha: result.intercept.unwrap_or(f64::NAN), beta: beta.coefficient, beta_se: beta.standard_error.unwrap_or(f64::NAN),
            ddof: fc.ddof(), dof: beta.t_test_dof.unwrap_or(f64::NAN), reject_h0: beta.t_test_reject_h0.unwrap_or(false),
            t_score: beta.t_test_t_score.unwrap_or(f64::NAN), p_value: beta.t_test_p_value.unwrap_or(f64::NAN),
            p_cl: beta.t_test_cl.unwrap_or(f64::NAN),
        })
    }

    fn unstack_linear_regression_final_step(&self, fc: &FeatureCollection, result: LinearRegressionResult) -> Result<BKMediationAnalysisFinalStep, DigiFiError> {
        let beta: &LinearRegressionFeatureResult = &result.coefficients[fc.get_feature_index(self.feature_name_x())?];
        let me: &LinearRegressionFeatureResult = &result.coefficients[fc.get_feature_index(self.feature_name_mediator())?];
        Ok(BKMediationAnalysisFinalStep {
            alpha: result.intercept.unwrap_or(f64::NAN), indep_beta: beta.coefficient, indep_beta_se: beta.standard_error.unwrap_or(f64::NAN),
            me_beta: me.coefficient, me_beta_se: me.standard_error.unwrap_or(f64::NAN), ddof: fc.ddof(),
            dof: beta.t_test_dof.unwrap_or(f64::NAN), indep_reject_h0: beta.t_test_reject_h0.unwrap_or(false),
            indep_t_score: beta.t_test_t_score.unwrap_or(f64::NAN), indep_p_value: beta.t_test_p_value.unwrap_or(f64::NAN),
            me_reject_h0: me.t_test_reject_h0.unwrap_or(false), me_t_score: me.t_test_t_score.unwrap_or(f64::NAN),
            me_p_value: me.t_test_p_value.unwrap_or(f64::NAN), p_cl: beta.t_test_cl.unwrap_or(f64::NAN),
        })
    }

    /// Runs the Baron and Kenny's mediation analysis.
    /// 
    /// # Input
    /// - `y`: Dependent variable
    /// - `x`: Independent variable
    /// - `me`: Mediator variable
    /// 
    /// # Ouput
    /// - Result of the mediation analysis with intermediate results for every step
    pub fn run(&self, y: &Array1<f64>, x: &Array1<f64>, me: &Array1<f64>) -> Result<BKMediationAnalysisResult, DigiFiError> {
        self.validate(y, x, me)?;
        // Standardise data
        let mut fc: FeatureCollection = FeatureCollection::new();
        fc.add_constant = true;
        fc.add_feature(x.iter(), self.feature_name_x())?;
        // Analysis steps
        // Step 1 - Verification that the independent variable is a significant predictor of the dependent variable
        let lr_result: LinearRegressionResult = self.linear_regression(y, &fc)?;
        let step_1_result: BKMediationAnalysisStep = self.unstack_linear_regression_step(&fc, lr_result, 1)?;
        // Step 2 - Verification that the dependent variable is a significant predictor of the mediator variable
        let lr_result: LinearRegressionResult = self.linear_regression(me, &fc)?;
        let step_2_result: BKMediationAnalysisStep = self.unstack_linear_regression_step(&fc, lr_result, 2)?;
        // Step 3 - Verification that the mediator variable is a significant predictor of the dependent variable,
        // and that the strength of the coefficient of the previously significant independent variable in step one is now greatly reduced,
        // if not rendered nonsignificant
        fc.add_feature(me.iter(), self.feature_name_mediator())?;
        let lr_result: LinearRegressionResult = self.linear_regression(y, &fc)?;
        let step_3_result: BKMediationAnalysisFinalStep = self.unstack_linear_regression_final_step(&fc, lr_result)?;
        // Final result
        let is_mediated: bool = step_1_result.reject_h0 && step_2_result.reject_h0 && step_3_result.me_reject_h0 && (step_3_result.indep_beta.abs() < step_1_result.beta.abs());
        Ok(BKMediationAnalysisResult { step_1_result, step_2_result: step_2_result, step_3_result: step_3_result, is_mediated, })
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Result of Sobel test that determines the statistical significance of indirect effect.
pub struct SobelTestResult {
    /// Result of the Baron and Kenny's mediation analysis
    pub bk_mediation_analysis_result: BKMediationAnalysisResult,
    /// Sobel test result
    pub reject_h0: bool,
    /// Statistic of the Sobel test
    pub t_score: f64,
    /// p-value of the Sobel test
    pub p_value: f64,
    /// Confidence level of the Sobel test
    pub p_cl: f64,
}

impl Display for SobelTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = self.bk_mediation_analysis_result.to_string()
            + "\n\n"
            + LARGE_TEXT_BREAK
            + "Sobel Mediation Test\n"
            + LARGE_TEXT_BREAK
            + &format!("Reject Null Hypothesis: {}\n", self.reject_h0)
            + &format!("t-statistic: {}\n", self.t_score)
            + &format!("p-value: {}\n", self.p_value)
            + &format!("Confidence Level: {}\n", self.p_cl)
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}


/// Sobel test that determines the statistical significance of indirect effect.
/// 
/// # Input
/// - `y`: Dependent variable
/// - `x`: Independent variable
/// - `me`: Mediator variable
/// - `cl`: Confidence level for significance testing
/// 
/// # Output
/// - Result of the Sobel test, along with result for Baron and Kenny's mediation analysis
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Mediation_(statistics)#Sobel's_test>
/// - Original Source: <https://doi.org/10.2307%2F270723>
/// 
/// # Examples
/// 
/// ```rust
/// use ndarray::Array1;
/// use digifi::statistics::{ConfidenceLevel, SobelTestResult, sobel_test};
/// 
/// #[cfg(feature = "sample_data")]
/// fn test_sobel_test() -> () {
///     use digifi::utilities::SampleData;
/// 
///     // Sample data
///     let sample: SampleData = SampleData::CAPM; 
///     let (_, mut sample_data) = sample.load_sample_data();
///     let smb: Array1<f64> = sample_data.remove("SMB").unwrap();
///     let market_premium: Array1<f64> = sample_data.get("Market").unwrap() - sample_data.get("RF").unwrap();
///     let asset_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
/// 
///     // Test whether market premium is a mediator variable in the model where smb is used to predict asset premium
///     let test_result: SobelTestResult = sobel_test(&asset_premium, &smb, &market_premium, Some(ConfidenceLevel::Five)).unwrap();
///     assert_eq!(test_result.reject_h0, true);
/// }
/// ```
pub fn sobel_test(y: &Array1<f64>, x: &Array1<f64>, me: &Array1<f64>, cl: Option<ConfidenceLevel>) -> Result<SobelTestResult, DigiFiError> {
    // Runs Baron Kenny mediation analysis to get linear regression models' results
    let bk_analysis: BaronKennyMeriationAnalysis = BaronKennyMeriationAnalysis::new(cl);
    let bkr: BKMediationAnalysisResult = bk_analysis.run(y, x, me)?;
    // Sobol test
    // Note: crossed multipliers alpha^{2} * se^{2}_{beta} + beta^{2} * se^{2}_{alpha}
    let se: f64 = (bkr.step_2_result.beta.powi(2) * bkr.step_3_result.me_beta_se.powi(2) + bkr.step_3_result.me_beta.powi(2) * bkr.step_2_result.beta_se.powi(2)).sqrt();
    let t_score: f64 = (bkr.step_1_result.beta - bkr.step_3_result.indep_beta) / se;
    // Obtain confidence interval value
    let p_value: f64 = 1.0 - NormalDistribution::build(0.0, 1.0)?.cdf(t_score)?;
    let p_cl: f64 = match cl { Some(v) => v.get_p(), None => ConfidenceLevel::default().get_p() };
    let reject_h0: bool = if p_value < p_cl { true } else { false };
    Ok(SobelTestResult { bk_mediation_analysis_result: bkr, reject_h0, t_score, p_value, p_cl, })
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MJYAnalysisStep {
    step_number: usize,
    /// Intercept (Alpha) of the linear regression
    pub alpha: f64,
    /// Slope of variable `x` in the linear regression
    pub beta_x: f64,
    /// Slope of moderation variable in the linear regression
    pub beta_mo: f64,
    /// Slope of the moderation effect (i.e., `x * mo`) in the linear regression
    pub beta_x_mo: f64,
}

impl Display for MJYAnalysisStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = format!("\tStep {}\n", self.step_number)
            + SMALL_TEXT_BREAK
            + &format!("\tIntercept (Alpha): {}\n", self.alpha)
            + &format!("\tSlope of Variable X (Beta X): {}\n", self.beta_x)
            + &format!("\tSlope of Moderation Variable (Beta Mo): {}\n", self.beta_mo)
            + &format!("\tSlope of Moderation Effect (Beta X Mo): {}\n", self.beta_x_mo);
        write!(f, "{}", result)
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MJYAnalysisFinalStep {
    /// Intercept (Alpha) of the linear regression
    pub alpha: f64,
    /// Slope of variable `x` in the linear regression
    pub beta_x: f64,
    /// Slope of the moderation variable in the linear regression
    pub beta_mo: f64,
    /// Slope of the moderation effect (i.e., `x * mo`) in the linear regression
    pub beta_x_mo: f64,
    /// Slope of the mediation variable in thew linear regression
    pub beta_me: f64,
    ///Slope of the moderated mediation (i.e., `me * mo`)
    pub beta_me_mo: f64,
}

impl Display for MJYAnalysisFinalStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = "\tStep 3\n".to_owned()
            + SMALL_TEXT_BREAK
            + &format!("\tIntercept (Alpha): {}\n", self.alpha)
            + &format!("\tSlope of Variable X (Beta X): {}\n", self.beta_x)
            + &format!("\tSlope of Moderation Variable (Beta Mo): {}\n", self.beta_mo)
            + &format!("\tSlope of Moderation Effect (Beta X Mo): {}\n", self.beta_x_mo)
            + &format!("\tSlope of Mediation Variable (Beta Me): {}\n", self.beta_me)
            + &format!("\tSlope of Moderated Mediation Effect (Beta Me Mo): {}\n", self.beta_me_mo);
        write!(f, "{}", result)
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MJYAnalysisResult {
    /// Result of Step 1 - Moderation of the relationship between the independent variable (X) and the dependent variable (Y),
    /// also called the overall treatment effect
    pub step_1_result: MJYAnalysisStep,
    /// Result of Step 2 - Moderation of the relationship between the independent variable and the mediator
    pub step_2_result: MJYAnalysisStep,
    /// Result of Step 3 - Moderation of both the relationship between the independent and dependent variables
    /// and the relationship between the mediator and the dependent variable
    pub step_3_result: MJYAnalysisFinalStep,
}

impl Display for MJYAnalysisResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = LARGE_TEXT_BREAK.to_owned()
            + "Muller, Judd and Yzerbyt's Analysis Result\n"
            + LARGE_TEXT_BREAK
            + &format!("{}", self.step_1_result.to_string())
            + LARGE_TEXT_BREAK
            + &format!("{}", self.step_2_result.to_string())
            + LARGE_TEXT_BREAK
            + &format!("{}", self.step_3_result.to_string())
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Muller, Judd, and Yzerbyt analysis that outlines three fundamental models that underlie both moderated mediation and mediated moderation.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Mediation_(statistics)#Regression_equations_for_moderated_mediation_and_mediated_moderation>
/// - Original Source: <https://doi.org/10.1037%2F0022-3514.89.6.852>
/// 
/// # Examples
/// ```rust
/// use ndarray::Array1;
/// use digifi::statistics::{MJYAnalysisResult, MullerJuddYzerbytAnalysis};
/// 
/// #[cfg(feature = "sample_data")]
/// fn test_bk_analysis() -> () {
///     use digifi::utilities::SampleData;
/// 
///     // Sample data
///     let sample: SampleData = SampleData::CAPM; 
///     let (_, mut sample_data) = sample.load_sample_data();
///     let smb: Array1<f64> = sample_data.remove("SMB").unwrap();
///     let hml: Array1<f64> = sample_data.remove("HML").unwrap();
///     let market_premium: Array1<f64> = sample_data.get("Market").unwrap() - sample_data.get("RF").unwrap();
///     let asset_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
/// 
///     // Run models needed to test for moderated mediation or mediated moderation
///     let test_result: MJYAnalysisResult = MullerJuddYzerbytAnalysis.run(&asset_premium, &smb, &market_premium, &hml).unwrap();
///     println!("{test_result}");
/// }
/// ```
pub struct MullerJuddYzerbytAnalysis;

impl MullerJuddYzerbytAnalysis {
    fn validate(&self, y: &Array1<f64>, x: &Array1<f64>, me: &Array1<f64>, mo: &Array1<f64>) -> Result<(), DigiFiError> {
        compare_len(&y.iter(), &x.iter(), "y", "x")?;
        compare_len(&y.iter(), &me.iter(), "y", "me")?;
        compare_len(&y.iter(), &mo.iter(), "y", "mo")
    }

    fn feature_name_x(&self) -> &str { "x" }

    fn feature_name_mediator(&self) -> &str { "Mediator" }

    fn feature_name_x_mo(&self) -> &str { "XMo" }

    fn feature_name_moderator(&self) -> &str { "Moderator" }

    fn feature_name_me_mo(&self) -> &str { "MeMo" }

    fn unstack_linear_regression_step(&self, fc: &FeatureCollection, result: LinearRegressionResult, step_number: usize) -> Result<MJYAnalysisStep, DigiFiError> {
        let beta_x: &LinearRegressionFeatureResult = &result.coefficients[fc.get_feature_index(self.feature_name_x())?];
        let beta_mo: &LinearRegressionFeatureResult = &result.coefficients[fc.get_feature_index(self.feature_name_moderator())?];
        let beta_x_mo: &LinearRegressionFeatureResult = &result.coefficients[fc.get_feature_index(self.feature_name_x_mo())?];
        Ok(MJYAnalysisStep {
            step_number, alpha: result.intercept.unwrap_or(f64::NAN), beta_x: beta_x.coefficient, beta_mo: beta_mo.coefficient,
            beta_x_mo: beta_x_mo.coefficient,
        })
    }

    fn unstack_linear_regression_final_step(&self, fc: &FeatureCollection, result: LinearRegressionResult) -> Result<MJYAnalysisFinalStep, DigiFiError> {
        let beta_x: &LinearRegressionFeatureResult = &result.coefficients[fc.get_feature_index(self.feature_name_x())?];
        let beta_mo: &LinearRegressionFeatureResult = &result.coefficients[fc.get_feature_index(self.feature_name_moderator())?];
        let beta_x_mo: &LinearRegressionFeatureResult = &result.coefficients[fc.get_feature_index(self.feature_name_x_mo())?];
        let beta_me: &LinearRegressionFeatureResult = &result.coefficients[fc.get_feature_index(self.feature_name_mediator())?];
        let beta_me_mo: &LinearRegressionFeatureResult = &result.coefficients[fc.get_feature_index(self.feature_name_me_mo())?];
        Ok(MJYAnalysisFinalStep {
            alpha: result.intercept.unwrap_or(f64::NAN), beta_x: beta_x.coefficient, beta_mo: beta_mo.coefficient,
            beta_x_mo: beta_x_mo.coefficient, beta_me: beta_me.coefficient, beta_me_mo: beta_me_mo.coefficient,
        })
    }

    /// Runs the Muller, Judd and Yzerbyt's analysis.
    /// 
    /// # Input
    /// - `y`: Dependent variable
    /// - `x`: Independent variable
    /// - `me`: Mediator variable
    /// - `mo`: Moderator variable
    /// 
    /// # Ouput
    /// - Result of models needed to test for moderated mediation or mediated moderation
    pub fn run(&self, y: &Array1<f64>, x: &Array1<f64>, me: &Array1<f64>, mo: &Array1<f64>) -> Result<MJYAnalysisResult, DigiFiError> {
        self.validate(y, x, me, mo)?;
        // Standardise data
        let mut fc: FeatureCollection = FeatureCollection::new();
        fc.add_constant = true;
        fc.add_feature(x.iter(), self.feature_name_x())?;
        fc.add_feature(mo.iter(), self.feature_name_moderator())?;
        fc.add_feature((x * mo).into_iter(), self.feature_name_x_mo())?;
        // Analysis steps
        // Step 1 - Moderation of the relationship between the independent variable (X) and the dependent variable (Y),
        // also called the overall treatment effect
        let lr_result: LinearRegressionResult = LinearRegressionAnalysis::new(LinearRegressionSettings::disable_all()).run(&fc, y)?;
        let step_1_result: MJYAnalysisStep = self.unstack_linear_regression_step(&fc, lr_result, 1)?;
        // Step 2 - Moderation of the relationship between the independent variable and the mediator
        let lr_result: LinearRegressionResult = LinearRegressionAnalysis::new(LinearRegressionSettings::disable_all()).run(&fc, me)?;
        let step_2_result: MJYAnalysisStep = self.unstack_linear_regression_step(&fc, lr_result, 2)?;
        // Step 3 - Moderation of both the relationship between the independent and dependent variables
        // and the relationship between the mediator and the dependent variable
        fc.add_feature(me.iter(), self.feature_name_mediator())?;
        fc.add_feature((me * mo).into_iter(), self.feature_name_me_mo())?;
        let lr_result: LinearRegressionResult = LinearRegressionAnalysis::new(LinearRegressionSettings::disable_all()).run(&fc, y)?;
        let step_3_result: MJYAnalysisFinalStep = self.unstack_linear_regression_final_step(&fc, lr_result)?;
        // Final result
        Ok(MJYAnalysisResult { step_1_result, step_2_result: step_2_result, step_3_result: step_3_result, })
    }
}


#[cfg(all(test, feature = "sample_data"))]
mod tests {
    use ndarray::Array1;
    use crate::statistics::stat_tests::ConfidenceLevel;

    #[test]
    fn unit_test_baron_kenny_mediation_analysis() -> () {
        use crate::utilities::sample_data::SampleData;
        use crate::statistics::mediation_analysis::{BKMediationAnalysisResult, BaronKennyMeriationAnalysis};
        // Sample data
        let sample: SampleData = SampleData::CAPM; 
        let (_, mut sample_data) = sample.load_sample_data();
        let smb: Array1<f64> = sample_data.remove("SMB").unwrap();
        let market_premium: Array1<f64> = sample_data.get("Market").unwrap() - sample_data.get("RF").unwrap();
        let asset_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
        // Test whether market premium is a mediator variable in the model where smb is used to predict asset premium
        let bk_analysis: BaronKennyMeriationAnalysis = BaronKennyMeriationAnalysis::new(Some(ConfidenceLevel::Five));
        let bkr: BKMediationAnalysisResult = bk_analysis.run(&asset_premium, &smb, &market_premium).unwrap();
        assert_eq!(bkr.is_mediated, false);
    }

    #[test]
    fn unit_test_sobel_test() -> () {
        use crate::utilities::sample_data::SampleData;
        use crate::statistics::mediation_analysis::{SobelTestResult, sobel_test};
        // Sample data
        let sample: SampleData = SampleData::CAPM; 
        let (_, mut sample_data) = sample.load_sample_data();
        let smb: Array1<f64> = sample_data.remove("SMB").unwrap();
        let market_premium: Array1<f64> = sample_data.get("Market").unwrap() - sample_data.get("RF").unwrap();
        let asset_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
        // Test whether market premium is a mediator variable in the model where smb is used to predict asset premium
        let test_result: SobelTestResult = sobel_test(&asset_premium, &smb, &market_premium, Some(ConfidenceLevel::Five)).unwrap();
        assert_eq!(test_result.reject_h0, true);
    }

    #[test]
    fn unit_test_mjy_analysis() -> () {
        use crate::utilities::sample_data::SampleData;
        use crate::statistics::mediation_analysis::{MJYAnalysisResult, MullerJuddYzerbytAnalysis};
        // Sample data
        let sample: SampleData = SampleData::CAPM; 
        let (_, mut sample_data) = sample.load_sample_data();
        let smb: Array1<f64> = sample_data.remove("SMB").unwrap();
        let hml: Array1<f64> = sample_data.remove("HML").unwrap();
        let market_premium: Array1<f64> = sample_data.get("Market").unwrap() - sample_data.get("RF").unwrap();
        let asset_premium: Array1<f64> = sample_data.remove("Stock Returns").unwrap();
        // Run models needed to test for moderated mediation or mediated moderation
        let test_result: MJYAnalysisResult = MullerJuddYzerbytAnalysis.run(&asset_premium, &smb, &market_premium, &hml).unwrap();
        println!("{test_result}");
    }
}