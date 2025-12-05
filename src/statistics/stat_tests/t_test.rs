use std::fmt::Display;
use ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::LARGE_TEXT_BREAK;
use crate::statistics::{
    ProbabilityDistribution, se_lr_coefficient,
    continuous_distributions::StudentsTDistribution,
    stat_tests::ConfidenceLevel,
};


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Result of the t-test.
pub struct TTestResult {
    /// Whether to reject the null hypothesis
    pub reject_h0: bool,
    /// t-statistic of the t-test
    pub t_score: f64,
    /// Degrees of freedom used in the test
    pub dof: f64,
    /// p-value of the t-test
    pub p_value: f64,
    /// Confidence level for t-test (Quoted as probability to compare `p_value` against)
    pub p_cl: f64,
}

impl Display for TTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = LARGE_TEXT_BREAK.to_owned()
            + "t-Test Result\n"
            + LARGE_TEXT_BREAK
            + &format!("Degrees of Freedom: {}\n", self.dof)
            + &format!("Reject Null Hypothesis: {}\n", self.reject_h0)
            + &format!("t-Test t-statistic: {}\n", self.t_score)
            + &format!("t-Test p-value: {}\n", self.p_value)
            + &format!("Confidence Level: {}\n", self.p_cl)
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}


#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Variant of the two sample t-test.
/// 
/// Note: `EqualVariance` assumes that the samples are drawn from the distributions that have the same variance, but it does not explicitly check for that to hold.
pub enum TTestTwoSampleCase {
    #[default]
    /// Equal sample sizes and variance - Assumed that the two distributions have the same variance
    EqualVariance,
    /// Equal or unequal sizes, similar variance (`0.5 < s^{2}_{1} / s^{2}_{2} < 2`)
    SimilarVariance,
    /// Equal or unequal sample sizes, unequal variances (`s_{1} > 2s_{2}` or `2s_{1} < s_{2}`) - Welch's t-test
    UnequalVariance,
}

impl TTestTwoSampleCase {

    /// Returns the ratio of unbiased estimators of standard deviations `s_{1} / s_{2}`.
    fn unbiased_std_ratio(sample_1: &Array1<f64>, sample_2: &Array1<f64>) -> f64 {
        sample_1.std(1.0) / sample_2.std(1.0)
    }

    /// Selects two sample t-test that best fits the data.
    /// 
    /// # Input
    /// - `sample_1`: First sample
    /// - `sample_2`: Second sample
    /// 
    /// # Output
    /// - Two sample t-test case that best fits the data
    pub fn select_case(sample_1: &Array1<f64>, sample_2: &Array1<f64>) -> Self {
        if sample_1.len() == sample_2.len() {
            TTestTwoSampleCase::EqualVariance
        } else {
            let unbiased_std_ratio: f64 = TTestTwoSampleCase::unbiased_std_ratio(sample_1, sample_2);
            if 0.5 <= unbiased_std_ratio && unbiased_std_ratio <= 2.0 {
                TTestTwoSampleCase::SimilarVariance
            } else {
                TTestTwoSampleCase::UnequalVariance
            }
        }
    }

    /// Validates the t-test case.
    fn validate(&self, sample_1: &Array1<f64>, sample_2: &Array1<f64>) -> Result<(), DigiFiError> {
        if sample_1.len() < 1 || sample_2.len() < 1 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "Both samples must contain at least one data point.".to_owned(),
            });
        }
        let unbiased_std_ratio: f64 = TTestTwoSampleCase::unbiased_std_ratio(sample_1, sample_2);
        match self {
            TTestTwoSampleCase::EqualVariance => {
                if sample_1.len() != sample_2.len() {
                    return Err(DigiFiError::UnmatchingLength { array_1: "sample_1".to_owned(), array_2: "sample_2".to_owned(), })
                }
            },
            TTestTwoSampleCase::SimilarVariance => {
                if unbiased_std_ratio < 0.5 || 2.0 < unbiased_std_ratio {
                    return Err(DigiFiError::ParameterConstraint {
                        title: Self::error_title(),
                        constraint: "The unbiased estimators of variance of `sample_1` and `sample_2` have to satisfy the unequality `0.5 < s^{2}_{1} / s^{2}_{2} < 2` to use `SimilarVariance` two sample t-test case.".to_owned(),
                    })
                }
            },
            TTestTwoSampleCase::UnequalVariance => {
                if 0.5 <= unbiased_std_ratio && unbiased_std_ratio <= 2.0 {
                    return Err(DigiFiError::ParameterConstraint {
                        title: Self::error_title(),
                        constraint: "The unbiased estimators of variance of `sample_1` and `sample_2` have to satisfy either the unequality `s_{1} > 2s_{2}` or `2s_{1} < s_{2}` to use `UnequalVariance` two sample t-test case.".to_owned(),
                    })
                }
            },
        }
        Ok(())
    }
}

impl ErrorTitle for TTestTwoSampleCase {
    fn error_title() -> String {
        String::from("T-test (Two Sample) Case")
    }
}

impl Display for TTestTwoSampleCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TTestTwoSampleCase::EqualVariance => write!(f, "Equal Variance"),
            TTestTwoSampleCase::SimilarVariance => write!(f, "Similar Variance"),
            TTestTwoSampleCase::UnequalVariance => write!(f, "Unequal Variance"),
        }
    }
}


/// Two sample t-test. It is used to test whether two samples have an equal mean. The null hypothesis is that the populations of both samples have equal mean.
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
/// use digifi::statistics::{ConfidenceLevel, TTestResult, TTestTwoSampleCase, t_test_two_sample};
///
/// #[cfg(feature = "sample_data")]
/// fn test_t_test_two_sample() -> () {
///     use digifi::utilities::SampleData;
///     
///     let sample: SampleData = SampleData::Portfolio; 
///     let (_, mut sample_data) = sample.load_sample_data();
///     let jpm: Array1<f64> = sample_data.remove("JPM").unwrap();
///     let gs: Array1<f64> = sample_data.remove("GS").unwrap();
/// 
///     // Test whether jpm and gs have equal mean
///     let test_result: TTestResult = t_test_two_sample(&jpm, &gs, Some(ConfidenceLevel::Five), Some(TTestTwoSampleCase::EqualVariance)).unwrap();
///     assert_eq!(test_result.reject_h0, true);
/// }
/// ```
pub fn t_test_two_sample(sample_1: &Array1<f64>, sample_2: &Array1<f64>, cl: Option<ConfidenceLevel>, case: Option<TTestTwoSampleCase>) -> Result<TTestResult, DigiFiError> {
    let error_title: String = String::from("T-test (Two Sample)");
    // Select the t-test case
    let case: TTestTwoSampleCase = match case {
        Some(v) => v,
        None => TTestTwoSampleCase::select_case(sample_1, sample_2),
    };
    case.validate(sample_1, sample_2)?;
    // Compute t-statistic and degrees of freedom
    let sample_1_len: f64 = sample_1.len() as f64;
    let sample_2_len: f64 = sample_2.len() as f64;
    let mean_1: f64 = sample_1.mean().ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: "series_1".to_owned(), })?;
    let mean_2: f64 = sample_2.mean().ok_or(DigiFiError::MeanCalculation { title: error_title, series: "series_2".to_owned(), })?;
    let unbiased_var_1: f64 = sample_1.var(1.0);
    let unbiased_var_2: f64 = sample_2.var(1.0);
    let (t_score, dof) = match case {
        TTestTwoSampleCase::EqualVariance => {
            let dof: f64 = sample_1_len;
            let pooled_std: f64 = ((unbiased_var_1 + unbiased_var_2) / 2.0).sqrt();
            let t_score: f64 = (mean_1 - mean_2) / (pooled_std * (2.0/dof).sqrt());
            (t_score, dof)
        },
        TTestTwoSampleCase::SimilarVariance => {
            let dof: f64 = sample_1_len + sample_2_len - 2.0;
            let pooled_std: f64 = (((sample_1_len - 1.0) * unbiased_var_1 + (sample_2_len - 1.0) * unbiased_var_2) / dof).sqrt();
            let t_score: f64 = (mean_1 - mean_2) / (pooled_std * (1.0/sample_1_len + 1.0/sample_2_len).sqrt());
            (t_score, dof)
        },
        TTestTwoSampleCase::UnequalVariance => {
            let scaled_unbiased_var_1: f64 = unbiased_var_1 / sample_1_len;
            let scaled_unbiased_var_2: f64 = unbiased_var_2 / sample_2_len;
            let dof: f64 = (scaled_unbiased_var_1 + scaled_unbiased_var_2).powi(2) / (scaled_unbiased_var_1.powi(2)/(sample_1_len - 1.0) + scaled_unbiased_var_2.powi(2)/(sample_2_len - 1.0));
            let pooled_std: f64 = (scaled_unbiased_var_1 + scaled_unbiased_var_2).sqrt();
            let t_score: f64 = (mean_1 - mean_2) / pooled_std;
            (t_score, dof)
        },
    };
    // Obtain confidence interval value
    let p_value: f64 = 1.0 - StudentsTDistribution::build(dof)?.cdf(t_score)?;
    let p_cl: f64 = match cl { Some(v) => 1.0 - v.get_p(), None => 1.0 - ConfidenceLevel::default().get_p() };
    let reject_h0: bool = if p_cl < p_value { true } else { false };
    Ok(TTestResult { reject_h0, t_score, dof, p_value, p_cl })
}


/// T-test for the coefficient of a regression model. It is used to test whether an assumption about the value of the coefficient is
/// supported by the empirical data. The null hypothesis is that the estimated coefficient `beta` is equal to the assumed coefficient `beta_0`.
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
/// fn test_t_test_lr() -> () {
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
///     let capm: CAPM = CAPM::build(market_return.clone(), rf.clone(), capm_type, solution_type).unwrap();
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
    let dof: f64 = y.len()
        .checked_sub(ddof)
        .ok_or(DigiFiError::Other {
            title: "T-test (Linear Regression)".to_owned(),
            details: "There are fewer data points in `y` array than `ddof`.".to_owned(),
        })? as f64;
    let p_value: f64 = 1.0 - StudentsTDistribution::build(dof)?.cdf(t_score)?;
    let p_cl: f64 = match cl { Some(v) => v.get_p(), None => ConfidenceLevel::default().get_p() };
    let reject_h0: bool = if p_value < p_cl { true } else { false };
    Ok(TTestResult { reject_h0, t_score, dof, p_value, p_cl })
}


#[cfg(test)]
mod tests {
    #[cfg(all(test, feature = "sample_data"))]
    #[test]
    fn unit_test_t_test_two_sample() -> () {
        use ndarray::Array1;
        use crate::utilities::sample_data::SampleData;
        use crate::statistics::stat_tests::{ConfidenceLevel, TTestResult, TTestTwoSampleCase, t_test_two_sample};
        let sample: SampleData = SampleData::Portfolio; 
        let (_, mut sample_data) = sample.load_sample_data();
        let jpm: Array1<f64> = sample_data.remove("JPM").unwrap();
        let gs: Array1<f64> = sample_data.remove("GS").unwrap();
        // Test whether jpm and gs have equal mean
        let test_result: TTestResult = t_test_two_sample(
            &jpm, &gs, Some(ConfidenceLevel::Five), Some(TTestTwoSampleCase::EqualVariance)
        ).unwrap();
        assert_eq!(test_result.reject_h0, true);
    }

    #[cfg(all(test, feature = "sample_data"))]
    #[test]
    fn unit_test_t_test_lr() -> () {
        use ndarray::Array1;
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
        let capm: CAPM = CAPM::build(market_return.clone(), rf.clone(), capm_type, solution_type).unwrap();
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