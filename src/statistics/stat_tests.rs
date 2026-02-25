// Re-Exports
pub use self::adf_test::{ADFType, ADFResult, adf};
pub use self::t_test::{TTestResult, TTestTwoSampleCase, t_test_two_sample, t_test_lr};
pub use self::f_test::{nested_f_statistic, FTestResult, f_test_anova};
pub use self::granger_causality::{
    GrangerCausalityTestType, GrangerCausalityRejectReason, GrangerCausalityResult, GrangerCausalitySettings,
    granger_causality_test, simple_granger_causality_test,
};

mod adf_test;
mod f_test;
mod t_test;
mod granger_causality;


use std::fmt::Display;
use ndarray::{Array1, Array2};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError};
use crate::utilities::{LARGE_TEXT_BREAK, compare_len};
use crate::statistics::linear_regression;


#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Confidence interval for different tests.
pub enum ConfidenceLevel {
    /// 0.1 Confidence level
    Ten,
    /// 0.05 confidence level
    #[default]
    Five,
    /// 0.025 confidence level
    TwoHalf,
    /// 0.01 confidence level
    One,
}

impl ConfidenceLevel {
    pub fn get_p(&self) -> f64 {
        match self {
            Self::Ten => 0.1,
            Self::Five => 0.05,
            Self::TwoHalf => 0.25,
            Self::One => 0.01,
        }
    }
}

impl Display for ConfidenceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get_p().to_string())
    }
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