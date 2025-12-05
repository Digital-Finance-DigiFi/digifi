use std::fmt::Display;
use ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;
use crate::utilities::LARGE_TEXT_BREAK;
use crate::statistics::{
    ProbabilityDistribution,
    continuous_distributions::FDistribution,
    stat_tests::ConfidenceLevel,
};

/// Computes F-statistic from explained and unexplained variance.
/// 
/// # Input
/// - `explained_var`: Explained variance (e.g., between-group variability)
/// - `unexplained_var`: Unexplained variance (e.g., within-group variability)
/// 
/// # Output
/// - F-statistic
/// 
/// # LaTeX Formula
/// - F = \\frac{\\text{Explained Variance}}{\\text{Unexplained Variance}}
fn f_statistic(explained_var: f64, unexplained_var: f64) -> f64 {
    explained_var / unexplained_var
}


/// Computes F-statistice from the data about two models, where one is nested into another (i.e., the "nested" model can be obtained by
/// constraining parameters in the "parent" model).
/// 
/// # Input
/// - `rss_1`: Residual sum of squares of the "nested" model
/// - `rss_2`: Residual sum of squares of the "parent" model
/// - `n`: Number of data points in the data set
/// - `d_1`: Number of parameters in the "nested" model
/// - `d_2`: Number of parameters in the "parent" model
/// 
/// - # Output
/// - F-statistic
/// 
/// # Errors
/// - Returns an error if the number of parameters in the "nested" model (i.e., `d_1`) is
/// smaller than the number of parameters in the "parent" model (i.e., `d_2`)
/// 
/// # LaTeX Formula
/// - F = \\frac{RSS_{1} - RSS_{2}}{RSS_{2}}\\frac{n - d_{2}}{d_{2} - d_{1}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/F-test#Regression_problems>
/// - Original Source: N/A
pub fn nested_f_statistic(rss_1: f64, rss_2: f64, n: usize, d_1: usize, d_2: usize) -> Result<f64, DigiFiError> {
    if d_2 <= d_1 {
        return Err(DigiFiError::ParameterConstraint {
            title: "F-Statistic".to_owned(),
            constraint: "The number of parameters in nested model (i.e., `d_1`) must be smaller than the number of parameters in large model (i.e., `d_2`).".to_owned(),
        });
    }
    Ok(f_statistic((rss_1 - rss_2) / (d_2 - d_1) as f64, (rss_2) / (n - d_2) as f64))
}


#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Result of the F-test.
pub struct FTestResult {
    /// Whether to reject the null hypothesis
    pub reject_h0: bool,
    /// F-score of the F-test
    pub f_score: f64,
    /// Number of groups
    pub k: f64,
    /// Total data size (Total number of data points in all groups)
    pub n: f64,
    /// Overall mean of the data
    pub overall_mean: f64,
    /// Mean of each sample
    pub group_means: Vec<f64>,
    /// Number of points in each sample
    pub group_ns: Vec<f64>,
    /// Explained variance (Between-group variability)
    pub explained_variance: f64,
    /// Unexplained variance (Within-group variability)
    pub unexplained_variance: f64,
    /// Numerator degrees of freedom used in the test (i.e., `k-1`)
    pub dof_1: f64,
    /// Denominator degrees of freedom used in the test (i.e., `n-k`)
    pub dof_2: f64,
    /// p-value of the F-test
    pub p_value: f64,
    /// Confidence level for F-test (Quoted as probability to compare `p_value` against)
    pub p_cl: f64,
}

impl Display for FTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result: String = LARGE_TEXT_BREAK.to_owned()
            + "F-Test Result\n"
            + LARGE_TEXT_BREAK
            + &format!("Number of Groups: {}\n", self.k)
            + &format!("Total Data Size: {}\n", self.n)
            + &format!("Overall Mean: {}\n", self.overall_mean)
            + &format!("Group Means: {:?}\n", self.group_means)
            + &format!("Group Sizes: {:?}\n", self.group_ns)
            + &format!("Explained Variance: {}\n", self.explained_variance)
            + &format!("Unexplained Variance: {}\n", self.unexplained_variance)
            + &format!("Numerator Degrees of Freedom (d_1): {}\n", self.dof_1)
            + &format!("Denominator Degrees of Freedom (d_2): {}\n", self.dof_2)
            + &format!("Reject Null Hypothesis: {}\n", self.reject_h0)
            + &format!("F-Test F-statistic: {}\n", self.f_score)
            + &format!("F-Test p-value: {}\n", self.p_value)
            + &format!("Confidence Level: {}\n", self.p_cl)
            + LARGE_TEXT_BREAK;
        write!(f, "{}", result)
    }
}


/// F-test of analysis of variance (ANOVA) with a null hypothesis that the means of a given set of normally distributed samples,
/// all having the same standard deviation, are equal.
/// 
/// # Input
/// - `sample_groups`: Different samples that are being tested (i.e., each array is a different sample)
/// - `cl`: Confidence level that should be used for the hypothesis test (Note: if `None` is provided, then defaults to the `ConfidenceLevel` deafult value)
/// 
/// # Output
/// - F-test result with additional information about the test
/// 
/// # Errors
/// - Returns an error if `sample_groups` is empty
/// - Returns an error if a mean of a sample group cannot be computed
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/F-test#One-way_analysis_of_variance>
/// - Original Source: N/A
/// 
/// # Examples
/// 
/// 1. F-test with multiple sample groups
/// 
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ConfidenceLevel, FTestResult, f_test_anova};
/// 
/// // Data taken from `scipy.stats.f_oneway` function's documentation
/// let tillamook: Array1<f64> = Array1::from_vec(vec![0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735, 0.0659, 0.0923, 0.0836]);
/// let newport: Array1<f64> = Array1::from_vec(vec![0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835, 0.0725]);
/// let petersburg: Array1<f64> = Array1::from_vec(vec![0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]);
/// let magadan: Array1<f64> = Array1::from_vec(vec![0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764, 0.0689]);
/// let tvarminne: Array1<f64> = Array1::from_vec(vec![0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]);
/// 
/// // Test whether sample groups have different means
/// // Answer obtained using `scipy.stats.f_oneway` function
/// let test_result: FTestResult = f_test_anova(vec![&tillamook, &newport, &petersburg, &magadan, &tvarminne], Some(ConfidenceLevel::Five)).unwrap();
/// assert_eq!(test_result.overall_mean, 0.0852);
/// assert!((Array1::from_vec(test_result.group_means) - arr1(&[0.0802, 0.07479999999999999, 0.10344285714285714, 0.0780125, 0.09570000000000001])).sum() < TEST_ACCURACY);
/// assert!((test_result.explained_variance - 0.0011299185267857142).abs() < TEST_ACCURACY);
/// assert!((test_result.unexplained_variance - 0.00015867370273109246).abs() < TEST_ACCURACY);
/// assert!((test_result.f_score - 7.121019471642447).abs() < TEST_ACCURACY);
/// assert!((test_result.p_value - 0.00028122423145345444).abs() < TEST_ACCURACY);
/// assert_eq!(test_result.reject_h0, true);
/// ```
/// 
/// 2. F-test with two sample groups
/// 
/// ```rust
/// use ndarray::{Array1, arr1};
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ConfidenceLevel, FTestResult, f_test_anova};
/// 
/// #[cfg(feature = "sample_data")]
/// fn test_f_test_anova() -> () {
///     use digifi::utilities::SampleData;
///
///     let sample: SampleData = SampleData::Portfolio; 
///     let (_, mut sample_data) = sample.load_sample_data();
///     let jpm: Array1<f64> = sample_data.remove("JPM").unwrap();
///     let gs: Array1<f64> = sample_data.remove("GS").unwrap();
/// 
///     // Test whether jpm and gs have equal mean
///     let test_result: FTestResult = f_test_anova(vec![&jpm, &gs], Some(ConfidenceLevel::Five)).unwrap();
///     assert!((test_result.f_score - 10842.154721647501).abs() < TEST_ACCURACY);
///     assert!((test_result.p_value - 0.0).abs() < TEST_ACCURACY);
///     assert_eq!(test_result.reject_h0, true);    
/// }
/// ```
pub fn f_test_anova(sample_groups: Vec<&Array1<f64>>, cl: Option<ConfidenceLevel>) -> Result<FTestResult, DigiFiError> {
    let error_title: String = String::from("F-Test (ANOVA)");
    // Parameter validation
    let k: usize = sample_groups.len();
    if k == 0 {
        return Err(DigiFiError::ParameterConstraint {
            title: error_title,
            constraint: "The `sample_groups` must contain at least one group of samples.".to_owned(),
        });
    }
    // Variables definition
    let mut n: f64 = 0.0;
    let mut group_means: Vec<f64> = Vec::with_capacity(k);
    let mut group_ns: Vec<f64> = Vec::with_capacity(k);
    // Overall mean of the data
    let mut overall_mean: f64 = 0.0;
    for (i, group) in sample_groups.iter().enumerate() {
        let group_mean: f64 = group.mean().ok_or(DigiFiError::MeanCalculation { title: error_title.clone(), series: format!("smple group {}", i) })?;
        let group_n: f64 = group.len() as f64;
        overall_mean += group_mean * group_n;
        n += group_n;
        group_means.push(group_mean);
        group_ns.push(group_n);
    }
    overall_mean = overall_mean / n;
    // Explained and unexplained variance
    let mut explained_variance: f64 = 0.0;
    let mut unexplained_variance: f64 = 0.0;
    let dof_1: f64 = (k - 1) as f64;
    let dof_2: f64 = n - k as f64;
    for i in (0..k).into_iter() {
        explained_variance += group_ns[i] * (group_means[i] - overall_mean).powi(2);
        unexplained_variance += sample_groups[i].iter().fold(0.0, |sum, x| { sum + (x - group_means[i]).powi(2) } );
    }
    explained_variance = explained_variance / dof_1;
    unexplained_variance = unexplained_variance / dof_2;
    // F-test
    let f_score: f64 = f_statistic(explained_variance, unexplained_variance);
    let p_value: f64 = 1.0 - FDistribution::build(dof_1 as usize, dof_2 as usize)?.cdf(f_score)?;
    let p_cl: f64 = match cl { Some(v) => v.get_p(), None => ConfidenceLevel::default().get_p() };
    let reject_h0: bool = if p_value < p_cl { true } else { false };
    Ok(FTestResult {
        reject_h0, f_score, k: k as f64, n, overall_mean, group_means, group_ns, explained_variance, unexplained_variance, dof_1, dof_2, p_value, p_cl, })

}


#[cfg(test)]
mod tests {
    use ndarray::{Array1, arr1};
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_f_test_anova_1() -> () {
        use crate::statistics::stat_tests::{ConfidenceLevel, FTestResult, f_test_anova};
        // Data taken from `scipy.stats.f_oneway` function's documentation
        let tillamook: Array1<f64> = Array1::from_vec(vec![0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735, 0.0659, 0.0923, 0.0836]);
        let newport: Array1<f64> = Array1::from_vec(vec![0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835, 0.0725]);
        let petersburg: Array1<f64> = Array1::from_vec(vec![0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]);
        let magadan: Array1<f64> = Array1::from_vec(vec![0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764, 0.0689]);
        let tvarminne: Array1<f64> = Array1::from_vec(vec![0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]);
        // Test whether sample groups have different means
        // Answer obtained using `scipy.stats.f_oneway` function
        let test_result: FTestResult = f_test_anova(vec![&tillamook, &newport, &petersburg, &magadan, &tvarminne], Some(ConfidenceLevel::Five)).unwrap();
        assert_eq!(test_result.overall_mean, 0.0852);
        assert!((Array1::from_vec(test_result.group_means) - arr1(&[0.0802, 0.07479999999999999, 0.10344285714285714, 0.0780125, 0.09570000000000001])).sum() < TEST_ACCURACY);
        assert!((test_result.explained_variance - 0.0011299185267857142).abs() < TEST_ACCURACY);
        assert!((test_result.unexplained_variance - 0.00015867370273109246).abs() < TEST_ACCURACY);
        assert!((test_result.f_score - 7.121019471642447).abs() < TEST_ACCURACY);
        assert!((test_result.p_value - 0.00028122423145345444).abs() < TEST_ACCURACY);
        assert_eq!(test_result.reject_h0, true);
    }

    #[cfg(all(test, feature = "sample_data"))]
    #[test]
    fn unit_test_f_test_anova_2() -> () {
        use crate::utilities::sample_data::SampleData;
        use crate::statistics::stat_tests::{ConfidenceLevel, FTestResult, f_test_anova};
        let sample: SampleData = SampleData::Portfolio; 
        let (_, mut sample_data) = sample.load_sample_data();
        let jpm: Array1<f64> = sample_data.remove("JPM").unwrap();
        let gs: Array1<f64> = sample_data.remove("GS").unwrap();
        // Test whether jpm and gs have equal mean
        let test_result: FTestResult = f_test_anova(vec![&jpm, &gs], Some(ConfidenceLevel::Five)).unwrap();
        assert!((test_result.f_score - 10842.154721647501).abs() < TEST_ACCURACY);
        assert!((test_result.p_value - 0.0).abs() < TEST_ACCURACY);
        assert_eq!(test_result.reject_h0, true);
    }    
}