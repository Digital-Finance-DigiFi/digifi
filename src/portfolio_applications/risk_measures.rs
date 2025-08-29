use ndarray::Array1;
use crate::error::DigiFiError;
use crate::utilities::maths_utils::definite_integral;
use crate::statistics::ProbabilityDistribution;


/// Measure of the risk of a portfolio estimating how much a portfolio can lose in a specified period.
/// 
/// Note: This function uses the convention where 5% V@R of $1 million is the 0.05 probability that the portfolio will go down by $1 million.
/// 
/// Note: The V@R is quoted as a positive number, if V@R is negative it implies that the portfolio has very high chace of making a profit.
/// 
/// # Input
/// - `alpha`: Probability level for V@R
/// - `returns_distribution`: Probability distribution object with an inverse CDF method
/// 
/// # Output
/// - Value at risk (V@R)
/// 
/// # Errors
/// - Returns an error if the argument `alpha` is not in the range \[0, 1\].
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Value_at_risk#>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::arr1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, NormalDistribution};
/// use digifi::portfolio_applications::value_at_risk;
///
/// let norm_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0).unwrap();
///
/// assert!((value_at_risk(0.1, &norm_dist).unwrap() - 1.28).abs() < 1_000_000.0 * TEST_ACCURACY);
/// ```
pub fn value_at_risk(alpha: f64, returns_distribution: &impl ProbabilityDistribution) -> Result<f64, DigiFiError> {
    if (alpha < 0.0) || (1.0 < alpha) {
        return Err(DigiFiError::ParameterConstraint {
            title: "Value at Risk".to_owned(),
            constraint: "The argument `alpha` must be in the range `[0, 1]`.".to_owned(),
        });
    }
    Ok(-returns_distribution.inverse_cdf(&Array1::from_vec(vec![alpha]))?[0])
}


/// Measure of the risk of a portfolio that evaluates the expected return of a portfolio in the worst percentage of cases.
/// 
/// Note: This function uses the convention where ES at 5% is the expected shortfall of the 5% of worst cases.
/// 
/// # Input
/// - `alpha`: Probability level for ES
/// - `returns_distribution`: Probability distribution object with an inverse CDF method
/// 
/// # Output
/// - Expected shortfall (ES)
/// 
/// # Errors
/// - Returns an error if the argument `alpha` is not in the range \[0, 1\].
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Expected_shortfall>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::arr1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::statistics::{ProbabilityDistribution, NormalDistribution};
/// use digifi::portfolio_applications::expected_shortfall;
///
/// let norm_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0).unwrap();
///
/// let es: f64 = expected_shortfall(0.1, &norm_dist).unwrap();
///
/// let theor: f64 = (-norm_dist.inverse_cdf(&arr1(&[0.1])).unwrap()[0].powi(2) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt() / 0.1;
/// assert!((es - theor).abs() < 10_000_000.0 * TEST_ACCURACY);
/// ```
pub fn expected_shortfall(alpha: f64, returns_distribution: &impl ProbabilityDistribution) -> Result<f64, DigiFiError> {
    if (alpha < 0.0) || (1.0 < alpha) {
        return Err(DigiFiError::ParameterConstraint {
            title: "Expected Shortfall".to_owned(),
            constraint: "The argument `alpha` must be in the range `[0, 1]`.".to_owned(),
        });
    }
    let f = |p| { value_at_risk(p, returns_distribution).unwrap() };
    Ok(definite_integral(f, 0.0, alpha, 1_000_000) / alpha)
}


#[cfg(test)]
mod tests {
    use ndarray::arr1;
    use crate::statistics::ProbabilityDistribution;
    use crate::utilities::TEST_ACCURACY;
    use crate::statistics::continuous_distributions::NormalDistribution;

    #[test]
    fn unit_test_value_at_risk() -> () {
        use crate::portfolio_applications::risk_measures::value_at_risk;
        let norm_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0).unwrap();
        assert!((value_at_risk(0.1, &norm_dist).unwrap() - 1.28).abs() < 1_000_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_expected_shortfall() -> () {
        use crate::portfolio_applications::risk_measures::expected_shortfall;
        let norm_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0).unwrap();
        let es: f64 = expected_shortfall(0.1, &norm_dist).unwrap();
        let theor: f64 = (-norm_dist.inverse_cdf(&arr1(&[0.1])).unwrap()[0].powi(2) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt() / 0.1;
        assert!((es - theor).abs() < 10_000_000.0 * TEST_ACCURACY);
    }
}