use std::io::Error;
use ndarray::Array1;
use crate::utilities::input_error;
use crate::statistics::ProbabilityDistribution;


/// # Description
/// Measure of the risk of a portfolio estimating how much a portfolio can lose in a specified period.
/// 
/// Note: This function uses the convention where 5% V@R of $1 million is the 0.05 probability that the portfolio will go down by $1 million.
/// 
/// Note: The V@R is quoted as a positive number, if V@R is negative it implies that the portfolio has very high chace of making a profit.
/// 
/// # Input
/// - alpha: Probability level for V@R
/// - returns_distribution: Probability distribution object with an inverse CDF method
/// 
/// # Output
/// - Value at risk (V@R)
/// 
/// # links
/// - Wikipedia: https://en.wikipedia.org/wiki/Value_at_risk#
/// - Original Source: N/A
pub fn value_at_risk(alpha: f64, returns_distribution: &impl ProbabilityDistribution) -> Result<f64, Error> {
    if (alpha < 0.0) || (1.0 < alpha) {
        return Err(input_error("The argument alpha must be in the range [0, 1]."));
    }
    Ok(-returns_distribution.inverse_cdf(&Array1::from_vec(vec![alpha]))?[0])
}


/// # Description
/// Measure of the risk of a portfolio that evaluates the expected return of a portfolio in the worst percentage of cases.
/// 
/// Note: This function uses the convention where ES at 5% is the expected shortfall of the 5% of worst cases.
/// 
/// # Input
/// - alpha: Probability level for ES
/// - returns_distribution: Probability distribution object with an inverse CDF method
/// 
/// # Output
/// - Expected shortfall (ES)
/// 
/// # links
/// - Wikipedia: https://en.wikipedia.org/wiki/Expected_shortfall
/// - Original Source: N/A
pub fn expected_shortfall(alpha: f64, returns_distribution: &impl ProbabilityDistribution) -> Result<f64, Error> {
    if (alpha < 0.0) || (1.0 < alpha) {
        return Err(input_error("The argument alpha must be in the range [0, 1]."));
    }
    todo!();
    // TODO: Implement expected_shortfall
    // let function_integral: Fn(f64) -> f64 = |p| { value_at_risk(p, returns_distribution)? };
    // Ok(numerical_integration(function_integral, 0.0, alpha) / alpha)
}