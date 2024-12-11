/// # Dscription
/// Exponential utility is a constant absolute risk aversion (CARA) utility measure with respect to consumption.
/// 
/// # Input
/// - `consumption`: Wealth or goods being measured
/// - `absolute_risk_aversion`: Parameter determining how risk averse the utility function is
/// 
/// # Output
/// - Utility
/// 
/// # LaTeX Formula
/// - u(c) = 1 - e^{-\\alpha c}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Risk_aversion#Absolute_risk_aversion>
/// - Original Source: N/A
pub fn cara(consumption: f64, absolute_risk_aversion: f64) -> f64 {
    1.0 - (-absolute_risk_aversion * consumption).exp()
}


/// # Description
/// Isoelastic utility is a constant relative risk aversion (CRRA) utility measure with respect to consumption.
/// 
/// # Input
/// - `consumption`: Wealth or goods being measured
/// - `relative_risk_aversion`: Parameter determining how risk averse the utility function is
/// 
/// # Output
/// - Utility
/// 
/// # LaTeX Formula
/// - u(c) = \\frac{c^{1-\\rho} - 1}{1 - \\rho}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Risk_aversion#Relative_risk_aversion>
/// - Original Source: N/A
pub fn crra(consumption: f64, relative_risk_aversion: f64) -> f64 {
    (consumption.powf(1.0-relative_risk_aversion) - 1.0) / (1.0 - relative_risk_aversion)
}