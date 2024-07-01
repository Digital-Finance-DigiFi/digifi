/// # Description
/// Measure of the performance of a portfolio compared to risk-free rate and adjusted for risk.
/// 
/// # Input
/// - portfolio_return: Expected return of the portfolio
/// - rf: Risk-free rate of return
/// - portfolio_std: Standard deviation of portfolio returns
/// 
/// # Output
/// - Sharpe raio
/// 
/// # LaTeX Formula
/// - S_{P} = \\frac{E[R_{P}]-r_{f}}{\\sigma^{2}_{P}}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Sharpe_ratio
/// - Original Source: https://doi.org/10.1086%2F294846
pub fn sharpe_ratio(portfolio_return: f64, rf: f64, portfolio_std: f64) -> f64 {
    (portfolio_return - rf) / portfolio_std
}


/// # Description
/// Measure of the performance of a portfolio compared to a benchmark relative to the volatility of the active return.
/// 
/// # Input
/// - portfolio_sharpe: Sharpe ratio of the portfolio
/// - benchmark_sharpe: Sharpe ratio of the benchmark portfolio
/// 
/// # Output
/// - Information ratio
/// 
/// # LaTeX Formula
/// - IR = \\frac{E[R_{P}-R_{B}]}{\\sqrt{Var[R_{P}-R_{B}]}} = \\sqrt{S^{2}_{P} - S^{2}_{B}}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Information_ratio
/// - Original Source: N/A
pub fn information_ratio(portfolio_sharpe: f64, benchmark_sharpe: f64) -> f64 {
    (portfolio_sharpe.powi(2) - benchmark_sharpe.powi(2)).sqrt()
}


/// # Description
/// Measure of the performance of a portfolio in excess of what could have been earned on an investment with no diversifiable risk.
/// 
/// # Input
/// - portfolio_return: Expected return of the portfolio
/// - rf: Risk-free rate of return
/// - portfolio_beta: Beta of the portfolio with respect to the market returns
/// 
/// # Output
/// - Treynor ratio
/// 
/// # LaTeX Formula
/// - T_{P}  =\\frac{E[R_{P}]-r_{f}}{\\beta_{P}}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Treynor_ratio
/// - Original Source: N/A
pub fn treynor_ratio(portfolio_return: f64, rf: f64, portfolio_beta: f64) -> f64 {
    (portfolio_return - rf) / portfolio_beta
}


/// # Description
/// Measure of the performance of the portfolio in excess of its theoretical expected return.
/// 
/// # Input
/// - portfolio_return: Actual portfolio return
/// - rf: Risk-free rate of return
/// - portfolio_beta: Beta of the portfolio with respect to the market returns
/// - market_return: Expected return of the market
/// 
/// # Output
/// - Jensen's alpha
/// 
/// # LaTeX Formula
/// - \\alpha_{J} = R_{P} - (r_{f} + \\beta_{P}(R_{M} - r_{f}))
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Jensen%27s_alpha
/// - Original Source: https://dx.doi.org/10.2139/ssrn.244153
pub fn jensens_alpha(portfolio_return: f64, rf: f64, portfolio_beta: f64, market_return: f64) -> f64 {
    portfolio_return - rf - portfolio_beta*(market_return - rf)
}


/// # Description
/// Measure of the performance of the portfolio in the form of risk-adjusted return.
/// 
/// This measure is the extension of Sharpe ratio, but it penalizes the 'downside' volatility and not the 'upside' volatility.
/// 
/// # Input
/// - portfolio_realized_return: Realized return of the portfolio
/// - target_return: Target or required return of the portfolio
/// - downside_risk: Downside risk (Standard deviation of the negative returns only)
/// 
/// # Output
/// - Sortino ratio
/// 
/// # LaTeX Formula
/// - S = \\frac{R_{P} - T}{DR}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Sortino_ratio
/// - Original Source: https://doi.org/10.3905%2Fjoi.3.3.59
pub fn sortino_ratio(portfolio_realized_return: f64, target_return: f64, downside_risk: f64) -> f64 {
    (portfolio_realized_return - target_return) / downside_risk
}