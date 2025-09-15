#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};


pub trait PortfolioPerformanceMetric: {
    /// Measures the performance of the portfolio based on portfolio data.
    ///
    /// # Input
    /// - `portfolio_return`: Expected return of the portfolio
    /// - `portfolio_std`: Standard deviation of portfolio returns
    fn performance(&self, portfolio_return: f64, portfolio_std: f64) -> f64;

    /// Objective function that will be minimized to find the optimal weights distribution of the portfolio.
    ///
    /// # Input
    /// - `portfolio_return`: Expected return of the portfolio
    /// - `portfolio_std`: Standard deviation of portfolio returns
    fn objective_function(&self, portfolio_return: f64, portfolio_std: f64) -> f64;
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Measure of the performance of a portfolio compared to risk-free rate and adjusted for risk.
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Sharpe_ratio>
/// - Original Source: <https://doi.org/10.1086%2F294846>
pub struct SharpeRatio {
    /// Risk-free rate of return
    pub rf: f64,
}

impl PortfolioPerformanceMetric for SharpeRatio {

    /// Measure of the performance of a portfolio compared to risk-free rate and adjusted for risk.
    /// 
    /// # Input
    /// - `portfolio_return`: Expected return of the portfolio
    /// - `portfolio_std`: Standard deviation of portfolio returns
    /// 
    /// # Output
    /// - Sharpe raio
    /// 
    /// # LaTeX Formula
    /// - S_{P} = \\frac{E[R_{P}]-r_{f}}{\\sigma^{2}_{P}}
    fn performance(&self, portfolio_return: f64, portfolio_std: f64) -> f64 {
        (portfolio_return - self.rf) / portfolio_std
    }

    fn objective_function(&self, portfolio_return: f64, portfolio_std: f64) -> f64 {
        -self.performance(portfolio_return, portfolio_std)
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Measure of the performance of a portfolio compared to a benchmark relative to the volatility of the active return.
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Information_ratio>
/// - Original Source: N/A
pub struct InformationRatio {
    /// Risk-free rate of return
    pub rf: f64,
    /// Sharpe ratio of the benchmark portfolio
    pub benchmark_sharpe_ratio: f64,
}

impl PortfolioPerformanceMetric for InformationRatio {

    /// Measure of the performance of a portfolio compared to a benchmark relative to the volatility of the active return.
    ///
    /// # Input
    /// - `portfolio_return`: Expected return of the portfolio
    /// - `portfolio_std`: Standard deviation of portfolio returns
    ///
    /// # Output
    /// - Information ratio
    /// 
    /// # LaTeX Formula
    /// - IR = \\frac{E[R_{P}-R_{B}]}{\\sqrt{Var[R_{P}-R_{B}]}} = \\sqrt{S^{2}_{P} - S^{2}_{B}}
    fn performance(&self, portfolio_return: f64, portfolio_std: f64) -> f64 {
        (((portfolio_return - self.rf) / portfolio_std).powi(2) - self.benchmark_sharpe_ratio.powi(2)).sqrt()
    }

    fn objective_function(&self, portfolio_return: f64, portfolio_std: f64) -> f64 {
        self.performance(portfolio_return, portfolio_std)
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Measure of the performance of a portfolio in excess of what could have been earned on an investment with no diversifiable risk.
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Treynor_ratio>
/// - Original Source: N/A
pub struct TreynorRatio {
    /// Risk-free rate of return
    pub rf: f64,
    /// Expected return of the benchmark market
    pub expected_market_return: f64,
}

impl PortfolioPerformanceMetric for TreynorRatio {

    /// Measure of the performance of a portfolio in excess of what could have been earned on an investment with no diversifiable risk.
    /// 
    /// # Input
    /// - `portfolio_return`: Expected return of the portfolio
    /// - `portfolio_std`: Standard deviation of portfolio returns
    /// 
    /// # Output
    /// - Treynor ratio
    /// 
    /// # LaTeX Formula
    /// - T_{P}  =\\frac{E[R_{P}]-r_{f}}{\\beta_{P}}
    fn performance(&self, portfolio_return: f64, _: f64) -> f64 {
        let portfolio_beta: f64 = (portfolio_return - self.rf) / (self.expected_market_return - self.rf);
        (portfolio_return - self.rf) / portfolio_beta
    }

    fn objective_function(&self, portfolio_return: f64, portfolio_std: f64) -> f64 {
        -self.performance(portfolio_return, portfolio_std)
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Measure of the performance of the portfolio in excess of its theoretical expected return.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Jensen%27s_alpha>
/// - Original Source: <https://dx.doi.org/10.2139/ssrn.244153>
pub struct JensensAlpha {
    /// Risk-free rate of return
    pub rf: f64,
    /// Beta of the portfolio with respect to the market returns
    pub theoretical_portfolio_beta: f64,
    /// Expected return of the benchmark market
    pub expected_market_return: f64,
}

impl PortfolioPerformanceMetric for JensensAlpha {

    /// Measure of the performance of the portfolio in excess of its theoretical expected return.
    /// 
    /// # Input
    /// - `portfolio_return`: Actual portfolio return
    /// - `portfolio_std`: Standard deviation of portfolio returns
    /// 
    /// # Output
    /// - Jensen's alpha
    /// 
    /// # LaTeX Formula
    /// - \\alpha_{J} = R_{P} - (r_{f} + \\beta_{P}(R_{M} - r_{f}))
    fn performance(&self, portfolio_return: f64, _: f64) -> f64 {
        portfolio_return - self.rf - self.theoretical_portfolio_beta * (self.expected_market_return - self.rf)
    }

    fn objective_function(&self, portfolio_return: f64, portfolio_std: f64) -> f64 {
        -self.performance(portfolio_return, portfolio_std)
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Measure of the performance of the portfolio in the form of risk-adjusted return.
/// 
/// This measure is the extension of Sharpe ratio, but it penalizes the 'downside' volatility and not the 'upside' volatility.
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Sortino_ratio>
/// - Original Source: <https://doi.org/10.3905%2Fjoi.3.3.59>
pub struct SortinoRatio {
    /// Target or required return of the portfolio
    pub target_portfolio_return: f64,
    /// Downside risk (Standard deviation of the negative returns only)
    pub downside_risk: f64,
}

impl PortfolioPerformanceMetric for SortinoRatio {

    /// Measure of the performance of the portfolio in the form of risk-adjusted return.
    /// 
    /// This measure is the extension of Sharpe ratio, but it penalizes the 'downside' volatility and not the 'upside' volatility.
    /// 
    /// # Input
    /// - `portfolio_return`: Realized return of the portfolio
    /// - `portfolio_std`: Standard deviation of portfolio returns
    /// 
    /// # Output
    /// - Sortino ratio
    /// 
    /// # LaTeX Formula
    /// - S = \\frac{R_{P} - T}{DR}
    fn performance(&self, portfolio_return: f64, _: f64) -> f64 {
        (portfolio_return - self.target_portfolio_return) / self.downside_risk
    }

    fn objective_function(&self, portfolio_return: f64, portfolio_std: f64) -> f64 {
        -self.performance(portfolio_return, portfolio_std)
    }
}