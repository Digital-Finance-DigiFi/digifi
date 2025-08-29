//! # Corporate Finance
//! 
//! Provides a set of basic functions commonly used in corporate finance and tools for constructing Captial Asset Pricing Model (CAPM).


// Re-Exports
pub use self::capm::{CAPMParams, CAPMType, CAPMSolutionType, CAPM};


pub mod capm;


/// Measure of how revenue growth translates to growth of income.
/// 
/// Degree of Operating Leverage = (% Change in Profits) / (% Change in Sales) = 1 + Total Fixed Cost / (Quantity of Goods Sold * (Price per Unit - Variable Cost per Unit) - Total Fixed Cost)
/// 
/// # Input
/// - `quantity_of_goods`: Quantity of goods sold
/// - `price_per_unit`: Price of every unit of good sold
/// - `variable_cost_per_unit`: Variable cost accumulated when producing a unit of good
/// - `total_fixed_cost`: Total fixed costs of producing all units sold
/// 
/// # Output
/// - Degree of operating leverage (DOL)
/// 
/// # LaTeX Formula
/// - DOL = 1 = \\frac{F}{Q(P-V) - F}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Operating_leverage>
/// - Original Source: N/A
pub fn dol(quantity_of_goods: f64, price_per_unit: f64, variable_cost_per_unit: f64, total_fixed_cost: f64) -> Option<f64> {
    let denominator: f64 = quantity_of_goods*(price_per_unit - variable_cost_per_unit) - total_fixed_cost;
    if denominator == 0.0 {
        None
    } else {
        Some(1.0 + total_fixed_cost / denominator)
    }
}


/// Monetary value of earnings per outstanding share of common stock for a company during a defined period of time.
/// 
/// EPS = (Net Income - Preferred Dividends) / Number of Common Shares Outstanding
/// 
/// # Input
/// - `net_income`: Net income
/// - `preferred_dividends`: Total dividends paid to the holders of the preferred stock
/// - `n_common_shares_outstanding`: Number of common shares outstanding
/// 
/// # Output
/// - Earnings per share (EPS)
/// 
/// # LaTeX Formula
/// - EPS = \\frac{(I-D_{pref})}{N_{common}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Earnings_per_share>
/// - Original Source: N/A
pub fn earnings_per_share(net_income: f64, preferred_dividends: f64, n_common_shares_outstanding: usize) -> Option<f64> {
    if n_common_shares_outstanding == 0 {
        None
    } else {
        Some((net_income - preferred_dividends) / n_common_shares_outstanding as f64)
    }
}


/// The ratio of market price to earnings.
/// 
/// Price-to-Earnings Ratio = Share Price / Earnings per Share
/// 
/// # Input
/// - `share_price`: Share price of the company
/// - `eps`: Earnings per share of the company
/// 
/// # Output
/// - P/E ratio
/// 
/// # LaTeX Formula
/// - PE = \\frac{P}{EPS}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Price%E2%80%93earnings_ratio>
/// - Original Source: N/A
pub fn pe_ratio(share_price: f64, eps: f64) -> Option<f64> {
    if eps == 0.0 {
        None
    } else {
        Some(share_price / eps)
    }
}


/// # Deescription
/// The ratio of market price to book value.
/// 
/// Price-to-Book Ratio = Market Capitalization / Book Value
/// 
/// # Input
/// - `market_cap`: Market capitalization of the company
/// - `book_value`: Value of the assets minus liabilities
/// 
/// # Output
/// - PB ratio
/// 
/// # LaTeX Formula
/// - PB = \\frac{Market Capitalization}{Book Value}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/P/B_ratio>
/// - Original Source: N/A
pub fn pb_ratio(market_cap: f64, book_value: f64) -> Option<f64> {
    if book_value == 0.0 {
        None
    } else {
        Some(market_cap / book_value)
    }
}


/// The ratio of dividend issued by the company to share price.
/// 
/// Dividend Yield = 100 * Dividend / Share Price
/// 
/// # Input
/// - `share_price`: Share price of the company
/// - `dividend`: Amount of dividend Paid out by the company per defined period
/// 
/// # Output
/// - Dividend yield
/// 
/// # LaTeX Formula
/// - D_{Y} = 100\\frac{D}{P}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Dividend_yield>
/// - Original Source: N/A
pub fn dividend_yield(share_price: f64, dividend: f64) -> Option<f64> {
    if share_price == 0.0 {
        None
    } else {
        Some(100.0 * dividend / share_price)
    }
}


/// Enterprise value is the sum of a company's market capitalization and any debts, minus cash or cash equivalents on hand.
/// 
/// EV = Maarket Cap - Total Debt + Cash & Cash Equivalents
/// 
/// # Input
/// - `market_cap`: Market capitalization of the company
/// - `total_debt`: Total debt of the company
/// - `cash`: Cash and cash equivalents (May not include marketable securities)
/// 
/// # Output
/// - Enterprise value (EV)
/// 
/// # LaTeX Formula
/// - EV = \\textit{Market Capitalization}+\\textit{Total Debt}-\\textit{Cash and Cash Equivalents}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Enterprise_value>
/// - Original Source: N/A
pub fn enterprise_value(market_cap: f64, total_debt: f64, cash: f64) -> f64 {
    market_cap + total_debt - cash
}

/// Measure of the value of a stock that compares a company's enterprise value to its revenue.
/// EV/R is one of several fundamental indicators that investors use to determine whether a stock is priced fairly.
/// The EV/R multiple is also often used to determine a company's valuation in the case of a potential acquisition.
/// It’s also called the enterprise value-to-sales multiple.
/// 
/// # Input
/// - `market_cap`: Market capitalization of the company
/// - `total_debt`: Total debt of the company
/// - `cash`: Cash and cash equivalents (May not include marketable securities)
/// - `revenue`: Revenue of the company
/// 
/// # Output
/// - EV/Revenue multiple
/// 
/// # LaTeX Formula
/// - \\frac{EV}{R} = \\frac{\\textit{Market Capitalization}+\\textit{Total Debt}-\\textit{Cash and Cash Equivalents}}{\\textit{Revenue}}
/// 
/// # Links
/// - Wikipedia: N/A
/// - Original Source: N/A
pub fn ev_to_revenue(market_cap: f64, total_debt: f64, cash: f64, revenue: f64) -> Option<f64> {
    if revenue == 0.0 {
        None
    } else {
        Some(enterprise_value(market_cap, total_debt, cash) / revenue)
    }
}


/// Valuation multiple used to determine the fair market value of a company.
/// By contrast to the more widely available P/E ratio (price-earnings ratio) it includes debt as part of the value of the
/// company in the numerator and excludes costs such as the need to replace depreciating plant, interest on debt,
/// and taxes owed from the earnings or denominator.
/// 
/// # Input
/// - `market_cap`: Market capitalization of the company
/// - `total_debt`: Total debt of the company
/// - `cash`: Cash and cash equivalents (May not include marketable securities)
/// - `ebitda`: EBITDA of the company
/// 
/// # Output
/// - EV/EBITDA multiple
/// 
/// # LaTeX Formula
/// - \\frac{EV}{EBITDA} = \\frac{\\textit{Market Capitalization}+\\textit{Total Debt}-\\textit{Cash and Cash Equivalents}}{\\textit{EBITDA}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/EV/Ebitda>
/// - Original Source: N/A
pub fn ev_to_ebitda(market_cap: f64, total_debt: f64, cash: f64, ebitda: f64) -> Option<f64> {
    if ebitda == 0.0 {
        None
    } else {
        Some(enterprise_value(market_cap, total_debt, cash) / ebitda)
    }
}


/// Value of assets of the company minus its liabilities.
/// 
/// Book Value = Assets - Liabilities
/// 
/// # Input
/// - `assets`: Total assets of the company
/// - `liabilities`: Total liabilities of the company
/// 
/// # Output
/// - Book value
/// 
/// # LaTeX Formula
/// - \\textit{Book Value} = \\textit{Assets} - \\textit{Liabilities}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Book_value>
/// - Original Source: N/A
pub fn book_value(assets: f64, liabilities: f64) -> f64 {
    assets - liabilities
}


/// Cost of equity capital (Market capitalization rate).
/// 
/// Cost of Equity Capital = (Expected Dividend + Expected Share Price - Share Price) / Share Price
/// 
/// # Input
/// - `share_price`: Share price of the company
/// - `expected_dividend`: Expected dividend to be received in the future
/// - `expected_share_price`: Expected share price of the company in the future
/// 
/// # Output
/// - Cost of equity capital (Market capitalization rate)
/// 
/// # LaTeX Formula
/// - r = \\frac{D_{t+1} + P_{t+1} - P_{t}}{P_{t}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Cost_of_equity>
/// - Original Source: N/A
pub fn cost_of_equity_capital(share_price: f64, expected_dividend: f64, expected_share_price: f64) -> Option<f64> {
    if share_price == 0.0 {
        None
    } else {
        Some((expected_dividend + expected_share_price - share_price) / share_price)
    }
}


/// Ratio of dividends to earnings per share.
/// 
/// Payout Ratio = Dividend per Share / Earnings per Share
/// 
/// # Input
/// - `dividend_per_share`: Dividend per share paid out closest to the latest earnings
/// - `earnings_per_share`: Earnings per share
/// 
/// # Output
/// - Payout ratio
/// 
/// # LaTeX Formula
/// - \\textit{Payout Ratio} = \\frac{D_{t}}{EPS_{t}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Dividend_payout_ratio>
/// - Original Source: N/A
pub fn payout_ratio(dividend_per_share: f64, earnings_per_share: f64) -> Option<f64> {
    if earnings_per_share == 0.0 {
        None
    } else {
        Some(dividend_per_share / earnings_per_share)
    }
}


/// One minus payout ratio.
/// 
/// Plowback Ratio = 1 - (Dividend per Share / Earnings per Share)
/// 
/// # Input
/// - `dividend_per_share`: Dividend per share paid out closest to the latest earnings
/// - `earnings_per_share`: Earnings per share
/// 
/// # Output
/// - Plowback ratio
/// 
/// # LaTeX Formula
/// - \\textit{Plowback Ratio} = 1 - \\frac{D_{t}}{EPS_{t}}
/// 
/// # Links
/// - Wikipedia: N/A
/// - Original Source: N/A
pub fn plowback_ratio(dividend_per_share: f64, earnings_per_share: f64) -> Option<f64> {
    if earnings_per_share == 0.0 {
        None
    } else {
        Some(1.0 - dividend_per_share / earnings_per_share)
    }
}


/// Measure for predicting the likelihood of bankrupcy of a company.
/// 
/// # Input
/// - `ebit`: EBIT of the company
/// - `total_assets`: Total assets of the company
/// - `sales`: Total sales of the company
/// - `equity`: Market value of equity
/// - `total_liabilities`: Total liabilities of the company
/// - `retained_earnings`: Retained earnings of the company
/// - `working_capital`: Working capital of the company
/// 
/// # Output
/// - Altman's Z-Score: If the value is below 1.81 - there is a high vulnerability to bankrupcy,
/// if the value is above 2.99 - there is a low vulnerability to bankrupcy
/// 
/// # LaTeX Formula
/// - Z = 3.3\\frac{EBIT}{\\textit{Total Assets}} + 1.0\\frac{Sales}{Assets} + 0.6\\frac{Equity}{\\textit{Total Liabilities}} + 
/// 1.4\\frac{\\textit{Retained Earning}}{\\textit{Total Assets}} + 1.2\\frac{\\textit{Working Capital}}{\\textit{Total Assets}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Altman_Z-score>
/// - Origina Source: <https://doi.org/10.1002/9781118266236.ch19>
pub fn altman_z_score(ebit: f64, total_assets: f64, sales: f64, equity: f64, total_liabilities: f64, retained_earnings: f64, working_capital: f64) -> Option<f64> {
    if total_assets == 0.0 || total_liabilities == 0.0 {
        None
    } else {
        Some(3.3*ebit/total_assets + sales/total_assets + 0.6*equity/total_liabilities + 1.4*retained_earnings/total_assets + 1.2*working_capital/total_assets)
    }
}


/// Computes the weighted average cost of capital (WACC), which is the expected return on the company's assets.
/// 
/// WACC = (Debt / (Debt+Equity) * (1 - Corporate Tax Rate) * Return on Debt) + (Equity / (Debt+Equity) * Return on Equity)
/// 
/// # Input
/// - `equity`: Total equity of the company
/// - `debt`: Total debt of the company
/// - `return_on_equity`: Expected return on equity of the company
/// - `return_on_debt`: Expected return on debt of the company
/// - `corporate_tax`: Corporate tax rate on earnings after interest, EBT
/// 
/// # Output
/// - Weighted average cost of capital (WACC)
/// 
/// # LaTeX Formula
/// - r_{A} = [r_{D}(1-T_{c})\\frac{D}{E+D}] + [r_{E}\\frac{E}{E+D}]
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Weighted_average_cost_of_capital>
/// - Original Source: N/A
pub fn weighted_average_cost_of_capital(equity: f64, debt: f64, return_on_equity: f64, return_on_debt: f64, corporate_tax: f64) -> Option<f64> {
    let capital_structure: f64 = debt + equity;
    if capital_structure == 0.0 {
        None
    } else {
        Some(return_on_debt*(1.0-corporate_tax)*debt/capital_structure + return_on_equity*equity/capital_structure)
    }
}


/// Computes expected return on the equity (ROE) portion of the company.
/// 
/// ROE = Return on Assets + (Return on Assets - Return on Debt * (1 - Corporate Tax Rate)) * Debt / Equity
/// 
/// # Input
/// - `equity`: Total equity of the company
/// - `debt`: Total debt of the company
/// - `return_on_assets`: Return on all assets (WACC) of the company
/// - `return_on_debt`: Expected return on debt of the company
/// - `corporate_tax`: Corporate tax rate on earnings after interest, EBT
/// 
/// # Output
/// - Expected return on equity (ROE)
/// 
/// # LaTeX Formula
/// - r_{E} = r_{A} + (r_{A}-r_{D}(1-T_{c}))\\frac{D}{E}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Return_on_equity>
/// - Original Source: N/A
pub fn expected_return_on_equity(equity: f64, debt: f64, return_on_assets: f64, return_on_debt: f64, corporate_tax: f64) -> Option<f64> {
    if equity == 0.0 {
        None
    } else {
        Some(return_on_assets + (return_on_assets - return_on_debt*(1.0-corporate_tax)) * (debt/equity))
    }
}


// # Description
/// Measure of a company's financial performance. It is calculated by dividing net income by shareholders' equity.
/// Because shareholders' equity is equal to a company’s assets minus its debt, ROE is a way of showing a company's
/// return on net assets.
/// 
/// ROE = Net Income / Shareholder's Equity
/// 
/// # Input
/// - `net_income`: Net income of the company
/// - `equity`: Average shareholder's equity
/// 
/// # Output
/// - Return on equity (ROE)
/// 
/// # LaTeX Formula
/// - ROE = \\frac{\\textit{Net Income}}{Equity}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Return_on_equity>
/// - Original Source: N/A
pub fn return_on_equity(net_income: f64, equity: f64) -> Option<f64> {
    if equity == 0.0 {
        None
    } else {
        Some(net_income / equity)
    }
}


/// Measure of a company's performance, but the ROE is split into multiple components.
/// 
/// ROE = Net Profit Margin * Total Asset Turnover * Financial Leverage
/// 
/// # Input
/// - `net_profit_margin`: Net profit margin
/// - `total_asset_turnover`: Financial ratio that measures the efficiency of a company's use of its assets in generating sales revenue or sales income to the company
/// - `total_assets`: Total assets of the company
/// - `total_shareholders_equity`: Total shareholder's equity
/// 
/// # Output
/// - DuPont return on equity (ROE)
/// 
/// # LaTeX Formula
/// - ROE_{DuPont} = \\textit{Net Profit Margin} \\times \\textit{Total Asset Turnover} \\times\\frac{\\textit{Total Assets}}{\\textit{Total Shareholders' Equity}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/DuPont_analysis>
/// - Original Source: N/A
pub fn return_on_equity_dupont(net_profit_margin: f64, total_asset_turnover: f64, total_assets: f64, total_shareholders_equity: f64) -> Option<f64> {
    if total_shareholders_equity == 0.0 {
        None
    } else {
        let financial_leverage: f64 = total_assets / total_shareholders_equity;
        Some(net_profit_margin * total_asset_turnover * financial_leverage)
    }
}


/// Financial ratio that indicates how profitable a company is relative to its total assets.
/// It can used to determine how efficiently a company uses its resources to generate a profit.
/// 
/// ROA = Net Income / Total Assets
/// 
/// # Input
/// - `net_income`: Net income of the company
/// - `total_assets`: Average total assets
/// 
/// # Output
/// - Return on assets (ROA)
/// 
/// # LaTeX Formula
/// - ROA = \\frac{\\textit{Net Income}}{\\textit{Total Assets}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Return_on_assets>
/// - Original Source: N/A
pub fn return_on_assets(net_income: f64, total_assets: f64) -> Option<f64> {
    if total_assets == 0.0 {
        None
    } else {
        Some(net_income / total_assets)
    }
}


/// Performance measure used to evaluate the efficiency or profitability of an investment or compare the efficiency
/// of a number of different investments. ROI tries to directly measure the amount of return on a particular investment,
/// relative to the investment’s cost.
/// 
/// ROI = (Gain from Investment - Cost of Investment) / Cost of Investment
/// 
/// # Input
/// - `gain_from_investment`: Gain from investment (e.g., revenue)
/// - `cost_of_investment`: Cost of investment (e.g., cost of goods sold)
/// 
/// # Output
/// - Return on investment (ROI)
/// 
/// # LaTeX Formula
/// - ROI = \\frac{Gain - Cost}{Cost}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Return_on_investment>
/// - Original Source: N/A
pub fn return_on_investment(gain_from_investment: f64, cost_of_investment: f64) -> Option<f64> {
    if cost_of_investment == 0.0 {
        None
    } else {
        Some((gain_from_investment - cost_of_investment) / cost_of_investment)
    }
}


/// Ratio indicating the relative proportion of shareholders' equity and debt used to finance the company's assets.
/// 
/// D/E = Debt / Equity
/// 
/// # Input
/// - `debt`: Debt portion of the corporate structure
/// - `equity`: Equity portion of the corporate structure
/// 
/// # Output
/// Debt-to-Equity ratio
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Debt-to-equity_ratio>
/// - Original Source: N/A
pub fn debt_to_equity(debt: f64, equity: f64) -> Option<f64> {
    if equity == 0.0 {
        None
    } else {
        Some(debt / equity)
    }
}


/// Unlevered beta, which is the systematic risk of the company's assets.
/// 
/// Unlevered Beta = (Debt / (Debt+Equity) * Beta of Debt) + (Equity / (Debt+Equity) * Beta of Equity)
/// 
/// # Input
/// - `equity`: Total equity of the company
/// - `debt`: Total debt of the company
/// - `beta_equity`: Levered beta of the company
/// - `beta_debt`: Beta debt of the company
/// 
/// # Output
/// - Unlevered beta
/// 
/// # LaTeX Formula
/// - \\beta_{A} = [\\beta_{D}\\frac{D}{E+D}] + [\\beta_{E}\\frac{E}{E+D}]
pub fn unlevered_beta(equity: f64, debt: f64, beta_equity: f64, beta_debt: f64) -> Option<f64> {
    let capital_structure: f64 = debt + equity;
    if capital_structure == 0.0 {
        None
    } else {
        Some(beta_debt*debt/capital_structure + beta_equity*equity/capital_structure)
    }
}


/// Levered beta, which is the equity-only beta of the company.
/// 
/// Levered Beta = Beta of Assets + (Beta of Assets - Beta of Debt) * (Debt / Equity)
/// 
/// # Input
/// - `equity`: Total equity of the company
/// - `debt`: Total debt of the company
/// - `beta_assets`: Unlevered beta of the company
/// - `beta_debt`: Beta debt of the company
/// 
/// # Output
/// - Levered beta
/// 
/// # LaTeX Formula
/// - \\beta_{E} = \\beta_{A} + (\\beta_{A} - \\beta_{D})\\frac{D}{E}
pub fn levered_beta(equity: f64, debt: f64, beta_assets: f64, beta_debt: f64) -> Option<f64> {
    if equity == 0.0 {
        None
    } else {
        Some(beta_assets + (beta_assets - beta_debt)*debt/equity)
    }
}


/// Calculates an advantage of debt financing for a company as opposed to equity financing from perspective of tax optimization.
/// 
/// Relative Tax Advantage of Debt = (1 - Personal Tax on Interest Income) / ((1 - Effective Personal Tax) * (1 - Corporate Tax))
/// 
/// # Input
/// - `corporate_tax`: Corporate tax rate applied to a company after debt payout
/// - `personal_tax`: Personal tax rate on a interest income
/// - `effective_personal_tax`: Effective tax rate on equity income comprising personal tax on dividend income and personal tax on capital gains income
/// 
/// # Output
/// - Relative tax advantage of debt ratio
/// 
/// # LaTeX Formula
/// - \\textit{Relative Tax Advantage of Debt} = \\frac{1-T_{p}}{(1-T_{pE})(1-T_{c})}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Tax_benefits_of_debt>
/// - Original Source: N/A
pub fn relative_tax_advantage_of_debt(corporate_tax: f64, personal_tax: f64, effective_personal_tax: f64) -> Option<f64> {
    let denominator: f64 = (1.0-effective_personal_tax) * (1.0-corporate_tax);
    if denominator == 0.0 {
        None
    } else {
        Some((1.0-personal_tax) / denominator)
    }
}