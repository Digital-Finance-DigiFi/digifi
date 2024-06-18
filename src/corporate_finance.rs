pub mod capm;


/// # Description
/// Measure of how revenue growth translates to growth of income.\n
/// Degree of Operating Leverage = (% Change in Profits) / (% Change in Sales) = 1 + Total Fixed Cost / (Quantity of Goods Sold * (Price per Unit - Variable Cost per Unit) - Total Fixed Cost)
/// 
/// # Input
/// - quantity_of_goods: Quantity of goods sold
/// - price_per_unit: Price of every unit of good sold
/// - variable_cost_per_unit: Variable cost accumulated when producing a unit of good
/// - total_fixed_cost: Total fixed costs of producing all units sold
/// 
/// # Output
/// - Degree of operating leverage (DOL)
/// 
/// # LaTeX Formula
/// - DOL = 1 = \\frac{F}{Q(P-V) - F}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Operating_leverage
/// - Original Source: N/A
pub fn dol(quantity_of_goods: f64, price_per_unit: f64, variable_cost_per_unit: f64, total_fixed_cost: f64) -> f64 {
    1.0 + total_fixed_cost / (quantity_of_goods*(price_per_unit - variable_cost_per_unit) - total_fixed_cost)
}


/// # Description
/// The ratio of market price to earnings.\n
/// Price-to-Earnings Ratio = Share Price / Earnings per Share
/// 
/// # Input
/// - share_price: Share price of the company
/// - eps: Earnings per share of the company
/// 
/// # Output
/// - P/E ratio
/// 
/// # LaTeX Formula
/// - PE = \\frac{P}{EPS}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Price%E2%80%93earnings_ratio
/// - Original Source: N/A
pub fn pe_ratio(share_price: f64, eps: f64) -> f64 {
    share_price / eps
}


/// # Deescription
/// The ratio of market price to book value.\n
/// Price-to-Book Ratio = Market Capitalization / Book Value
/// 
/// # Input
/// - market_cap: Market capitalization of the company
/// - book_value: Value of the assets minus liabilities
/// 
/// # Output
/// - PB ratio
/// 
/// # LaTeX Formula
/// - PB = \\frac{Market Capitalization}{Book Value}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/P/B_ratio
/// - Original Source: N/A
pub fn pb_ratio(market_cap: f64, book_value: f64) -> f64 {
    market_cap / book_value
}


/// # Description
/// The ratio of dividend issued by the company to share price.\n
/// Dividend Yield = 100 * Dividend / Share Price
/// 
/// # Input
/// - share_price: Share price of the company
/// - dividend: Amount of dividend Paid out by the company per defined period
/// 
/// # Output
/// - Dividend yield
/// 
/// # LaTeX Formula
/// - D_{Y} = 100\\frac{D}{P}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Dividend_yield
/// - Original Source: N/A
pub fn dividend_yield(share_price: f64, dividend: f64) -> f64 {
    100.0 * dividend / share_price
}


/// # Description
/// Value of assets of the company minus its liabilities.\n
/// Book Value = Assets - Liabilities
/// 
/// # Input
/// - assets: Total assets of the company
/// - liabilities: Total liabilities of the company
/// 
/// # Output
/// - Book value
/// 
/// # LaTeX Formula
/// - \\textit{Book Value} = \\textit{Assets} - \\textit{Liabilities}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Book_value
/// - Original Source: N/A
pub fn book_value(assets: f64, liabilities: f64) -> f64 {
    assets - liabilities
}


/// # Description
/// Cost of equity capital (Market capitalization rate).\n
/// Cost of Equity Capital = (Expected Dividend + Expected Share Price - Share Price) / Share Price
/// 
/// # Input
/// - share_price: Share price of the company
/// - expected_dividend: Expected dividend to be received in the future
/// - expected_share_price: Expected share price of the company in the future
/// 
/// # Output
/// - Cost of equity capital (Market capitalization rate) (float)
/// 
/// # LaTeX Formula
/// - r = \\frac{D_{t+1} + P_{t+1} - P_{t}}{P_{t}}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Cost_of_equity
/// - Original Source: N/A
pub fn cost_of_equity_capital(share_price: f64, expected_dividend: f64, expected_share_price: f64) -> f64 {
    (expected_dividend + expected_share_price - share_price) / share_price
}


/// # Description
/// Measure of profitability of the company in relation to its equity.\n
/// ROE = Total Earnings / Book Value
/// 
/// # Input
/// - total_earnings: Total earnings of the company
/// - book_value: Value of the assets minus liabilities
/// 
/// # Output
/// - Return on equity (ROE)
/// 
/// # LaTeX Formula
/// - ROE  =\\frac{\\textit{Total Earnings}}{\\textit{Book Value}}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Return_on_equity
/// - Original Source: N/A
pub fn roe(total_earning: f64, book_value: f64) -> f64 {
    total_earning / book_value
}


/// # Description
/// Ratio of dividends to earnings per share.\n
/// Payout Ratio = Dividend per Share / Earnings per Share
/// 
/// # Input
/// - dividend_per_share: Dividend per share paid out closest to the latest earnings
/// - earnings_per_share: Earnings per share
/// 
/// # Output
/// - Payout ratio
/// 
/// # LaTeX Formula
/// - \\textit{Payout Ratio} = \\frac{D_{t}}{EPS_{t}}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Dividend_payout_ratio
/// - Original Source: N/A
pub fn payout_ratio(dividend_per_share: f64, earnings_per_share: f64) -> f64 {
    dividend_per_share / earnings_per_share
}


/// # Description
/// One minus payout ratio.\n
/// Plowback Ratio = 1 - (Dividend per Share / Earnings per Share)
/// 
/// # Input
/// - dividend_per_share: Dividend per share paid out closest to the latest earnings
/// - earnings_per_share: Earnings per share
/// 
/// # Output
/// - Plowback ratio
/// 
/// # LaTeX Formula
/// - \\textit{Plowback Ratio} = 1 - \\frac{D_{t}}{EPS_{t}}
pub fn plowback_ratio(dividend_per_share: f64, earnings_per_share: f64) -> f64 {
    1.0 - dividend_per_share / earnings_per_share
}


/// # Description
/// Measure for predicting the likelihood of bankrupcy of a company.
/// 
/// # Input
/// - ebit: EBIT of the company
/// - total_assets: Total assets of the company
/// - sales: Total sales of the company
/// - equity: Market value of equity
/// - total_liabilities: Total liabilities of the company
/// - retained_earnings: Retained earnings of the company
/// - working_capital: Working capital of the company
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
/// - Wikipedia: https://en.wikipedia.org/wiki/Altman_Z-score
/// - Origina Source: https://doi.org/10.1002/9781118266236.ch19
pub fn altman_z_score(ebit: f64, total_assets: f64, sales: f64, equity: f64, total_liabilities: f64, retained_earnings: f64, working_capital: f64) -> f64 {
    3.3*ebit/total_assets + sales/total_assets + 0.6*equity/total_liabilities + 1.4*retained_earnings/total_assets + 1.2*working_capital/total_assets
}


/// # Description
/// Computes the weighted average cost of capital (WACC), which is the expected return on the company's assets.\n
/// WACC = (Debt / (Debt+Equity) * (1 - Corporate Tax Rate) * Return on Debt) + (Equity / (Debt+Equity) * Return on Equity)
/// 
/// # Input
/// - equity: Total equity of the company
/// - debt: Total debt of the company
/// - return_on_equity: Expected return on equity of the company
/// - return_on_debt: Expected return on debt of the company
/// - corporate_tax: Corporate tax rate on earnings after interest, EBT
/// 
/// # Output
/// - Weighted average cost of capital (WACC)
/// 
/// # LaTeX Formula
/// - r_{A} = [r_{D}(1-T_{c})\\frac{D}{E+D}] + [r_{E}\\frac{E}{E+D}]
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Weighted_average_cost_of_capital
/// - Original Source: N/A
pub fn weighted_average_cost_of_capital(equity: f64, debt: f64, return_on_equity: f64, return_on_debt: f64, corporate_tax: f64) -> f64 {
    return_on_debt*(1.0-corporate_tax)*debt/(debt+equity) + return_on_equity*equity/(debt+equity)
}


/// # Description
/// Computes expected return on the equity (ROE) portion of the company.\n
/// ROE = Return on Assets + (Return on Assets - Return on Debt * (1 - Corporate Tax Rate)) * Debt / Equity
/// 
/// # Input
/// - equity: Total equity of the company
/// - debt: Total debt of the company
/// - return_on_assets: Return on all assets (WACC) of the company
/// - return_on_debt: Expected return on debt of the company
/// - corporate_tax: Corporate tax rate on earnings after interest, EBT
/// 
/// # Output
/// - Expected return on equity (ROE)
/// 
/// # LaTeX Formula
/// - r_{E} = r_{A} + (r_{A}-r_{D}(1-T_{c}))\\frac{D}{E}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Return_on_equity
/// - Original Source: N/A
pub fn expected_return_on_equity(equity: f64, debt: f64, return_on_assets: f64, return_on_debt: f64, corporate_tax: f64) -> f64 {
    return_on_assets + (return_on_assets - return_on_debt*(1.0-corporate_tax)) * (debt/equity)
}


/// # Description
/// Unlevered beta, which is the systematic risk of the company's assets.\n
/// Unlevered Beta = (Debt / (Debt+Equity) * Beta of Debt) + (Equity / (Debt+Equity) * Beta of Equity)
/// 
/// # Input
/// - equity: Total equity of the company
/// - debt: Total debt of the company
/// - beta_equity: Levered beta of the company
/// - beta_debt: Beta debt of the company
/// 
/// # Output
/// - Unlevered beta
/// 
/// # LaTeX Formula
/// - \\beta_{A} = [\\beta_{D}\\frac{D}{E+D}] + [\\beta_{E}\\frac{E}{E+D}]
pub fn unlevered_beta(equity: f64, debt: f64, beta_equity: f64, beta_debt: f64) -> f64 {
    beta_debt*debt/(debt+equity) + beta_equity*equity/(debt+equity)
}


/// # Description
/// Levered beta, which is the equity-only beta of the company.\n
/// Levered Beta = Beta of Assets + (Beta of Assets - Beta of Debt) * (Debt / Equity)
/// 
/// # Input
/// - equity: Total equity of the company
/// - debt: Total debt of the company
/// - beta_assets: Unlevered beta of the company
/// - beta_debt: Beta debt of the company
/// 
/// # Output
/// - Levered beta
/// 
/// # LaTeX Formula
/// - \\beta_{E} = \\beta_{A} + (\\beta_{A} - \\beta_{D})\\frac{D}{E}
pub fn levered_beta(equity: f64, debt: f64, beta_assets: f64, beta_debt: f64) -> f64 {
    beta_assets + (beta_assets - beta_debt)*debt/equity
}


/// # Description
/// Calculates an advantage of debt financing for a company as opposed to equity financing from perspective of tax optimization.\n
/// Relative Tax Advantage of Debt = (1 - Personal Tax on Interest Income) / ((1 - Effective Personal Tax) * (1 - Corporate Tax))
/// 
/// # Input
/// - corporate_tax: Corporate tax rate applied to a company after debt payout
/// - personal_tax: Personal tax rate on a interest income
/// - effective_personal_tax: Effective tax rate on equity income comprising personal tax on dividend income and personal tax on capital gains income
/// 
/// # Output
/// - Relative tax advantage of debt ratio
/// 
/// # LaTeX Formula
/// - \\textit{Relative Tax Advantage of Debt} = \\frac{1-T_{p}}{(1-T_{pE})(1-T_{c})}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Tax_benefits_of_debt
/// - Original Source: N/A
pub fn relative_tax_advantage_of_debt(corporate_tax: f64, personal_tax: f64, effective_personal_tax: f64) -> f64 {
    (1.0-personal_tax) / ((1.0-effective_personal_tax) * (1.0-corporate_tax))
}