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
pub fn dol(quantity_of_goods: f64, price_per_unit: f64, variable_cost_per_unit: f64, total_fixed_cost: f64) -> f64 {
    let denominator: f64 = quantity_of_goods * (price_per_unit - variable_cost_per_unit) - total_fixed_cost;
    1.0 + total_fixed_cost / denominator
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
pub fn earnings_per_share(net_income: f64, preferred_dividends: f64, n_common_shares_outstanding: usize) -> f64 {
    (net_income - preferred_dividends) / n_common_shares_outstanding as f64
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
pub fn pe_ratio(share_price: f64, eps: f64) -> f64 {
    share_price / eps
}


/// Valuation metric for determining the relative trade-off between the price of a stock, the earnings generated per share (i.e., EPS),
/// and the company's expected growth.
/// 
/// PEG Ratio = (Share Price / Earnings per Share) / EPS Growth
/// 
/// # Input
/// - `share_price`: Share price of the company
/// - `eps`: Earnings per share of the company
/// - `eps_growth`: Growth rate of EPS in the specified time windoes (e.g., annual)
/// 
/// # Ouput
/// - PEG ratio
/// 
/// # LaTeX Formula
/// - PEG = \\frac{P/EPS}{EPS_{\\text{growth rate}}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/PEG_ratio>
/// - Original Source: N/A
pub fn peg_ratio(share_price: f64, eps: f64, eps_growth: f64) -> f64 {
    pe_ratio(share_price, eps) / eps_growth
}


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
pub fn pb_ratio(market_cap: f64, book_value: f64) -> f64 {
    market_cap / book_value
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
pub fn dividend_yield(share_price: f64, dividend: f64) -> f64 {
    100.0 * dividend / share_price
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
/// - `revenue`: Total amount of income generated by the sale of goods and services related to the primary operations of a business.
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
pub fn ev_to_revenue(market_cap: f64, total_debt: f64, cash: f64, revenue: f64) -> f64 {
    enterprise_value(market_cap, total_debt, cash) / revenue
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
pub fn ev_to_ebitda(market_cap: f64, total_debt: f64, cash: f64, ebitda: f64) -> f64 {
    enterprise_value(market_cap, total_debt, cash) / ebitda
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


/// Liquidity ratio that measures whether a firm has enough resources to meet its short-term obligations.
/// 
/// Current Ratio = Current Assets / Current Liabilities
/// 
/// # Input
/// - `current_assets`: Asset that can reasonably be expected to be sold, consumed, or exhausted through the normal operations of
/// a business within the current fiscal year, operating cycle, or financial year
/// - `current_liabilities`: Liabilities of a business that are expected to be settled in cash within one fiscal year or
/// the firm's operating cycle, whichever is longer
/// 
/// # Output
/// - Current ratio
/// 
/// # LaTeX Formula
/// \\text{Current Ratio} = \\frac{\\text{Current Assets}}{\\text{Current Liabilities}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Current_ratio>
/// - Original Source: N/A
pub fn current_ratio(current_assets: f64, current_liabilities: f64) -> f64 {
    current_assets / current_liabilities
}


/// Liquidity ratio that measures the ability of a company to use near-cash assets (or 'quick' assets
/// to extinguish or retire current liabilities immediately.
/// 
/// Quick Ratio = (Current Assets - Inventories) / Current Liabilities
/// 
/// # Input
/// - `current_assets`: Asset that can reasonably be expected to be sold, consumed, or exhausted through the normal operations of
/// a business within the current fiscal year, operating cycle, or financial year
/// - `inventories`: Raw materials used in production as well as the goods produced that are available for sale
/// - `current_liabilities`: Liabilities of a business that are expected to be settled in cash within one fiscal year or
/// the firm's operating cycle, whichever is longer
/// 
/// # Output
/// - Quick ratio (i.e., acid-test ratio)
/// 
/// # LaTeX Formula
/// - \\text{Quick Ratio} = \\frac{\\text{Current Assets - Inventories}}{\\text{Current Liabilities}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Quick_ratio>
/// - Original Source: N/A
pub fn quick_ratio(current_assets: f64, inventories: f64, current_liabilities: f64) -> f64 {
    (current_assets - inventories) / current_liabilities
}


/// Financial ratio that determines the value of incremental sales, which guides pricing and promotion decisions.
/// 
/// Gross Margin = (Revenue - CoGS) / Revenue
/// 
/// # Input
/// - `revenue`: Total amount of income generated by the sale of goods and services related to the primary operations of a business.
/// - `cost_of_goods_sold`: Carrying value of goods sold during a particular period
/// 
/// # Output
/// - Gross margin
/// 
/// # LaTeX Formula
/// - \\text{Gross Margin} = \\frac{\\text{Revenue - CoGS}}{\\text{Revenue}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Gross_margin>
/// - Original Source: N/A
pub fn gross_margin(revenue: f64, cost_of_goods_sold: f64) -> f64 {
    (revenue - cost_of_goods_sold) / revenue
}


/// Financial ratio that measures the profitability of ventures after accounting for all costs.
/// 
/// Operating Margin = Operating Income / Revenue
/// 
/// # Input
/// - `operating_income`: Profit a company makes from its core business operations after deducting all operating expenses
/// - `revenue`: Total amount of income generated by the sale of goods and services related to the primary operations of a business.
/// 
/// # Output
/// - Operating margin
/// 
/// # LaTeX Formula
/// - \\text{Operating Margin} = \\frac{\\text{Operating Income}}{\\text{Revenue}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Operating_margin>
/// - Original Source: N/A
pub fn operating_margin(operating_income: f64, revenue: f64) -> f64 {
    operating_income / revenue
}


/// Financial ratio that measures the percentage of profit earned by a company in relation to its revenue.
/// 
/// Net Profit Margin = Net Income / Revenue
/// 
/// # Input
/// - `net_income`: Company's income minus cost of goods sold, expenses, depreciation and amortization, interest,
/// and taxes, and other expenses for an accounting period
/// - `revenue`: Total amount of income generated by the sale of goods and services related to the primary operations of a business.
/// 
/// # Output
/// - Net profit margin
/// 
/// # LaTeX Formula
/// - \\text{Net Profit Margin} = \\frac{\\text{Net Income}}{\\text{Revenue}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Profit_margin
/// - Original Source: N/A
pub fn net_profit_margin(net_income: f64, revenue: f64) -> f64 {
    net_income / revenue
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
pub fn cost_of_equity_capital(share_price: f64, expected_dividend: f64, expected_share_price: f64) -> f64 {
    (expected_dividend + expected_share_price - share_price) / share_price
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
pub fn payout_ratio(dividend_per_share: f64, earnings_per_share: f64) -> f64 {
    dividend_per_share / earnings_per_share
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
pub fn plowback_ratio(dividend_per_share: f64, earnings_per_share: f64) -> f64 {
    1.0 - dividend_per_share / earnings_per_share
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
pub fn altman_z_score(ebit: f64, total_assets: f64, sales: f64, equity: f64, total_liabilities: f64, retained_earnings: f64, working_capital: f64) -> f64 {
    3.3*ebit/total_assets + sales/total_assets + 0.6*equity/total_liabilities + 1.4*retained_earnings/total_assets + 1.2*working_capital/total_assets
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
pub fn weighted_average_cost_of_capital(equity: f64, debt: f64, return_on_equity: f64, return_on_debt: f64, corporate_tax: f64) -> f64 {
    let capital_structure: f64 = debt + equity;
    return_on_debt*(1.0-corporate_tax)*debt/capital_structure + return_on_equity*equity/capital_structure
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
pub fn expected_return_on_equity(equity: f64, debt: f64, return_on_assets: f64, return_on_debt: f64, corporate_tax: f64) -> f64 {
    return_on_assets + (return_on_assets - return_on_debt*(1.0-corporate_tax)) * (debt/equity)
}


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
pub fn return_on_equity(net_income: f64, equity: f64) -> f64 {
    net_income / equity
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
pub fn return_on_equity_dupont(net_profit_margin: f64, total_asset_turnover: f64, total_assets: f64, total_shareholders_equity: f64) -> f64 {
    let financial_leverage: f64 = total_assets / total_shareholders_equity;
    net_profit_margin * total_asset_turnover * financial_leverage
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
pub fn return_on_assets(net_income: f64, total_assets: f64) -> f64 {
    net_income / total_assets
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
pub fn return_on_investment(gain_from_investment: f64, cost_of_investment: f64) -> f64 {
    (gain_from_investment - cost_of_investment) / cost_of_investment
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
pub fn debt_to_equity(debt: f64, equity: f64) -> f64 {
    debt / equity
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
pub fn unlevered_beta(equity: f64, debt: f64, beta_equity: f64, beta_debt: f64) -> f64 {
    let capital_structure: f64 = debt + equity;
    beta_debt*debt/capital_structure + beta_equity*equity/capital_structure
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
pub fn levered_beta(equity: f64, debt: f64, beta_assets: f64, beta_debt: f64) -> f64 {
    beta_assets + (beta_assets - beta_debt)*debt/equity
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
pub fn relative_tax_advantage_of_debt(corporate_tax: f64, personal_tax: f64, effective_personal_tax: f64) -> f64 {
    let denominator: f64 = (1.0-effective_personal_tax) * (1.0-corporate_tax);
    (1.0-personal_tax) / denominator
}


/// Interest coverage ratio (i.e., Times-Interest Earned) is a measure of company's ability to honour its debt payments.
/// 
/// Note: EBITDA can be used instead of EBIT.
/// 
/// Interest Coverage Ratio = EBIT / Interest Expense
/// 
/// # Input
/// - `EBIT`: Earnings before interest and taxes
/// - `interest_expense`: Cost of borrowing money from financial institutions incurred by the company
/// 
/// # Output
/// - Interest coverage ratio
/// 
/// # LaTeX Formula
/// - \\textit{Interest Coverage Ratio} = \\frac{EBIT}{\\textit{Interest Expense}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Times_interest_earned>
/// - Original Source: N/A
pub fn interest_coverage_ratio(ebit: f64, interest_expense: f64) -> f64 {
    ebit / interest_expense
}


/// Debt service covergae ratio (i.e., Debt coverage Ratio) is a measure of company's ability to generate sufficient cash to cover its debt obligations,
/// including interest, principal, and lease payments.
/// 
/// DSCR = Operating Income / Current Debt Obligations
/// 
/// # Input
/// - `operating_income`: Profit a company makes from its core business operations after deducting all operating expenses
/// - `current_debt_obligations`: Current debt obligations, including any interest, principal, sinking funds, and lease payments
/// that are due over the next financial period (e.g., year)
/// 
/// # Output
/// - Debt service coverage ratio
/// 
/// # LaTeX Formula
/// - \\textit{DSCR} = \\frac{\\textit{Operating Income}}{\\textit{Current Debt Obligations}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Debt_service_coverage_ratio>
/// - Original Source: N/A
pub fn debt_service_coverage_ratio(operating_income: f64, current_debt_obligations: f64) -> f64 {
    operating_income / current_debt_obligations
}


/// Asset coverage ratio is a measure of how well a company can repay its debts by sellingor liquidating its assets.
/// 
/// Asset Coverage Ratio = (Total Assets - Current Liabilities) / Total Debt
/// 
/// # Input
/// - `total_assets`: Total assets of the company
/// - `current_liabilities`: Liabilities of a business that are expected to be settled in cash within one fiscal year or
/// the firm's operating cycle, whichever is longer
/// - `total_debt`: Total debt of the company
/// 
/// # Output
/// - Asset coverage ratio
/// 
/// # LaTeX Formula
/// - \\textit{Asset Coverage Ratio} = \\frac{\\textit{Total Assets} - \\textit{Current Liabilities}}{\\textit{Total Debt}}
/// 
/// # Links
/// - Wikipedia: N/A
/// - Original Source: N/A
pub fn asset_coverage_ratio(total_assets: f64, current_liabilities: f64, total_debt: f64) -> f64 {
    (total_assets - current_liabilities) / total_debt
}


/// Liquidity ratio is a measure a company's ability to repay short-term creditors out of its total cash.
/// 
/// Liquidity Ratio = Liquid Assets / Current Liabilities
/// 
/// # Input
/// - `Liquid_assets`: Cash of the company on the balance sheet
/// - `current_liabilities`: Liabilities of a business that are expected to be settled in cash within one fiscal year or
/// the firm's operating cycle, whichever is longer
/// 
/// # Output
/// - Liquidity ratio
/// 
/// # LaTeX Formula
/// - \\textit{Liquidity Ratio} = \\frac{\\textit{Liquid Assets}}{\\textit{Current Liabilities}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Liquidity_ratio>
/// - Original Source: N/A
pub fn liquidity_ratio(liquid_assets: f64, current_liabilities: f64) -> f64 {
    liquid_assets / current_liabilities
}


/// Cash flow to debt ratio is a measure of how long it would take the company to repay all its debt using only cash flow from operations.
/// 
/// CF/D = Operating Cash Flow / Total Debt
/// 
/// # Input
/// - `operating_cash_flow`: Cash generated from core business activities
/// - `total_debt`: Total debt of the company
/// 
/// # Output
/// - Cash flow to debt ratio
/// 
/// # LaTeX Formula
/// - \\textit{CF/D} = \\frac{\\textit{Operating Cash Flow}}{\\textit{Total Debt}}
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Cash-flow-to-debt_ratio>
/// - Original Source: N/A
pub fn cash_flow_to_debt(operating_cash_flow: f64, total_debt: f64) -> f64 {
    operating_cash_flow / total_debt
}