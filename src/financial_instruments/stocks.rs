use ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::{compare_len, FeatureCollection, time_value_utils::{Compounding, CompoundingType, Perpetuity}};
use crate::financial_instruments::{FinancialInstrument, FinancialInstrumentId};
use crate::corporate_finance;
use crate::portfolio_applications::{AssetHistData, PortfolioInstrument};
use crate::statistics::{LinearRegressionSettings, LinearRegressionResult, LinearRegressionAnalysis};
use crate::stochastic_processes::StochasticProcess;


#[derive(Clone, Copy, Debug)]
/// Choice of how to quote the values returned by functions.
pub enum QuoteValues {
    PerShare,
    Total,
}


#[derive(Debug)]
/// Type of model to use for valuation of stock price.
pub enum StockValuationType {
    DividendDiscountModel,
    ValuationByComparables { params: ValuationByComparablesParams },
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Parameters for valuation by comparables.
pub struct ValuationByComparablesParams {
    /// Array of valuations of the companies
    valuations: Array1<f64>,
    /// Array of P/E ratios of the companies
    pe_ratios: Option<Array1<f64>>,
    /// Array of P/B ratios of the companies
    pb_ratios: Option<Array1<f64>>,
    /// Array of EV/EBITDA ratios of the companies
    ev_to_ebitda: Option<Array1<f64>>,
}

impl ValuationByComparablesParams {
    /// Parameters for valuation by comparables.
    ///
    /// # Input
    /// - `valuations`: Array of valuations of the companies
    /// - `pe_ratios`: Array of P/E ratios of the companies
    /// - `pb_ratios`: Array of P/B ratios of the companies
    /// - `ev_to_ebitda`: Array of EV/EBITDA ratios of the companies
    ///
    /// # Errors
    /// - Returns an error if the number of datapoints (i.e., number of points in arrays) is less than `5`.
    pub fn build(valuations: Array1<f64>, pe_ratios: Option<Array1<f64>>, pb_ratios: Option<Array1<f64>>, ev_to_ebitda: Option<Array1<f64>>) -> Result<Self, DigiFiError> {
        if valuations.len() < 5 {
            return Err(DigiFiError::ValidationError { title: Self::error_title(), details: "Minimum number of datapoints required is `5`.".to_owned(), });
        }
        Self::validate_array(&valuations, &pe_ratios, "pe_ratios")?;
        Self::validate_array(&valuations, &pb_ratios, "pb_ratios")?;
        Self::validate_array(&valuations, &ev_to_ebitda, "ev_to_ebitda")?;
        Ok(Self { valuations, pe_ratios, pb_ratios, ev_to_ebitda })
    }

    /// Validates an array against the valuations array.
    fn validate_array(valuations: &Array1<f64>, array: &Option<Array1<f64>>, array_name: &str) -> Result<(), DigiFiError> {
        match array {
            Some(v) => {
                compare_len(&valuations.iter(), &v.iter(), "valuations", array_name)?;
                Ok(())
            },
            None => Ok(()),
        }
    }

    /// Returns the number of series that are not `None`.
    pub fn n_parameters(&self) -> usize {
        1 + self.pe_ratios.is_some() as usize + self.pb_ratios.is_some() as usize + self.ev_to_ebitda.is_some() as usize
    }

    /// Returns an array of stocks' valuations.
    pub fn valuations(&self) -> &Array1<f64> {
        &self.valuations
    }

    /// Returns an array of stocks' P/E raatios.
    pub fn pe_ratios(&self) -> &Option<Array1<f64>> {
        &self.pe_ratios
    }

    /// Returns an array of stock's P/B ratios.
    pub fn pb_ratios(&self) -> &Option<Array1<f64>> {
        &self.pb_ratios
    }

    /// Returns an array of stocks' EV/EBITDA ratios.
    pub fn ev_to_ebitda(&self) ->& Option<Array1<f64>> {
        &self.ev_to_ebitda
    }
}

impl ErrorTitle for ValuationByComparablesParams {
    fn error_title() -> String {
        String::from("Valuation by Comparables Params")
    }
}


/// Stock financial instrument and its methods.
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Stock>
/// - Original Source: N/A
///
/// # Examples
///
/// 1. Valuation using Dividend Discount Model
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, Time, CompoundingType};
/// use digifi::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass};
/// use digifi::financial_instruments::{QuoteValues, StockValuationType, ValuationByComparablesParams, Stock};
/// use digifi::portfolio_applications::AssetHistData;
///
/// // Stock definition
/// let compounding_type: CompoundingType = CompoundingType::Continuous;
/// let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
///     instrument_type: FinancialInstrumentType::CashInstrument, asset_class: AssetClass::EquityBasedInstrument, identifier: String::from("32198407128904"),
/// };
/// let asset_historical_data: AssetHistData = AssetHistData::build(
///     Array1::from_vec(vec![0.4, 0.5]), Array1::from_vec(vec![0.0, 0.0]), Time::new(Array1::from_vec(vec![0.0, 1.0]))
/// ).unwrap();
/// let stock: Stock = Stock::new(
///     100.0, 1_000_000, 3.0, 2.5, QuoteValues::PerShare, 99.0, compounding_type, 0.0, StockValuationType::DividendDiscountModel, Some(10.0), Some(3.0), Some(6.5),
///     5.0, financial_instrument_id, asset_historical_data, None
/// );
///
/// // Theoretical value
/// let r: f64 = 3.0 / 100.0 + 0.0;
/// let perpetuity_pv: f64 = 3.0 * 0.0_f64.exp() / ((r - 0.0).exp() - 1.0);
/// assert!((stock.present_value().unwrap() - perpetuity_pv).abs() < TEST_ACCURACY);
/// ```
///
/// 2. Valuation using Valuation-by-Comparables
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, Time, CompoundingType};
/// use digifi::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass};
/// use digifi::financial_instruments::{QuoteValues, StockValuationType, ValuationByComparablesParams, Stock};
/// use digifi::portfolio_applications::AssetHistData;
///
/// // Stock definition
/// let compounding_type: CompoundingType = CompoundingType::Continuous;
/// let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
///     instrument_type: FinancialInstrumentType::CashInstrument, asset_class: AssetClass::EquityBasedInstrument,
///     identifier: String::from("32198407128904"),
/// };
/// let asset_historical_data: AssetHistData = AssetHistData::build(
///     Array1::from_vec(vec![0.4, 0.5]), Array1::from_vec(vec![0.0, 0.0]), Time::new(Array1::from_vec(vec![0.0, 1.0]))
/// ).unwrap();
///
/// // Valuation parameters
/// let valuations: Array1<f64> = Array1::from_vec(vec![200_000_000.0, 1_000_000_000.0, 3_000_000_000.0, 500_000_000.0, 1_500_000_000.0]);
/// let pe_ratios: Array1<f64> = Array1::from_vec(vec![20.0, 8.0, 9.0, 15.0, 11.0]);
/// let pb_ratios: Array1<f64> = Array1::from_vec(vec![7.0, 2.0, 4.0, 10.0, 5.0]);
/// let ev_to_ebitda: Array1<f64> = Array1::from_vec(vec![10.0, 5.0, 6.0, 7.0, 6.0]);
/// let valuation_params: ValuationByComparablesParams = ValuationByComparablesParams::build(
///     valuations, Some(pe_ratios), Some(pb_ratios), Some(ev_to_ebitda)
/// ).unwrap();
/// 
/// let stock: Stock = Stock::new(
///     100.0, 1_000_000, 3.0, 2.5, QuoteValues::Total, 99.0, compounding_type, 0.0, StockValuationType::ValuationByComparables { params: valuation_params },
///     Some(10.0), Some(3.0), Some(6.5), 5.0, financial_instrument_id, asset_historical_data, None
/// );
/// assert!(1_000_000_000.0 < stock.present_value().unwrap());
/// ````
pub struct Stock {
    /// Price per share
    price_per_share: f64,
    /// Number of shares outstanding
    n_shares_outstanding: f64,
    /// Dividend per share
    dividend_per_share: f64,
    /// Earnings per share (EPS)
    earnings_per_share: f64,
    /// Determines how output of Stock classs methods will be quoted (i.e., `PerShare` - for values per share, `Total` - for total values)
    quote_values: QuoteValues,
    /// Initial price at which the stock is purchased
    initial_price: f64,
    /// Compounding type used to discount cashflows
    compounding_type: CompoundingType,
    /// Growth rate of the dividend payouts
    dividend_growth_rate: f64,
    /// Method for performing stock valuation (i.e., `DividendDiscountModel` or `ValuationByComparables`)
    stock_valuation_type: StockValuationType,
    /// Current P/E ratio of the stock (Used for `ValuationByComparables`)
    pe: Option<f64>,
    /// Current P/B ratio of the stock (Used for `ValuationByComparables`)
    pb: Option<f64>,
    /// Current EV/EBITDA ratio of the stock (Used for `ValuationByComparables`)
    ev_to_ebitda: Option<f64>,
    /// Final time step for stochastic simulations and computation of future value of stock
    t_f: f64,
    /// Parameters for defining regulatory categorization of an instrument
    financial_instrument_id: FinancialInstrumentId,
    /// Time series asset data
    asset_historical_data: AssetHistData,
    /// Stochastic model to use for price paths generation
    stochastic_model: Option<Box<dyn StochasticProcess>>,
}

impl Stock {
    /// Creates a new `Stock` instance.
    ///
    /// # Input:
    /// - `price_per_share`: Price per share
    /// - `n_shares_outstanding`: Number of shares outstanding
    /// - `dividend_per_share`: Dividend per share
    /// - `earnings_per_share`: Earnings per share (EPS)
    /// - `quote_values`: Determines how output of Stock classs methods will be quoted (i.e., `PerShare` - for values per share, `Total` - for total values)
    /// - `initial_price`: Initial price at which the stock is purchased
    /// - `compounding_type`: Compounding type used to discount cashflows
    /// - `dividend_growth_rate`: Growth rate of the dividend payouts
    /// - `stock_valuation_type`: Type of valuation to use to compute the present value  of the stock (i.e., to compute stock price or market cap)
    /// - `pe`: PE ratio of the stock (Required for valuation by comparables)
    /// - `pb` PB ratio of the stock (Required for valuation by comparables)
    /// - `ev_to_ebitda`: EV/EBITDA ratio of the stock (Required for valuation by comparables)
    /// - `t_f`: Maturity of investment (Used to compute the future value of the stock price or market cap)
    /// - `financial_instrument_id`: Parameters for defining regulatory categorization of an instrument
    /// - `asset_historical_data`: Time series asset data
    /// - `stochastic_model`: Stochastic model to use for price paths generation
    pub fn new(
        price_per_share: f64, n_shares_outstanding: i32, dividend_per_share: f64, earnings_per_share: f64, quote_values: QuoteValues, initial_price: f64,
        compounding_type: CompoundingType, dividend_growth_rate: f64, stock_valuation_type: StockValuationType, pe: Option<f64>, pb: Option<f64>,
        ev_to_ebitda: Option<f64>, t_f: f64, financial_instrument_id: FinancialInstrumentId, asset_historical_data: AssetHistData,
        stochastic_model: Option<Box<dyn StochasticProcess>>
    ) -> Self {
        Self {
            price_per_share, n_shares_outstanding: n_shares_outstanding as f64, dividend_per_share, earnings_per_share, quote_values, initial_price,
            compounding_type, dividend_growth_rate, stock_valuation_type, pe, pb, ev_to_ebitda, t_f, financial_instrument_id, asset_historical_data,
            stochastic_model,
        }
    }

    fn apply_value_quotation_type(&self, value_per_share: f64) -> f64 {
        match self.quote_values {
            QuoteValues::PerShare => value_per_share,
            QuoteValues::Total => value_per_share * self.n_shares_outstanding,
        }
    }
    
    /// Description
    /// Returns current price per share.
    pub fn share_price(&self) -> f64 {
        self.price_per_share
    }

    /// Monetary value of earnings per outstanding share of common stock for a company during a defined period of time.
    /// 
    /// EPS = (Net Income - Preferred Dividends) / Number of Common Shares Outstanding
    /// 
    /// # Input
    /// - `net_income`: Net income
    /// - `preferred_dividends`: Total dividends paid to the holders of the preferred stock
    /// - `n_common_shares_outstanding`: Number of common shares outstanding
    /// - `in_place`: Update the earnings per share of the stuct instance with the result
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
    pub fn trailing_eps(&mut self, net_income: f64, preferred_dividends: f64, n_common_shares_outstanding: usize, in_place: bool) -> f64 {
        let trailing_eps: f64 = corporate_finance::earnings_per_share(net_income, preferred_dividends, n_common_shares_outstanding);
        if in_place {
            self.earnings_per_share = trailing_eps;
        }
        trailing_eps
    }

    /// The ratio of market price to earnings.
    /// 
    /// Price-to-Earnings Ratio = Share Price / Earnings per Share
    /// 
    /// # Input
    /// - `share_price`: Share price of the company
    /// - `eps`: Earnings per share of the company
    /// - `in_place`: Update the P/E of the stuct instance with the result
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
    pub fn trailing_pe(&mut self, in_place: bool) -> f64 {
        let trailing_pe: f64 = corporate_finance::pe_ratio(self.price_per_share, self.earnings_per_share);
        if in_place {
            self.pe = Some(trailing_pe);
        }
        trailing_pe
    }

    /// Valuation metric for determining the relative trade-off between the price of a stock, the earnings generated per share (i.e., EPS),
    /// and the company's expected growth.
    /// 
    /// PEG Ratio = (Share Price / Earnings per Share) / EPS Growth
    /// 
    /// # Input
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
    pub fn peg(&self, eps_growth: f64) -> f64 {
        corporate_finance::peg_ratio(self.price_per_share, self.earnings_per_share, eps_growth)
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
    pub fn pb(&mut self, market_cap: f64, assets: f64, liabilities: f64, in_place: bool) -> f64 {
        let book_value: f64 = corporate_finance::book_value(assets, liabilities);
        let pb: f64 = corporate_finance::pb_ratio(market_cap, book_value);
        if in_place {
            self.pb = Some(pb);
        }
        pb
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
    pub fn enterprise_value(&self, market_cap: f64, total_debt: f64, cash: f64) -> f64 {
        corporate_finance::enterprise_value(market_cap, total_debt, cash)
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
    pub fn ev_to_revenue(&self, market_cap: f64, total_debt: f64, cash: f64, revenue: f64) -> f64 {
        corporate_finance::ev_to_revenue(market_cap, total_debt, cash, revenue)
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
    pub fn ev_to_ebitda(&mut self, market_cap: f64, total_debt: f64, cash: f64, ebitda: f64, in_place: bool) -> f64 {
        let ev_to_ebitda: f64 = corporate_finance::ev_to_ebitda(market_cap, total_debt, cash, ebitda);
        if in_place {
            self.ev_to_ebitda = Some(ev_to_ebitda);
        }
        ev_to_ebitda
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
    pub fn book_value(&self, assets: f64, liabilities: f64) -> f64 {
        corporate_finance::book_value(assets, liabilities)
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
    pub fn current_ratio(&self, current_assets: f64, current_liabilities: f64) -> f64 {
        corporate_finance::current_ratio(current_assets, current_liabilities)
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
    pub fn quick_ratio(&self, current_assets: f64, inventories: f64, current_liabilities: f64) -> f64 {
        corporate_finance::quick_ratio(current_assets, inventories, current_liabilities)
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
    pub fn gross_margin(&self, revenue: f64, cost_of_goods_sold: f64) -> f64 {
        corporate_finance::gross_margin(revenue, cost_of_goods_sold)
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
    pub fn operating_margin(&self, operating_income: f64, revenue: f64) -> f64 {
        corporate_finance::operating_margin(operating_income, revenue)
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
    pub fn net_profit_margin(&self, net_income: f64, revenue: f64) -> f64 {
        corporate_finance::net_profit_margin(net_income, revenue)
    }

    /// Computes the cost of equity capital (Market capitalization rate).
    ///
    /// Cost of Equity Capital = (Expected Dividend / Current Share Price) + Sustainable Growth Rate
    ///
    /// Note: It is assumed that the sustainable growth rate is the dividend growth rate.
    ///
    /// # Input
    /// - `expected_dividend`: Expected dividend per share (If none provided, the dividend_per_share from the definition of the instance will be used instead)
    ///
    /// # Output
    /// - Cost of equity capital (Market capitalization rate)
    ///
    /// # LaTeX Formula
    /// - r = \\frac{D_{E}}{P} + g
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Cost_of_capital>
    /// - Original Source: <https://www.jstor.org/stable/1809766>
    pub fn cost_of_equity_capital(&self, expected_dividend: Option<f64>) -> f64 {
        let expected_dividend: f64 = match expected_dividend {
            Some(v) => v,
            None => self.dividend_per_share,
        };
        expected_dividend / self.price_per_share + self.dividend_growth_rate
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
    /// LaTeX Formula
    /// - ROE = \\frac{\\textit{Net Income}}{Equity}
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Return_on_equity>
    /// - Original SOurce: N/A
    pub fn return_on_equity(&self, net_income: f64, equity: f64) -> f64 {
        corporate_finance::return_on_equity(net_income, equity)
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
    pub fn return_on_assets(&self, net_income: f64, total_assets: f64) -> f64 {
        corporate_finance::return_on_assets(net_income, total_assets)
    }

    /// Performance measure used to evaluate the efficiency or profitability of an investment or compare the efficiency
    /// of a number of different investments. ROI tries to directly measure the amount of return on a particular investment,
    /// relative to the investment’s cost.
    /// 
    /// ROI = (Revenue - Cost of Goods Sold) / Cost of Goods Sold
    /// 
    /// # Input
    /// - `revenue`: Total amount of income generated by the sale of goods and services related to the primary operations of a business.
    /// - `cost_of_goods_sold`: Carrying value of goods sold during a particular period
    /// 
    /// # Output
    /// - Return on investment (ROI)
    /// 
    /// # LaTeX Formula
    /// - ROI = \\frac{Revenue - \\textit{Cost of Goods Sold}}{\\textit{Cost of Goods Sold}}
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Return_on_investment>
    /// - Original Source: N/A
    pub fn return_on_investment(&self, revenue: f64, cost_of_goods_sold: f64) -> f64 {
        corporate_finance::return_on_investment(revenue, cost_of_goods_sold)
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
    pub fn debt_to_equity(&self, debt: f64, equity: f64) -> f64 {
        corporate_finance::debt_to_equity(debt, equity)
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
    pub fn interest_coverage_ratio(&self, ebit: f64, interest_expense: f64) -> f64 {
        corporate_finance::interest_coverage_ratio(ebit, interest_expense)
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
    pub fn debt_service_coverage_ratio(&self, operating_income: f64, current_debt_obligations: f64) -> f64 {
        corporate_finance::debt_service_coverage_ratio(operating_income, current_debt_obligations)
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
    pub fn asset_coverage_ratio(&self, total_assets: f64, current_liabilities: f64, total_debt: f64) -> f64 {
        corporate_finance::asset_coverage_ratio(total_assets, current_liabilities, total_debt)
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
    pub fn liquidity_ratio(&self, liquid_assets: f64, current_liabilities: f64) -> f64 {
        corporate_finance::liquidity_ratio(liquid_assets, current_liabilities)
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
    pub fn cash_flow_to_debt(&self, operating_cash_flow: f64, total_debt: f64) -> f64 {
        corporate_finance::cash_flow_to_debt(operating_cash_flow, total_debt)
    }

    /// Dividend discount model evaluating the price of the stock.
    ///
    /// Note: This model assumes that the dividend cashflow grows with the rate dividend_growth_rate.
    ///
    /// # Input
    /// - `expected_dividend`: Expected dividend per share (If none provided, the dividend_per_share from the definition of the instance will be used instead)
    ///
    /// # Output
    /// - Present value of the stock based on the dividend discount model
    ///
    /// # LaTeX Formula
    /// - \\textit{PV(Share)} = \\sum^{\\infty}_{t=1} \\frac{D_{t}}{(1 + r)^{t}}
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Dividend_discount_model>
    /// - Original Source: <https://doi.org/10.2307/1927792>
    pub fn dividend_discount_model(&self, expected_dividend: Option<f64>) -> Result<f64, DigiFiError> {
        let cost_of_equity_capital: f64 = self.cost_of_equity_capital(expected_dividend);
        let dividend_perpetuity: Perpetuity = Perpetuity::build(
            self.dividend_per_share, cost_of_equity_capital, self.dividend_growth_rate, self.compounding_type.clone()
        )?;
        let pv: f64 = dividend_perpetuity.present_value();
        Ok(self.apply_value_quotation_type(pv))
    }

    /// Valuation of the stock by comparing its features to the features of similar stocks.
    ///
    /// # Input
    /// - `pe`: P/E ratio of the stock
    /// - `pb`: P/B ratio of the stock
    /// - `ev_to_ebitda`: EV/EBITDA ratio of the stock
    /// - `valuation_params`: Parameters of similar stocks that will be used for the valuation by compaison
    ///
    /// # Output
    /// - Valuation of the stock (Price per share)
    pub fn valuation_by_comparables(&self, pe: Option<f64>, pb: Option<f64>, ev_to_ebitda: Option<f64>, valuation_params: &ValuationByComparablesParams) -> Result<f64, DigiFiError> {
        // Closure that populates feature collection and the x vector that is needed for valuation prediction
        let add_feature = |feature: &Option<Array1<f64>>, multiple: &Option<f64>, fc: &mut FeatureCollection, x: &mut Vec<f64>, feature_name: &str, multiple_name: &str| -> Result<(), DigiFiError> {
            match (feature, multiple) {
                (Some(f), Some(m)) => {
                    fc.add_feature(f.iter(), feature_name)?;
                    x.push(*m);
                    Ok(())
                },
                (None, None) => Ok(()),
                _ => return Err(DigiFiError::ValidationError {
                    title: Self::error_title(),
                    details: format!("The argument `{}` is None, while the argument `{}` is not. They must be the same variant of `Option` enum.", feature_name, multiple_name),
                }),
            }
        };
        // Linear regression parameters
        let mut x: Vec<f64> = Vec::with_capacity(valuation_params.n_parameters());
        let mut fc: FeatureCollection = FeatureCollection::new();
        fc.add_constant = true;
        // fc.add_feature(valuation_params.valuations().iter(), "Valuation")?;
        add_feature(valuation_params.pe_ratios(), &pe, &mut fc, &mut x, "P/E Ratio", "P/E")?;
        add_feature(valuation_params.pb_ratios(), &pb, &mut fc, &mut x, "P/B Ratio", "P/B")?;
        add_feature(valuation_params.ev_to_ebitda(), &ev_to_ebitda, &mut fc, &mut x, "EV/EBITDA", "EV/EBITDA")?;
        x.push(1.0); // x for constant term alpha
        // Linear regression
        let lra_result: LinearRegressionResult = LinearRegressionAnalysis::new(LinearRegressionSettings::disable_all()).run(&fc, valuation_params.valuations())?;
        let valuation: f64 = lra_result.all_coefficients.dot(&Array1::from_vec(x));
        Ok(self.apply_value_quotation_type(valuation / self.n_shares_outstanding))
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
    pub fn payout_ratio(&self) -> f64 {
        corporate_finance::payout_ratio(self.dividend_per_share, self.earnings_per_share)
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
    pub fn plowback_ratio(&self) -> f64 {
        corporate_finance::plowback_ratio(self.dividend_per_share, self.earnings_per_share)
    }

    /// Computes dividend growth rate.
    ///
    /// Dividend Growth Rate = Plowback Ratio * ROE
    ///
    /// # Input
    /// - `plowback_ratio`: Plowback ratio of the stock
    /// - `roe`: Return on equity (ROE) of the stock
    /// - `in_place`: Update the dividend growth rate of the struct instance with the result
    ///
    /// # Output
    /// - Dividend growth rate
    ///
    /// # LaTeX Formula
    /// - \\textit{Dividend Growth Rate} = b*ROE
    pub fn dividend_growth_rate(&mut self, plowback_ratio: f64, roe: f64, in_place: bool) -> f64 {
        let g: f64 = plowback_ratio * roe;
        if in_place {
            self.dividend_growth_rate = g
        }
        g
    }

    /// Computes present value of growth opportunities (PVGO) which corresponds to the component of stock's valuation responsible for earnings growth.
    ///
    /// PVGO = Share Price - Earnings per Share / Cost of Equity Capital
    ///
    /// # Input
    /// - `expected_earnings`: Expected earnings per share (If none provided, the earnings_per_share from the definition of the instance will be used instead)
    /// - `expected_dividend`: Expected dividend per share (If none provided, the dividend_per_share from the definition of the instance will be used instead)
    ///
    /// # Output
    /// - Present value of growth opportunities (PVGO)
    ///
    /// # LaTeX Formula
    /// - PVGO = P - \\frac{E_{E}}{r}
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Present_value_of_growth_opportunities>
    /// - Original Source: N/A
    pub fn present_value_of_growth_opportunities(&self, expected_earnings: Option<f64>, expected_dividend: Option<f64>) -> f64 {
        let expected_earnings: f64 = expected_earnings.unwrap_or(self.earnings_per_share);
        let r: f64 = self.cost_of_equity_capital(expected_dividend);
        self.price_per_share - expected_earnings / r
    }
}

impl ErrorTitle for Stock {
    fn error_title() -> String {
        String::from("Stock")
    }
}

impl FinancialInstrument for Stock {

    /// Present value of the stock.
    /// 
    /// # Output
    /// - Present value of the stock
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Dividend_discount_model>, <https://en.wikipedia.org/wiki/Valuation_using_multiples>
    /// - Original Source: N/A
    fn present_value(&self) -> Result<f64, DigiFiError> {
        match &self.stock_valuation_type {
            StockValuationType::DividendDiscountModel => {
                Ok(self.dividend_discount_model(None)?)
            },
            StockValuationType::ValuationByComparables { params } => {
                self.valuation_by_comparables(self.pe, self.pb, self.ev_to_ebitda, params)
            },
        }
    }

    /// Net present value of the stock.
    ///
    /// # Output
    /// - Present value of the stock minus the initial price it took to purchase the stock
    fn net_present_value(&self) -> Result<f64, DigiFiError> {
        Ok(self.present_value()? - self.initial_price)
    }

    /// Future value of the stock.
    ///
    /// # Output
    /// - Future value of the stock at the given time (Computed from the present value of the stock)
    fn future_value(&self) -> Result<f64, DigiFiError> {
        let r: f64 = self.cost_of_equity_capital(None);
        let discount_term: Compounding = Compounding::new(r, &self.compounding_type);
        Ok(self.present_value()? / discount_term.compounding_term(self.t_f))
    }

    /// Returns asset's historical data.
    fn historical_data(&self) -> &AssetHistData {
        &self.asset_historical_data
    }

    /// Updates historical data of the asset with the newly generated data.
    fn update_historical_data(&mut self, new_data: &AssetHistData) -> () {
        self.asset_historical_data = new_data.clone();
    }

    /// Returns a mutable reference to the stochastic process that simulates price action.
    fn stochastic_model(&mut self) -> &mut Option<Box<dyn StochasticProcess>> {
        &mut self.stochastic_model
    }
}

impl PortfolioInstrument for Stock {

    fn asset_name(&self) -> String {
        self.financial_instrument_id.identifier.clone()
    }

    fn historical_data(&self) -> &AssetHistData {
        &self.asset_historical_data
    }
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::utilities::{TEST_ACCURACY, Time, time_value_utils::CompoundingType};
    use crate::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass};
    use crate::financial_instruments::stocks::{QuoteValues, StockValuationType, ValuationByComparablesParams, Stock};
    use crate::portfolio_applications::AssetHistData;

    #[test]
    fn unit_test_stock_dividend_discount_model() -> () {
        // Stock definition
        let compounding_type: CompoundingType = CompoundingType::Continuous;
        let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
            instrument_type: FinancialInstrumentType::CashInstrument, asset_class: AssetClass::EquityBasedInstrument,
            identifier: String::from("32198407128904"),
        };
        let asset_historical_data: AssetHistData = AssetHistData::build(
            Array1::from_vec(vec![0.4, 0.5]), Array1::from_vec(vec![0.0, 0.0]), Time::new(Array1::from_vec(vec![0.0, 1.0]))
        ).unwrap();
        let stock: Stock = Stock::new(
            100.0, 1_000_000, 3.0, 2.5, QuoteValues::PerShare, 99.0, compounding_type, 0.0,
            StockValuationType::DividendDiscountModel, Some(10.0), Some(3.0), Some(6.5), 5.0, financial_instrument_id,
            asset_historical_data, None
        );
        // Theoretical value
        let r: f64 = 3.0 / 100.0 + 0.0;
        let perpetuity_pv: f64 = 3.0 * 0.0_f64.exp() / ((r - 0.0).exp() - 1.0);
        assert!((stock.present_value().unwrap() - perpetuity_pv).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_stock_valuation_by_comparables() -> () {
        // Stock definition
        let compounding_type: CompoundingType = CompoundingType::Continuous;
        let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
            instrument_type: FinancialInstrumentType::CashInstrument, asset_class: AssetClass::EquityBasedInstrument,
            identifier: String::from("32198407128904"),
        };
        let asset_historical_data: AssetHistData = AssetHistData::build(
            Array1::from_vec(vec![0.4, 0.5]), Array1::from_vec(vec![0.0, 0.0]), Time::new(Array1::from_vec(vec![0.0, 1.0]))
        ).unwrap();
        // Valuation parameters
        let valuations: Array1<f64> = Array1::from_vec(vec![200_000_000.0, 1_000_000_000.0, 3_000_000_000.0, 500_000_000.0, 1_500_000_000.0]);
        let pe_ratios: Array1<f64> = Array1::from_vec(vec![20.0, 8.0, 9.0, 15.0, 11.0]);
        let pb_ratios: Array1<f64> = Array1::from_vec(vec![7.0, 2.0, 4.0, 10.0, 5.0]);
        let ev_to_ebitda: Array1<f64> = Array1::from_vec(vec![10.0, 5.0, 6.0, 7.0, 6.0]);
        let valuation_params: ValuationByComparablesParams = ValuationByComparablesParams::build(
            valuations, Some(pe_ratios), Some(pb_ratios), Some(ev_to_ebitda)
        ).unwrap();
        let stock: Stock = Stock::new(
            100.0, 1_000_000, 3.0, 2.5, QuoteValues::Total, 99.0, compounding_type, 0.0,
            StockValuationType::ValuationByComparables { params: valuation_params }, Some(10.0), Some(3.0),
            Some(6.5), 5.0, financial_instrument_id, asset_historical_data, None
        );
        assert!(1_000_000_000.0 < stock.present_value().unwrap());
    }
}