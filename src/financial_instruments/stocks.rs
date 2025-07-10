use ndarray::{Array1, Array2};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::DigiFiError;
use crate::utilities::{compare_array_len, time_value_utils::{Compounding, CompoundingType, Perpetuity}};
use crate::financial_instruments::{FinancialInstrument, FinancialInstrumentId};
use crate::corporate_finance;
use crate::portfolio_applications::{returns_average, returns_std, ReturnsMethod, ReturnsTransformation, AssetHistData, PortfolioInstrument};
use crate::statistics::linear_regression;
use crate::stochastic_processes::{StochasticProcess, standard_stochastic_models::GeometricBrownianMotion};


/// # Description
/// Choice of how to quote the values returned by functions.
pub enum QuoteValues {
    PerShare,
    Total,
}


#[derive(Debug)]
/// # Description
/// Type of model to use for valuation of stock price.
pub enum StockValuationType {
    DividendDiscountModel,
    ValuationByComparables { params: ValuationByComparablesParams },
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
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
    /// # Desscription
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
    pub fn new(valuations: Array1<f64>, pe_ratios: Option<Array1<f64>>, pb_ratios: Option<Array1<f64>>, ev_to_ebitda: Option<Array1<f64>>) -> Result<Self, DigiFiError> {
        if valuations.len() < 5 {
            return Err(DigiFiError::ValidationError { title: "Valuation by Comparables Params".to_owned(), details: "Minimum number of datapoints required is `5`.".to_owned(), });
        }
        ValuationByComparablesParams::validate_array(&valuations, &pe_ratios, "pe_ratios")?;
        ValuationByComparablesParams::validate_array(&valuations, &pb_ratios, "pb_ratios")?;
        ValuationByComparablesParams::validate_array(&valuations, &ev_to_ebitda, "ev_to_ebitda")?;
        Ok(ValuationByComparablesParams { valuations, pe_ratios, pb_ratios, ev_to_ebitda })
    }

    /// # Description
    /// Validates an array against the valuations array.
    fn validate_array(valuations: &Array1<f64>, array: &Option<Array1<f64>>, array_name: &str) -> Result<(), DigiFiError> {
        match array {
            Some(v) => {
                compare_array_len(&valuations, &v, "valuations", array_name)?;
                Ok(())
            },
            None => Ok(()),
        }
    }

    /// # Description
    /// Returns an array of stocks' valuations.
    pub fn valuations(&self) -> Array1<f64> {
        self.valuations.clone()
    }

    /// # Description
    /// Returns an array of stocks' P/E raatios.
    pub fn pe_ratios(&self) -> Option<Array1<f64>> {
        self.pe_ratios.clone()
    }

    /// # Description
    /// Returns an array of stock's P/B ratios.
    pub fn pb_ratios(&self) -> Option<Array1<f64>> {
        self.pb_ratios.clone()
    }

    /// # Description
    /// Returns an array of stocks' EV/EBITDA ratios.
    pub fn ev_to_ebitda(&self) -> Option<Array1<f64>> {
        self.ev_to_ebitda.clone()
    }
}


/// # Description
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
/// use digifi::utilities::{TEST_ACCURACY, CompoundingType};
/// use digifi::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass};
/// use digifi::financial_instruments::{QuoteValues, StockValuationType, ValuationByComparablesParams, Stock};
/// use digifi::portfolio_applications::AssetHistData;
///
/// // Stock definition
/// let compounding_type: CompoundingType = CompoundingType::Continuous;
/// let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {instrument_type: FinancialInstrumentType::CashInstrument,
///                                                                             asset_class: AssetClass::EquityBasedInstrument,
///                                                                             identifier: String::from("32198407128904") };
/// let asset_historical_data: AssetHistData = AssetHistData::new(Array1::from_vec(vec![0.4, 0.5]),
///                                                               Array1::from_vec(vec![0.0, 0.0]),
///                                                               Array1::from_vec(vec![0.0, 1.0])).unwrap();
/// let stock: Stock = Stock::new(100.0, 1_000_000, 3.0, 2.5, QuoteValues::PerShare, 99.0, compounding_type,
///                               0.0, StockValuationType::DividendDiscountModel, Some(10.0), Some(3.0), Some(6.5),
///                               5.0, financial_instrument_id, asset_historical_data, None).unwrap();
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
/// use digifi::utilities::{TEST_ACCURACY, CompoundingType};
/// use digifi::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass};
/// use digifi::financial_instruments::{QuoteValues, StockValuationType, ValuationByComparablesParams, Stock};
/// use digifi::portfolio_applications::AssetHistData;
///
/// // Stock definition
/// let compounding_type: CompoundingType = CompoundingType::Continuous;
/// let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {instrument_type: FinancialInstrumentType::CashInstrument,
///                                                                             asset_class: AssetClass::EquityBasedInstrument,
///                                                                             identifier: String::from("32198407128904") };
/// let asset_historical_data: AssetHistData = AssetHistData::new(Array1::from_vec(vec![0.4, 0.5]),
///                                                               Array1::from_vec(vec![0.0, 0.0]),
///                                                               Array1::from_vec(vec![0.0, 1.0])).unwrap();
///
/// // Valuation parameters
/// let valuations: Array1<f64> = Array1::from_vec(vec![200_000_000.0, 1_000_000_000.0, 3_000_000_000.0, 500_000_000.0, 1_500_000_000.0]);
/// let pe_ratios: Array1<f64> = Array1::from_vec(vec![20.0, 8.0, 9.0, 15.0, 11.0]);
/// let pb_ratios: Array1<f64> = Array1::from_vec(vec![7.0, 2.0, 4.0, 10.0, 5.0]);
/// let ev_to_ebitda: Array1<f64> = Array1::from_vec(vec![10.0, 5.0, 6.0, 7.0, 6.0]);
/// let valuation_params: ValuationByComparablesParams = ValuationByComparablesParams::new(valuations, Some(pe_ratios), Some(pb_ratios), Some(ev_to_ebitda)).unwrap();
/// let stock: Stock = Stock::new(100.0, 1_000_000, 3.0, 2.5, QuoteValues::Total, 99.0, compounding_type,
///                               0.0, StockValuationType::ValuationByComparables { params: valuation_params },
///                               Some(10.0), Some(3.0), Some(6.5),
///                               5.0, financial_instrument_id, asset_historical_data, None).unwrap();
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
    stochastic_model: Box<dyn StochasticProcess>,
}

impl Stock {

    /// # Description
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
    pub fn new(price_per_share: f64, n_shares_outstanding: i32, dividend_per_share: f64, earnings_per_share: f64, quote_values: QuoteValues, initial_price: f64,
               compounding_type: CompoundingType, dividend_growth_rate: f64, stock_valuation_type: StockValuationType, pe: Option<f64>, pb: Option<f64>,
               ev_to_ebitda: Option<f64>, t_f: f64, financial_instrument_id: FinancialInstrumentId, asset_historical_data: AssetHistData,
               stochastic_model: Option<Box<dyn StochasticProcess>>) -> Result<Self, DigiFiError> {
        let stochastic_model: Box<dyn StochasticProcess> = match stochastic_model {
            Some(v) => v,
            None => {
                // Default stochastic model for the case when the user doesn't provide one
                let end_index: usize = asset_historical_data.time_array.len() - 1;
                let prices: Array1<f64> = asset_historical_data.get_price(end_index, None)?;
                // Parameters estimated from log-returns
                let returns_transformation: ReturnsTransformation = ReturnsTransformation::LogReturn;
                let mu: f64 = returns_average(&prices, &ReturnsMethod::EstimatedFromTotalReturn, &returns_transformation, 252)?;
                let sigma: f64 = returns_std(&prices, &returns_transformation, 252)?;
                Box::new(GeometricBrownianMotion::new(mu, sigma, 1, asset_historical_data.time_array.len() - 1, t_f, initial_price))
            }
        };
        Ok(Stock {
            price_per_share, n_shares_outstanding: n_shares_outstanding as f64, dividend_per_share, earnings_per_share, quote_values, initial_price,
            compounding_type, dividend_growth_rate, stock_valuation_type, pe, pb, ev_to_ebitda, t_f, financial_instrument_id, asset_historical_data,
            stochastic_model })
    }

    fn apply_value_quotation_type(&self, prices_per_share: Array1<f64>) -> Array1<f64> {
        match self.quote_values {
            QuoteValues::PerShare => prices_per_share,
            QuoteValues::Total => prices_per_share * self.n_shares_outstanding,
        }
    }
    
    /// Description
    /// Returns current price per share.
    pub fn share_price(&self) -> f64 {
        self.price_per_share
    }

    /// # Description
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
        let trailing_eps: f64 = corporate_finance::earnings_per_share(net_income, preferred_dividends, n_common_shares_outstanding).unwrap_or(0.0);
        if in_place {
            self.earnings_per_share = trailing_eps;
        }
        trailing_eps
    }

    /// # Description
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
    pub fn trailing_pe(&mut self, in_place: bool) -> Option<f64> {
        let trailing_pe: Option<f64> = corporate_finance::pe_ratio(self.price_per_share, self.earnings_per_share);
        if in_place {
            self.pe = trailing_pe;
        }
        trailing_pe
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
    pub fn pb(&mut self, market_cap: f64, assets: f64, liabilities: f64, in_place: bool) -> Option<f64> {
        let book_value: f64 = corporate_finance::book_value(assets, liabilities);
        let pb: Option<f64> = corporate_finance::pb_ratio(market_cap, book_value);
        if in_place {
            self.pb = pb;
        }
        pb
    }

    /// # Description
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

    /// # Description
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
    pub fn ev_to_revenue(&self, market_cap: f64, total_debt: f64, cash: f64, revenue: f64) -> Option<f64> {
        corporate_finance::ev_to_revenue(market_cap, total_debt, cash, revenue)
    }

    /// # Description
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
    pub fn ev_to_ebitda(&mut self, market_cap: f64, total_debt: f64, cash: f64, ebitda: f64, in_place: bool) -> Option<f64> {
        let ev_to_ebitda: Option<f64> = corporate_finance::ev_to_ebitda(market_cap, total_debt, cash, ebitda);
        if in_place {
            self.ev_to_ebitda = ev_to_ebitda;
        }
        ev_to_ebitda
    }

    /// # Description
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

    /// # Description
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
    pub fn return_on_equity(&self, net_income: f64, equity: f64) -> Option<f64> {
        corporate_finance::return_on_equity(net_income, equity)
    }

    /// # Description
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
    pub fn return_on_assets(&self, net_income: f64, total_assets: f64) -> Option<f64> {
        corporate_finance::return_on_assets(net_income, total_assets)
    }

    /// # Description
    /// Performance measure used to evaluate the efficiency or profitability of an investment or compare the efficiency
    /// of a number of different investments. ROI tries to directly measure the amount of return on a particular investment,
    /// relative to the investment’s cost.
    /// 
    /// ROI = (Revenue - Cost of Goods Sold) / Cost of Goods Sold
    /// 
    /// # Input
    /// - `revenue`: Revenue of the company
    /// - `cost_of_goods_sold`: Cost of goods sold
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
    pub fn return_on_investment(&self, revenue: f64, cost_of_goods_sold: f64) -> Option<f64> {
        corporate_finance::return_on_investment(revenue, cost_of_goods_sold)
    }

    /// # Description
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
    pub fn debt_to_equity(&self, debt: f64, equity: f64) -> Option<f64> {
        if equity == 0.0 {
            None
        } else {
            Some(debt / equity)
        }
    }

    /// # Discription
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
        let dividend_perpetuity: Perpetuity = Perpetuity::new(self.dividend_per_share, cost_of_equity_capital,
                                                              self.dividend_growth_rate, self.compounding_type.clone())?;
        let pv: f64 = dividend_perpetuity.present_value();
        Ok(self.apply_value_quotation_type(Array1::from_vec(vec![pv]))[0])
    }

    fn validate_option_pair(&self, v: &Option<f64>, a: &Option<Array1<f64>>, v_name: &str, a_name: &str) -> Result<(), DigiFiError> {
        let error_title: String = String::from("Stock");
        match a {
            Some(_) => {
                match v {
                    Some(_) => Ok(()),
                    None => Err(DigiFiError::ValidationError { title: error_title.clone(), details: format!("The argument `{}` is None, while the argument `{}` is not. They must be the same variant of `Option` enum.", v_name, a_name) }),
                }
            },
            None => {
                match v {
                    Some(_) => Err(DigiFiError::ValidationError { title: error_title.clone(), details: format!("The argument `{}` is None, while the argument `{}` is not. They must be the same variant of `Option` enum.", a_name, v_name) }),
                    None => Ok(()),
                }
            },
        }
    }

    /// # Description
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
    pub fn valuation_by_comparables(&self, pe: Option<f64>, pb: Option<f64>, ev_to_ebitda: Option<f64>,
                                    valuation_params: &ValuationByComparablesParams) -> Result<f64, DigiFiError> {
        let error_title: String = String::from("Valuation by Comparables");
        let valuations: Array1<f64> = valuation_params.valuations();
        let pe_ratios: Option<Array1<f64>> = valuation_params.pe_ratios();
        let pb_ratios: Option<Array1<f64>> = valuation_params.pb_ratios();
        let ev_to_ebitdas: Option<Array1<f64>> = valuation_params.ev_to_ebitda();
        // Parameters validation
        self.validate_option_pair(&pe, &pe_ratios, &"pe", &"pe_ratios")?;
        self.validate_option_pair(&pb, &pb_ratios, &"pb", &"pb_ratios")?;
        self.validate_option_pair(&ev_to_ebitda, &ev_to_ebitdas, &"ev_to_ebitda", &"ev_to_ebitda")?;
        // Composition of linear regression data matrix
        let mut x: Vec<Vec<f64>> = Vec::<Vec<f64>>::new();
        x.push(vec![1.0; valuations.len()]);
        let mut index: usize = 1;
        let pe_index: Option<usize> = match pe_ratios {
            Some(v) => {
                x.push(v.to_vec());
                index += 1;
                Some(index - 1)
            },
            None => None,
        };
        let pb_index: Option<usize> = match pb_ratios {
            Some(v) => {
                x.push(v.to_vec());
                index += 1;
                Some(index - 1)
            },
            None => None,
        };
        let ev_to_ebitda_index: Option<usize> = match ev_to_ebitdas {
            Some(v) => {
                x.push(v.to_vec());
                index += 1;
                Some(index - 1)
            },
            None => None,
        };
        // Linear regression
        let x: Array2<f64> = Array2::from_shape_vec((index, valuations.len()), x.into_iter().flatten().collect())?;
        let params: Array1<f64> = linear_regression(&x.t().to_owned(), &valuations)?;
        // Valuation of stock based on linear regression parameters
        let mut valuation: f64 = params[0];
        match pe {
            Some(v) => {
                valuation += params[pe_index.ok_or(DigiFiError::ParameterConstraint { title: error_title.clone(), constraint: "No P/E index is defined.".to_owned(), })?] * v;
            },
            None => (),
        }
        match pb {
            Some(v) => {
                valuation += params[pb_index.ok_or(DigiFiError::ParameterConstraint { title: error_title.clone(), constraint: "No P/B index is defined.".to_owned(), })?] * v;
            },
            None => (),
        }
        match ev_to_ebitda {
            Some(v) => {
                valuation += params[ev_to_ebitda_index.ok_or(DigiFiError::ParameterConstraint { title: error_title.clone(), constraint: "No EV/EBITDA index is defined.".to_owned(), })?] * v;
            },
            None => (),
        }
        Ok(self.apply_value_quotation_type(Array1::from_vec(vec![valuation / self.n_shares_outstanding]))[0])
    }

    /// # Description
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
    pub fn payout_ratio(&self) -> Option<f64> {
        corporate_finance::payout_ratio(self.dividend_per_share, self.earnings_per_share)
    }

    /// # Description
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
    pub fn plowback_ratio(&self) -> Option<f64> {
        corporate_finance::plowback_ratio(self.dividend_per_share, self.earnings_per_share)
    }

    /// # Description
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

    /// # Description
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
        let expected_earnings: f64 = match expected_earnings {
            Some(v) => v,
            None => self.earnings_per_share,
        };
        let r: f64 = self.cost_of_equity_capital(expected_dividend);
        self.price_per_share - expected_earnings / r
    }
}

impl FinancialInstrument for Stock {

    /// # Descdription
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

    /// # Descdription
    /// Net present value of the stock.
    ///
    /// # Output
    /// - Present value of the stock minus the initial price it took to purchase the stock
    fn net_present_value(&self) -> Result<f64, DigiFiError> {
        Ok(self.present_value()? - self.initial_price)
    }

    /// # Descdription
    /// Future value of the stock.
    ///
    /// # Output
    /// - Future value of the stock at the given time (Computed from the present value of the stock)
    fn future_value(&self) -> Result<f64, DigiFiError> {
        let r: f64 = self.cost_of_equity_capital(None);
        let discount_term: Compounding = Compounding::new(r, &self.compounding_type);
        Ok(self.present_value()? / discount_term.compounding_term(self.t_f))
    }

    /// # Description
    /// Returns an array of stock prices.
    ///
    /// # Input
    /// - `end_index`: Time index beyond which no data will be returned
    /// - `start_index`: Time index below which no data will be returned
    fn get_prices(&self, end_index: usize, start_index: Option<usize>) -> Result<Array1<f64>, DigiFiError> {
        self.asset_historical_data.get_price(end_index, start_index)
    }

    /// # Description
    /// Returns an array of predictable incomes for the stock (i.e., dividends).
    ///
    /// # Input
    /// - `end_index`: Time index beyond which no data will be returned
    /// - `start_index`: Time index below which no data will be returned
    fn get_predictable_income(&self, end_index: usize, start_index: Option<usize>) -> Result<Array1<f64>, DigiFiError> {
        self.asset_historical_data.get_predictable_income(end_index, start_index)
    }

    /// # Description
    /// Returns an array of time steps at which the asset price and predictable_income are recorded.
    fn get_time_array(&self) -> Array1<f64> {
        self.asset_historical_data.time_array.clone()
    }

    /// # Description
    /// Updates the number of paths the stochastic model will produce when called.
    ///
    /// # Input
    /// - `n_paths`: New number of paths to use
    fn update_n_stochastic_paths(&mut self, n_paths: usize) -> () {
        self.stochastic_model.update_n_paths(n_paths)
    }

    /// # Description
    /// Simulated stochastic paths of the stock.
    /// 
    /// # Output
    /// - Simulated prices of the stock
    fn stochastic_simulation(&self) -> Result<Vec<Array1<f64>>, DigiFiError> {
        self.stochastic_model.get_paths()
    }

    /// # Description
    /// Generates an array of prices and predictable income, and updates the `asset_historical_data`.
    /// 
    /// # Input
    /// - `in_place`: If true, uses generated data to update the asset history data 
    fn generate_historic_data(&mut self, in_place: bool) -> Result<AssetHistData, DigiFiError> {
        let prices: Array1<f64> = self.stochastic_model.get_paths()?.remove(0);
        let new_data: AssetHistData = AssetHistData::new(prices,
                                                         Array1::from_vec(vec![0.0; self.asset_historical_data.time_array.len()]),
                                                         self.asset_historical_data.time_array.clone())?;
        if in_place {
            self.asset_historical_data = new_data.clone();
        }
        Ok(new_data)
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
    use crate::utilities::TEST_ACCURACY;
    use crate::utilities::time_value_utils::CompoundingType;
    use crate::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass};
    use crate::financial_instruments::stocks::{QuoteValues, StockValuationType, ValuationByComparablesParams, Stock};
    use crate::portfolio_applications::AssetHistData;

    #[test]
    fn unit_test_stock_dividend_discount_model() -> () {
        // Stock definition
        let compounding_type: CompoundingType = CompoundingType::Continuous;
        let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {instrument_type: FinancialInstrumentType::CashInstrument,
                                                                                    asset_class: AssetClass::EquityBasedInstrument,
                                                                                    identifier: String::from("32198407128904") };
        let asset_historical_data: AssetHistData = AssetHistData::new(Array1::from_vec(vec![0.4, 0.5]),
                                                                      Array1::from_vec(vec![0.0, 0.0]),
                                                                      Array1::from_vec(vec![0.0, 1.0])).unwrap();
        let stock: Stock = Stock::new(100.0, 1_000_000, 3.0, 2.5, QuoteValues::PerShare, 99.0, compounding_type,
                                      0.0, StockValuationType::DividendDiscountModel, Some(10.0), Some(3.0), Some(6.5),
                                      5.0, financial_instrument_id, asset_historical_data, None).unwrap();
        // Theoretical value
        let r: f64 = 3.0 / 100.0 + 0.0;
        let perpetuity_pv: f64 = 3.0 * 0.0_f64.exp() / ((r - 0.0).exp() - 1.0);
        assert!((stock.present_value().unwrap() - perpetuity_pv).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_stock_valuation_by_comparables() -> () {
        // Stock definition
        let compounding_type: CompoundingType = CompoundingType::Continuous;
        let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {instrument_type: FinancialInstrumentType::CashInstrument,
                                                                                    asset_class: AssetClass::EquityBasedInstrument,
                                                                                    identifier: String::from("32198407128904") };
        let asset_historical_data: AssetHistData = AssetHistData::new(Array1::from_vec(vec![0.4, 0.5]),
                                                                      Array1::from_vec(vec![0.0, 0.0]),
                                                                      Array1::from_vec(vec![0.0, 1.0])).unwrap();
        // Valuation parameters
        let valuations: Array1<f64> = Array1::from_vec(vec![200_000_000.0, 1_000_000_000.0, 3_000_000_000.0, 500_000_000.0, 1_500_000_000.0]);
        let pe_ratios: Array1<f64> = Array1::from_vec(vec![20.0, 8.0, 9.0, 15.0, 11.0]);
        let pb_ratios: Array1<f64> = Array1::from_vec(vec![7.0, 2.0, 4.0, 10.0, 5.0]);
        let ev_to_ebitda: Array1<f64> = Array1::from_vec(vec![10.0, 5.0, 6.0, 7.0, 6.0]);
        let valuation_params: ValuationByComparablesParams = ValuationByComparablesParams::new(valuations, Some(pe_ratios), Some(pb_ratios), Some(ev_to_ebitda)).unwrap();
        let stock: Stock = Stock::new(100.0, 1_000_000, 3.0, 2.5, QuoteValues::Total, 99.0, compounding_type,
                                      0.0, StockValuationType::ValuationByComparables { params: valuation_params },
                                      Some(10.0), Some(3.0), Some(6.5),
                                      5.0, financial_instrument_id, asset_historical_data, None).unwrap();
        assert!(1_000_000_000.0 < stock.present_value().unwrap());
    }
}