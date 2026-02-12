use ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
#[cfg(feature = "plotly")]
use plotly::{Plot, Surface, Layout, layout::{Axis, LayoutScene}, common::Title};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::time_value_utils::{Compounding, CompoundingType};
use crate::financial_instruments::{FinancialInstrument, FinancialInstrumentId, Payoff};
use crate::portfolio_applications::{AssetHistData, PortfolioInstrument};
use crate::statistics::{ProbabilityDistribution, continuous_distributions::NormalDistribution};
use crate::lattice_models::LatticeModel;
use crate::stochastic_processes::StochasticProcess;
use crate::lattice_models::{binomial_models::BrownianMotionBinomialModel, trinomial_models::BrownianMotionTrinomialModel};
// TODO: Add implied volatility calculator, volatility smile and surface plots


/// Mininum-variance hedge ratio for a forward contract that hedges an underlying asset.
///
/// Note: this assumes that there is a linear relationship between the asset and the contract.
///
/// # Input
/// - `asset_price_sigma`: Volatility of the asset price
/// - `contract_price_sigma`: Volatility of the contract price
/// - `asset_to_contract_corr`: Correlation between the asset price and the contract price
///
/// # Output
/// - Minimum-variance hedge ratio (float)
///
/// # LaTeX Formula
/// - h^{*} = \\rho\\frac{\\sigma_{S}}{\\sigma_{F}}
pub fn minimum_variance_hedge_ratio(asset_price_sigma: f64, contract_price_sigma: f64, asset_to_contract_corr: f64) -> f64 {
    asset_to_contract_corr * asset_price_sigma / contract_price_sigma
}


#[derive(Clone, Copy, Debug)]
pub enum ContractType {
    Forward,
    Future,
    BillsOfExchange,
}


#[derive(Debug)]
pub enum OptionType {
    European,
    American,
    Bermudan { exercise_time_steps: Vec<bool> },
}


#[derive(Clone, Copy, Debug)]
pub enum BlackScholesType {
    Call,
    Put,
}


#[derive(Clone, Copy, Debug)]
pub enum OptionPricingMethod {
    BlackScholes { type_: BlackScholesType },
    Binomial { n_steps: usize },
    Trinomial { n_steps: usize },
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Surface (i.e., underlying asset price vs option value vs time to maturity) of the option price as it approaches maturity.
pub struct PresentValueSurface {
    /// Array of times to maturity
    pub times_to_maturity: Array1<f64>,
    /// Array of prices
    pub price_array: Array1<f64>,
    /// Matrix of present values of the option
    pub pv_matrix: Vec<Array1<f64>>,
}


/// Black-Scholes formula implementation for pricing European options.
/// 
/// # Input
/// - `s`: Underlying asset price
/// - `k`: Option strike price
/// - `sigma`: Standard deviation of underlying asset price
/// - `time_to_maturity`: Time to maturity of the option contract
/// - `r`: Discount rate (e.g., risk-free rate)
/// - `q`: Predicatable yield of the underlying asset (e.g., dividend yield)
/// - `type_`: Type of option contract (i.e., call or put option)
/// 
/// # Output
/// - European option premium (i.e., option price)
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model>
/// - Original Source: <https://www.cs.princeton.edu/courses/archive/fall09/cos323/papers/black_scholes73.pdf>
/// 
/// # Examples
/// 
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::financial_instruments::{BlackScholesType, black_scholes_formula};
/// 
/// let value: f64 = black_scholes_formula(10.0, 11.0, 0.2, 1.0, 0.02, 0.0, &BlackScholesType::Call).unwrap();
/// 
/// assert!((value - 0.49438669572304805).abs() < TEST_ACCURACY);
/// ```
pub fn black_scholes_formula(s: f64, k: f64, sigma: f64, time_to_maturity: f64, r: f64, q: f64, type_: &BlackScholesType) -> Result<f64, DigiFiError> {
    let normal_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0)?;
    let d_1_numerator: f64 = (s/k).ln() + time_to_maturity  * (r - q + 0.5*sigma.powi(2));
    let d_1: f64 = d_1_numerator / (sigma * time_to_maturity.sqrt());
    let d_2: f64 = d_1 - sigma * time_to_maturity.sqrt();
    match type_ {
        BlackScholesType::Call => {
            let component_1: f64 = s * (-q * time_to_maturity).exp() * normal_dist.cdf(d_1)?;
            let component_2: f64 = k * (-r * time_to_maturity).exp() * normal_dist.cdf(d_2)?;
            Ok(component_1 - component_2)
        },
        BlackScholesType::Put => {
            let component_1: f64 = s * (-q * time_to_maturity).exp() * normal_dist.cdf(-d_1)?;
            let component_2: f64 = k * (-r * time_to_maturity).exp() * normal_dist.cdf(-d_2)?;
            Ok(component_2 - component_1)
        },
    }
}


/// Contract financial instrument and its methods.
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Futures_contract>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, Time, CompoundingType};
/// use digifi::statistics::{ProbabilityDistribution, NormalDistribution};
/// use digifi::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass, ContractType, Contract};
/// use digifi::portfolio_applications::AssetHistData;
///
/// // Contract definition
/// let compounding_type: CompoundingType = CompoundingType::Continuous;
/// let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
///     instrument_type: FinancialInstrumentType::DerivativeInstrument, asset_class: AssetClass::EquityBasedInstrument,
///     identifier: String::from("32198407128904"),
/// };
/// let asset_historical_data: AssetHistData = AssetHistData::build(
///     Array1::from_vec(vec![0.4, 0.5]), Array1::from_vec(vec![0.0, 0.0]), Time::new(Array1::from_vec(vec![0.0, 1.0]))
/// ).unwrap();
/// let contract: Contract = Contract::build(
///     ContractType::Future, 1.5, 100.0, 0.03, 2.0, 101.0, compounding_type, financial_instrument_id, asset_historical_data, None
/// ).unwrap();
///
/// // Theoretical value
/// let forward_price: f64 = 101.0 * (0.03 * 2.0_f64).exp();
/// let pv: f64 = (forward_price - 100.0) * (-0.03 * 2.0_f64).exp();
/// assert!((contract.present_value().unwrap() - pv).abs() < TEST_ACCURACY);
/// ```
pub struct Contract {
    /// Type of contract
    _contract_type: ContractType,
    /// Current market price of the contract
    current_contract_price: f64,
    /// Agreed delivery price
    delivery_price: f64,
    /// Discount rate used to discount the future cashflow payments
    discount_rate: f64,
    /// Time to maturity of the contract
    time_to_maturity: f64,
    /// Current spot price of the underlying asset
    spot_price: f64,
    /// Compounding type used to discount cashflows
    compounding_type: CompoundingType,
    /// Parameters for defining regulatory categorization of an instrument
    financial_instrument_id: FinancialInstrumentId,
    /// Time series asset data
    asset_historical_data: AssetHistData,
    /// Stochastic model to use for price paths generation
    stochastic_model: Option<Box<dyn StochasticProcess>>,
}

impl Contract {
    /// Creates a new `Contract` instance.
    ///
    /// - `contract_type`: Type of contract
    /// - `current_contract_price`: Current market price of the contract
    /// - `delivery_price`: Agreed delivery price
    /// - `discount_rate`: Discount rate used to discount the future cashflow payments
    /// - `time_to_maturity`: Time to maturity of the contract
    /// - `spot_price`: Current spot price of the underlying asset
    /// - `compounding_type`: Compounding type used to discount cashflows
    /// - `financial_instrument_id`: Parameters for defining regulatory categorization of an instrument
    /// - `asset_historical_data`: Time series asset data
    /// - `stochastic_model`: Stochastic model to use for price paths generation
    ///
    /// # Errors
    /// - Returns an error if time to maturity is negative.
    pub fn build(
        _contract_type: ContractType, current_contract_price: f64, delivery_price: f64, discount_rate: f64, time_to_maturity: f64, spot_price: f64,
        compounding_type: CompoundingType, financial_instrument_id: FinancialInstrumentId, asset_historical_data: AssetHistData,
        stochastic_model: Option<Box<dyn StochasticProcess>>
    ) -> Result<Self, DigiFiError> {
        if time_to_maturity < 0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `time_to_maturity` must be non-negative.".to_owned(),
            });
        }
        Ok(Self {
            _contract_type, current_contract_price, delivery_price, discount_rate, time_to_maturity, spot_price, compounding_type,
            financial_instrument_id, asset_historical_data, stochastic_model,
        })
    }

    /// Updates the spot price.
    ///
    /// # Input
    /// -`new_spot_price`: New spot price of the underlying asset
    pub fn update_spot_price(&mut self, new_spot_price: f64) -> () {
        self.spot_price = new_spot_price;
    }

    /// Updates the time to maturity of the contract.
    ///
    /// # Input
    /// -`new_time_to_maturity`: New time to maturity maturity of the contract
    ///
    /// # Errors
    /// - Returns an error if the new time to maturity is negative.
    pub fn update_time_to_maturity(&mut self, new_time_to_maturity: f64) -> Result<(), DigiFiError> {
        if new_time_to_maturity < 0.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `new_time_to_maturity` must be non-negative.".to_owned(),
            });
        }
        self.time_to_maturity = new_time_to_maturity;
        Ok(())
    }

    /// Add predictable yield to discount rate and save it in the instance definition.
    ///
    /// # Input
    /// - `predictable_yield`: Predictable yield
    pub fn append_predictable_yield(&mut self, predictable_yield: f64) -> () {
        self.discount_rate += predictable_yield;
    }

    /// Add predictable income to the spot price and save it in the instance definition.
    ///
    /// # Input
    /// - `predictable_income`: Predictable income
    pub fn append_predictable_income(&mut self, predictable_income: f64) -> () {
        self.spot_price += predictable_income;
    }

    pub fn forward_price(&self) -> Result<f64, DigiFiError> {
        let discount_term: Compounding = Compounding::new(self.discount_rate, &self.compounding_type);
        Ok(self.spot_price / discount_term.compounding_term(self.time_to_maturity))
    }
}

impl ErrorTitle for Contract {
    fn error_title() -> String {
        String::from("Contract")
    }
}

impl FinancialInstrument for Contract {

    /// Present value of the contract.
    ///
    /// # Output
    /// - Present value of the contract
    ///
    /// # LaTeX Fomula
    /// - PV = (F_{t} - K)e^{-r\\tau}
    fn present_value(&self) -> Result<f64, DigiFiError> {
        let discount_term: Compounding = Compounding::new(self.discount_rate, &self.compounding_type);
        let forward_price: f64 = self.forward_price()?;
        Ok((forward_price - self.delivery_price) * discount_term.compounding_term(self.time_to_maturity))
    }

    /// Net present value of the contract.
    ///
    /// # Output
    /// - Present value of the contract minus the initial price it took to purchase the contract
    fn net_present_value(&self) -> Result<f64, DigiFiError> {
        Ok(-self.current_contract_price + self.present_value()?)
    }

    /// Future value of the contract.
    ///
    /// # Output
    /// - Future value of the contract at its maturity (Computed from the present value of the contract)
    fn future_value(&self) -> Result<f64, DigiFiError> {
        let discount_term: Compounding = Compounding::new(self.discount_rate, &self.compounding_type);
        Ok(self.present_value()? / discount_term.compounding_term(self.time_to_maturity))
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

impl PortfolioInstrument for Contract {

    fn asset_name(&self) -> String {
        self.financial_instrument_id.identifier.clone()
    }

    fn historical_data(&self) -> &AssetHistData {
        &self.asset_historical_data
    }
}


/// Option financial instrument and its methods.
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Option_(finance)>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, Time, CompoundingType};
/// use digifi::statistics::{ProbabilityDistribution, NormalDistribution};
/// use digifi::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass, LongCall, OptionType,
///                                     BlackScholesType, OptionPricingMethod, OptionContract};
/// use digifi::portfolio_applications::AssetHistData;
///
/// // Option definition
/// let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
///     instrument_type: FinancialInstrumentType::DerivativeInstrument, asset_class: AssetClass::EquityBasedInstrument,
///     identifier: String::from("32198407128904"),
/// };
/// let asset_historical_data: AssetHistData = AssetHistData::build(
///     Array1::from_vec(vec![0.4, 0.5]), Array1::from_vec(vec![0.0, 0.0]), Time::new(Array1::from_vec(vec![0.0, 1.0]))
/// ).unwrap();
/// let payoff: Box<LongCall> = Box::new(LongCall { k: 100.0, cost: 1.5 });
/// let option_pricing_method: OptionPricingMethod = OptionPricingMethod::BlackScholes { type_: BlackScholesType::Call };
/// let option: OptionContract = OptionContract::build(
///     99.0, 100.0, 0.02, 0.0, 3.0, 0.2, OptionType::European, payoff, 1.5, option_pricing_method, financial_instrument_id, asset_historical_data, None
///    ).unwrap();
///
/// // Theoretical value
/// let normal_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0).unwrap();
/// let d_1_numerator: f64 = (99.0 / 100.0_f64).ln() + 3.0  * (0.02 - 0.0 + 0.5 * 0.2_f64.powi(2));
/// let d_1: f64 = d_1_numerator / (0.2 * 3.0_f64.sqrt());
/// let d_2: f64 = d_1 - 0.2 * 3.0_f64.sqrt();
/// let component_1: f64 = 99.0 * (-0.0 * 3.0_f64).exp() * normal_dist.cdf(d_1).unwrap();
/// let component_2: f64 = 100.0 * (-0.02 * 3.0_f64).exp() * normal_dist.cdf(d_2).unwrap();
/// let pv: f64 = component_1 - component_2;
/// assert!((option.present_value().unwrap() - pv).abs() < TEST_ACCURACY);
/// ```
pub struct OptionContract {
    /// Current market price of the underlying asset
    asset_price: f64,
    /// Strike price of the option
    strike_price: f64,
    /// Discount rate used to discount the future cashflow payments
    discount_rate: f64,
    /// Yield of the underlying asset (e.g., dividend yield)
    yield_: f64,
    /// Time to maturity of the option
    time_to_maturity: f64,
    /// Volatility of the underlying asset returns
    sigma: f64,
    /// Type of option payoff times (i.e., European, American, Bermudan)
    option_type: OptionType,
    /// Payoff of the option
    payoff: Box<dyn Payoff>,
    /// Current price (premium) of the option
    initial_option_price: f64,
    /// Method for pricing the option
    option_pricing_method: OptionPricingMethod,
    /// Parameters for defining regulatory categorization of an instrument
    financial_instrument_id: FinancialInstrumentId,
    /// Time series asset data
    asset_historical_data: AssetHistData,
    /// Stochastic model to use for price paths generation
    stochastic_model: Option<Box<dyn StochasticProcess>>,
}

impl OptionContract {
    /// Creates a new `OptionContract` instance.
    ///
    /// # Input
    /// - `asset_price`: Current market price of the underlying asset
    /// - `strike_price`: Strike price of the option
    /// - `discount_rate`: Discount rate used to discount the future cashflow payments
    /// - `yield_`: Yield of the underlying asset (e.g., dividend yield)
    /// - `time_to_maturity`: Time to maturity of the option
    /// - `sigma`: Volatility of the option returns
    /// - `option_type`: Type of option payoff times (i.e., European, American, Bermudan)
    /// - `payoff`: Payoff of the option
    /// - `initial_option_price`: Current price (premium) of the option
    /// - `option_pricing_method`: Method for pricing the option
    /// - `financial_instrument_id`: Parameters for defining regulatory categorization of an instrument
    /// - `asset_historical_data`: Time series asset data
    /// - `stochastic_model`: Stochastic model to use for price paths generation
    ///
    /// # Errors
    /// - Returns an error if the maturity is negative.
    /// - Returns an error if Black-Scholes pricing method is used together with any but European option type.
    pub fn build(
        asset_price: f64, strike_price: f64, discount_rate: f64, yield_: f64, time_to_maturity: f64, sigma: f64, option_type: OptionType,
        payoff: Box<dyn Payoff>, initial_option_price: f64, option_pricing_method: OptionPricingMethod, financial_instrument_id: FinancialInstrumentId,
        asset_historical_data: AssetHistData, stochastic_model: Option<Box<dyn StochasticProcess>>
    ) -> Result<Self, DigiFiError> {
        if time_to_maturity < 0.0 {
            return Err(DigiFiError::ParameterConstraint { title: Self::error_title(), constraint: "The argument `time_to_maturity` must be non-negative.".to_owned(), });
        }
        if let OptionPricingMethod::BlackScholes { .. } = option_pricing_method {
            match option_type {
                OptionType::European => (),
                _ => return Err(DigiFiError::ValidationError {
                    title: Self::error_title(),
                    details: "`Black-Scholes` option pricing can only be used for `European` options.".to_owned(),
                }),
            }
        }
        payoff.validate_payoff(5)?;
        Ok(Self {
            asset_price, strike_price, discount_rate, yield_, time_to_maturity, sigma, option_type, payoff, initial_option_price, option_pricing_method,
            financial_instrument_id, asset_historical_data, stochastic_model,
        })
    }

    /// Check that the instance is a European option.
    fn is_european(&self) -> Result<(), DigiFiError> {
        if let OptionType::European = self.option_type { return Ok(()) }
        Err(DigiFiError::ValidationError { title: Self::error_title(), details: "The option is not `European`.".to_owned(), })
    }

    /// Measure of sensitivity of the option price to the underlying asset price.
    ///
    /// # Input
    /// - `increment`: Percent change in the asset price of the underlying
    ///
    /// # Output
    /// - Delta
    ///
    /// # LaTeX Formula
    /// - \\Delta = \\frac{\\partial V}{\\partial S}
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Greeks_(finance)#Delta>
    /// - Original Source: N/A
    pub fn delta(&mut self, increment: f64) -> Result<f64, DigiFiError> {
        if let OptionPricingMethod::BlackScholes { type_ } = &self.option_pricing_method {
            return self.bs_delta(type_)
        }
        let original_value: f64 = self.asset_price;
        let original_pv: f64 = self.present_value()?;
        let new_value: f64 = original_value * (1.0 + increment);
        self.asset_price = new_value;
        let new_pv: f64 = self.present_value()?;
        self.asset_price = original_value;
        Ok((new_pv - original_pv) / (new_value - original_value))
    }

    /// Measure of sensitivity of the option price to the volatility of the underlying asset.
    ///
    /// # Input
    /// - `increment`: Percent change in the volatility of the underlying
    ///
    /// # Output
    ///- Vega
    ///
    /// # LaTeX Formula
    ///- \\nu = \\frac{\\partial V}{\\partial\\sigma}
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Greeks_(finance)#Vega>
    /// - Original Source: N/A
    pub fn vega(&mut self, increment: f64) -> Result<f64, DigiFiError> {
        if let OptionPricingMethod::BlackScholes { .. } = &self.option_pricing_method {
            return self.bs_vega()
        }
        let original_value: f64 = self.sigma;
        let original_pv: f64 = self.present_value()?;
        let new_value: f64 = original_value * (1.0 + increment);
        self.sigma = new_value;
        let new_pv: f64 = self.present_value()?;
        self.sigma = original_value;
        Ok((new_pv - original_pv) / (new_value - original_value))
    }

    /// Measure of sensitivity of the option price to the time to maturity.
    ///
    /// # Input
    /// - `increment`: Percent change in the time to maturity of the option
    ///
    /// # Output
    /// - Theta
    /// # LaTeX Formula
    /// - \\Theta = \\frac{\\partial V}{\\partial\\tau}
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Greeks_(finance)#Theta>
    /// - Original Source: N/A
    pub fn theta(&mut self, increment: f64) -> Result<f64, DigiFiError> {
        if let OptionPricingMethod::BlackScholes { type_ } = &self.option_pricing_method {
            return self.bs_theta(type_)
        }
        let original_value: f64 = self.time_to_maturity;
        let original_pv: f64 = self.present_value()?;
        let new_value: f64 = original_value * (1.0 + increment);
        self.time_to_maturity = new_value;
        let new_pv: f64 = self.present_value()?;
        self.time_to_maturity = original_value;
        Ok((new_pv - original_pv) / (new_value - original_value))
    }

    /// Measure of sensitivity of the option price to the interest rate.
    ///
    /// # Input
    /// - `increment`: Percent change in the discount rate
    ///
    /// # Output
    /// - Rho
    ///
    /// # LaTeX Formula
    /// - \\rho = \\frac{\\partial V}{\\partial r}
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Greeks_(finance)#Rho>
    /// - Original Source: N/A
    pub fn rho(&mut self, increment: f64) -> Result<f64, DigiFiError> {
        if let OptionPricingMethod::BlackScholes { type_ } = &self.option_pricing_method {
            return self.bs_rho(type_)
        }
        let original_value: f64 = self.discount_rate;
        let original_pv: f64 = self.present_value()?;
        let new_value: f64 = original_value * (1.0 + increment);
        self.discount_rate = new_value;
        let new_pv: f64 = self.present_value()?;
        self.discount_rate = original_value;
        Ok((new_pv - original_pv) / (new_value - original_value))
    }

    /// Measure of sensitivity of the option price to the dividend of the underlying asset.
    ///
    /// # Input
    /// - `increment`: Percent change in the yield of the underlying
    ///
    /// # Output
    /// - Epsilon
    ///
    /// # LaTeX Formula
    /// - \\epsilon = \\frac{\\partial V}{\\partial q}
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Greeks_(finance)#Epsilon>
    /// - Original Source: N/A
    pub fn epsilon(&mut self, increment: f64) -> Result<f64, DigiFiError> {
        if let OptionPricingMethod::BlackScholes { type_ } = &self.option_pricing_method {
            return self.bs_epsilon(type_)
        }
        let original_value: f64 = self.yield_;
        let original_pv: f64 = self.present_value()?;
        let new_value: f64 = original_value * (1.0 + increment);
        self.yield_ = new_value;
        let new_pv: f64 = self.present_value()?;
        self.yield_ = original_value;
        Ok((new_pv - original_pv) / (new_value - original_value))
    }

    /// Measure of sensitivity of the option price to change in the underlying asset price.
    ///
    /// # Input
    /// - `increment`: Percent change in the asset price of the underlying
    ///
    /// # Output
    /// - Gamma
    ///
    /// # LaTeX Formula
    /// - \\Gamma = \\frac{\\partial^{2}\\Delta}{\\partial S^{2}}
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Greeks_(finance)#Gamma>
    /// - Original Source: N/A
    pub fn gamma(&mut self, increment: f64) -> Result<f64, DigiFiError> {
        if let OptionPricingMethod::BlackScholes { .. } = &self.option_pricing_method {
            return self.bs_gamma()
        }
        let original_value: f64 = self.asset_price;
        let original_pv: f64 = self.present_value()?;
        let up_value: f64 = original_value * (1.0 + increment);
        self.asset_price = up_value;
        let up_pv: f64 = self.present_value()?;
        let down_value: f64 = original_value * (1.0 - increment);
        self.asset_price = down_value;
        let down_pv: f64 = self.present_value()?;
        self.asset_price = original_value;
        let delta_1: f64 = (original_pv - down_pv) / (original_value - down_value);
        let delta_2: f64 = (up_pv - original_pv) / (up_value - original_value);
        Ok((delta_2 - delta_1) / (up_value - down_value))
    }

    /// Defines Black-Scholes parameters d_1 and d_2 for the European option.
    ///
    /// # Errors
    /// - Returns an error if the option is not European.
    fn european_option_black_scholes_params(&self) -> Result<(f64, f64), DigiFiError> {
        self.is_european()?;
        let d_1_numerator: f64 = (self.asset_price/self.strike_price).ln() + self.time_to_maturity  * (self.discount_rate - self.yield_ + 0.5*self.sigma.powi(2));
        let d_1: f64 = d_1_numerator / (self.sigma * self.time_to_maturity.sqrt());
        let d_2: f64 = d_1 - self.sigma * self.time_to_maturity.sqrt();
        Ok((d_1, d_2))
    }

    /// Value of the European option evaluated using Black-Scholes formula.
    ///
    /// # Input
    /// - `black_scholes_type`: Type of option (i.e., Call or Put) to price using the Black-Scholes method
    ///
    /// # Errors
    /// - Returns an error if the option is not European.
    fn european_option_black_scholes(&self, black_scholes_type: &BlackScholesType) -> Result<f64, DigiFiError> {
        self.is_european()?;
        black_scholes_formula(self.asset_price, self.strike_price, self.sigma, self.time_to_maturity, self.discount_rate, self.yield_, black_scholes_type)
    }

    /// Delta of the European option.
    ///
    /// Measure of sensitivity of the option price to the underlying asset price.
    ///
    /// # Output
    /// - Delta
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Greeks_(finance)#Formulae_for_European_option_Greeks>
    /// - Original Source: N/A
    pub fn bs_delta(&self, black_scholes_type: &BlackScholesType) -> Result<f64, DigiFiError> {
        let (d_1, _) = self.european_option_black_scholes_params()?;
        let normal_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0)?;
        match black_scholes_type {
            BlackScholesType::Call => {
                Ok((-self.yield_ * self.time_to_maturity).exp() * normal_dist.cdf(d_1)?)
            },
            BlackScholesType::Put => {
                Ok(-(-self.yield_ * self.time_to_maturity).exp() * normal_dist.cdf(-d_1)?)
            },
        }
    }

    /// Vega of the European option.
    ///
    /// Measure of sensitivity of the option price to the volatility of the underlying asset.
    ///
    /// # Output
    ///- Vega
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Greeks_(finance)#Formulae_for_European_option_Greeks>
    /// - Original Source: N/A
    pub fn bs_vega(&self) -> Result<f64, DigiFiError> {
        let (d_1, _) = self.european_option_black_scholes_params()?;
        let normal_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0)?;
        Ok(self.asset_price * (-self.yield_ * self.time_to_maturity).exp() * normal_dist.cdf(d_1)? * self.time_to_maturity.sqrt())
    }

    /// Theta of the European option.
    ///
    /// Measure of sensitivity of the option price to the time to maturity.
    ///
    /// # Output
    /// - Theta
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Greeks_(finance)#Formulae_for_European_option_Greeks>
    /// - Original Source: N/A
    pub fn bs_theta(&self, black_scholes_type: &BlackScholesType) -> Result<f64, DigiFiError> {
        let (d_1, d_2) = self.european_option_black_scholes_params()?;
        let normal_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0)?;
        match black_scholes_type {
            BlackScholesType::Call => {
                let component_1: f64 = -(-self.yield_ * self.time_to_maturity).exp() * self.asset_price * normal_dist.cdf(d_1)? * self.sigma / (2.0 * self.time_to_maturity.sqrt());
                let component_2: f64 = self.discount_rate * self.strike_price * (-self.discount_rate * self.time_to_maturity).exp() * normal_dist.cdf(d_2)?;
                let component_3: f64 = self.yield_ * self.asset_price * (-self.yield_ * self.time_to_maturity).exp() * normal_dist.cdf(d_1)?;
                Ok(component_1 - component_2 + component_3)
            },
            BlackScholesType::Put => {
                let component_1: f64 = -(-self.yield_ * self.time_to_maturity).exp() * self.asset_price * normal_dist.cdf(d_1)? * self.sigma / (2.0 * self.time_to_maturity.sqrt());
                let component_2: f64 = self.discount_rate * self.strike_price * (-self.discount_rate * self.time_to_maturity).exp() * normal_dist.cdf(-d_2)?;
                let component_3: f64 = self.yield_ * self.asset_price * (-self.yield_ * self.time_to_maturity).exp() * normal_dist.cdf(-d_1)?;
                Ok(component_1+ component_2 - component_3)
            },
        }
    }

    /// Rho of the European option.
    ///
    /// Measure of sensitivity of the option price to the interest rate.
    ///
    //// # Output
    /// - Rho
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Greeks_(finance)#Formulae_for_European_option_Greeks>
    /// - Original Source: N/A
    pub fn bs_rho(&self, black_scholes_type: &BlackScholesType) -> Result<f64, DigiFiError> {
        let (_, d_2) = self.european_option_black_scholes_params()?;
        let normal_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0)?;
        match black_scholes_type {
            BlackScholesType::Call => {
                Ok(self.strike_price * self.time_to_maturity * (-self.discount_rate * self.time_to_maturity).exp() * normal_dist.cdf(d_2)?)
            },
            BlackScholesType::Put => {
                Ok(-self.strike_price * self.time_to_maturity * (-self.discount_rate * self.time_to_maturity).exp() * normal_dist.cdf(-d_2)?)
            },
        }
    }

    /// Epsilon of the European option.
    ///
    /// Measure of sensitivity of the option price to the dividend of the underlying asset.
    ///
    /// # Output
    /// - Epsilon
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Greeks_(finance)#Formulae_for_European_option_Greeks>
    /// - Original Source: N/A
    pub fn bs_epsilon(&self, black_scholes_type: &BlackScholesType) -> Result<f64, DigiFiError> {
        let (d_1, _) = self.european_option_black_scholes_params()?;
        let normal_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0)?;
        match black_scholes_type {
            BlackScholesType::Call => {
                Ok(-self.asset_price * self.time_to_maturity * (-self.yield_*self.time_to_maturity).exp() * normal_dist.cdf(d_1)?)
            },
            BlackScholesType::Put => {
                Ok(self.asset_price * self.time_to_maturity * (-self.yield_*self.time_to_maturity).exp() * normal_dist.cdf(-d_1)?)
            },
        }
    }

    /// Gamma of the European option.
    /// 
    /// Measure of sensitivity of the option price to change in the underlying asset price.
    ///
    /// # Output
    /// - Gamma
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Greeks_(finance)#Formulae_for_European_option_Greeks>
    /// - Original Source: N/A
    pub fn bs_gamma(&self) -> Result<f64, DigiFiError> {
        let (d_1, _) = self.european_option_black_scholes_params()?;
        let normal_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0)?;
        Ok((-self.yield_ * self.time_to_maturity).exp() * normal_dist.cdf(d_1)? / (self.asset_price * self.sigma * self.time_to_maturity.sqrt()))
    }

    /// Surface (i.e., underlying asset price vs option value vs time-to-maturity) of the option price as it approaches maturity.
    ///
    /// # Input
    /// - `start_price`: Underlying asset price from which to generate the axis
    /// - `stop_price`: Undelying asset price until which to generate the axis
    /// - `n_prices`: Number of prices to generate for the axis
    /// - `n_time_steps`: Number of time steps between starting time and maturity
    /// - `lattice_model_n_steps`: Number of steps to use in the lattice model (i.e., BINOMIAL and TRINOMIAL implementations)
    ///
    /// # Output
    /// - Data to plot the option value surface
    pub fn present_value_surface(&mut self, start_price: f64, stop_price: f64, n_prices: usize, n_time_steps: usize) -> Result<PresentValueSurface, DigiFiError> {
        // Preserve state parameters
        let initial_time_to_maturity: f64 = self.time_to_maturity;
        let initial_asset_price: f64 = self.asset_price;
        // Present value surface grid
        let times_to_maturity: Array1<f64> = Array1::linspace(self.time_to_maturity, 0.0, n_time_steps);
        let price_array: Array1<f64> = Array1::linspace(start_price, stop_price, n_prices);
        // Computation of present value for every point on the grid
        let mut option_pv_matrix: Vec<Array1<f64>> = Vec::with_capacity(n_time_steps);
        for i in 0..n_time_steps {
            let option_pv_array: Array1<f64>;
            if i == (n_time_steps - 1) {
                option_pv_array = self.payoff.payoff_iter(&price_array);
            } else {
                self.time_to_maturity = times_to_maturity[i];
                let mut present_values: Vec<f64> = Vec::with_capacity(n_prices);
                for j in 0..n_prices {
                    self.asset_price = price_array[j];
                    present_values.push(self.present_value()?);
                }
                option_pv_array = Array1::from_vec(present_values);
            }
            option_pv_matrix.push(option_pv_array);
        }
        // Reverse the state of the instance
        self.time_to_maturity = initial_time_to_maturity;
        self.asset_price = initial_asset_price;
        Ok(PresentValueSurface { times_to_maturity, price_array, pv_matrix: option_pv_matrix })
    }
}

impl ErrorTitle for OptionContract {
    fn error_title() -> String {
        String::from("Option Contract")
    }
}

impl FinancialInstrument for OptionContract {

    /// Present values of the option.
    ///
    /// # Output
    /// - Present value of the option
    fn present_value(&self) -> Result<f64, DigiFiError> {
        match &self.option_pricing_method {
            OptionPricingMethod::BlackScholes { type_ } => {
                self.european_option_black_scholes(type_)
            },
            OptionPricingMethod::Binomial { n_steps } => {
                let lattice_model: BrownianMotionBinomialModel = BrownianMotionBinomialModel::build(self.payoff.clone_box(), self.asset_price, self.time_to_maturity, self.discount_rate, self.sigma, self.yield_, *n_steps)?;
                match &self.option_type {
                    OptionType::European => lattice_model.european(),
                    OptionType::American => lattice_model.american(),
                    OptionType::Bermudan { exercise_time_steps } => lattice_model.bermudan(exercise_time_steps),
                }
            },
            OptionPricingMethod::Trinomial { n_steps } => {
                let lattice_model: BrownianMotionTrinomialModel = BrownianMotionTrinomialModel::build(self.payoff.clone_box(), self.asset_price, self.time_to_maturity, self.discount_rate, self.sigma, self.yield_, *n_steps)?;
                match &self.option_type {
                    OptionType::European => lattice_model.european(),
                    OptionType::American => lattice_model.american(),
                    OptionType::Bermudan { exercise_time_steps } => lattice_model.bermudan(exercise_time_steps),
                }
            },
        }
    }

    /// Net present value of the option.
    ///
    /// # Output
    /// - Present value of the option minus the initial price it took to purchase the option
    fn net_present_value(&self) -> Result<f64, DigiFiError> {
        Ok(-self.initial_option_price + self.present_value()?)
    }

    /// Future value of the option.
    ///
    /// # Output
    /// - Future value of the option at it maturity (Computed from the present value of the option)
    fn future_value(&self) -> Result<f64, DigiFiError> {
        Ok(self.present_value()? * (self.discount_rate * self.time_to_maturity).exp())
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

impl PortfolioInstrument for OptionContract {

    fn asset_name(&self) -> String {
        self.financial_instrument_id.identifier.clone()
    }

    fn historical_data(&self) -> &AssetHistData {
        &self.asset_historical_data
    }
}


#[cfg(feature = "plotly")]
/// Plots the present value/option premium surface (i.e., underlying asset price vs time-to-maturity vs option premium).
///
/// # Input
/// - `surface`: Present value surface for the option
///
/// # Output
/// - Plot of the present value/option premium surface
///
/// # Examples
///
/// ```rust,ignore
/// use digifi::utilities::time_value_utils::CompoundingType;
/// use digifi::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass, LongCall};
/// use digifi::financial_instruments::derivatives::{OptionType, OptionPricingMethod, OptionContract, PresentValueSurface};
/// use digifi::portfolio_applications::AssetHistData;
///
/// #[cfg(feature = "plotly")]
/// #[test]
/// fn plot_option_premium_surface() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_present_value_surface;
///
///     // Option definition
///     let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
///         instrument_type: FinancialInstrumentType::DerivativeInstrument, asset_class: AssetClass::EquityBasedInstrument,
///         identifier: String::from("32198407128904"),
///     };
///     let asset_historical_data: AssetHistData = AssetHistData::build(
///         Array1::from_vec(vec![0.4, 0.5]), Array1::from_vec(vec![0.0, 0.0]), Time::new(Array1::from_vec(vec![0.0, 1.0]))
///     ).unwrap();
///     let payoff: Box<LongCall> = Box::new(LongCall { k: 100.0, cost: 1.5 });
///     let option_pricing_method: OptionPricingMethod = OptionPricingMethod::Binomial { n_steps: 50 };
///     let mut option: OptionContract = OptionContract::build(
///         99.0, 100.0, 0.02, 0.0, 3.0, 0.2, OptionType::European, payoff, 1.5, option_pricing_method, financial_instrument_id,
///         asset_historical_data, None
///     ).unwrap();
///
///     // Plot
///     let surface: PresentValueSurface = option.present_value_surface(70.0, 130.0, 61, 30).unwrap();
///     let plot: Plot = plot_present_value_surface(&surface);
///     plot.show();
/// }
/// ```
pub fn plot_present_value_surface(surface: &PresentValueSurface) -> Plot {
    let x: Vec<f64> = surface.price_array.to_vec();
    let y: Vec<f64> = surface.times_to_maturity.to_vec();
    let z: Vec<Vec<f64>> = surface.pv_matrix.iter().map(|a| a.to_vec() ).collect();
    let mut plot: Plot = Plot::new();
    plot.add_trace(Surface::new(z).x(x).y(y).name("Option Premium Surface"));
    let x_axis: Axis = Axis::new().title(Title::from("Underlying Asset Price"));
    let y_axis: Axis = Axis::new().title(Title::from("Time to Maturity"));
    let z_axis: Axis = Axis::new().title(Title::from("Option Premium"));
    let layout_scene: LayoutScene = LayoutScene::new().x_axis(x_axis).y_axis(y_axis).z_axis(z_axis);
    let layout: Layout = Layout::new().scene(layout_scene).title("<b>Option Premium Surface</b>");
    plot.set_layout(layout);
    plot
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::utilities::{TEST_ACCURACY, Time, time_value_utils::CompoundingType};
    use crate::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass, LongCall};
    use crate::financial_instruments::derivatives::{ContractType, Contract, OptionType, OptionPricingMethod, BlackScholesType, OptionContract};
    use crate::portfolio_applications::AssetHistData;
    use crate::statistics::{ProbabilityDistribution, continuous_distributions::NormalDistribution};

    #[test]
    fn unit_test_black_scholes_formula() -> () {
        use crate::financial_instruments::derivatives::{BlackScholesType, black_scholes_formula};
        let value: f64 = black_scholes_formula(10.0, 11.0, 0.2, 1.0, 0.02, 0.0, &BlackScholesType::Call).unwrap();
        assert!((value - 0.49438669572304805).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_contract() -> () {
        // Contract definition
        let compounding_type: CompoundingType = CompoundingType::Continuous;
        let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
            instrument_type: FinancialInstrumentType::DerivativeInstrument, asset_class: AssetClass::EquityBasedInstrument,
            identifier: String::from("32198407128904"),
        };
        let asset_historical_data: AssetHistData = AssetHistData::build(
            Array1::from_vec(vec![0.4, 0.5]), Array1::from_vec(vec![0.0, 0.0]),
            Time::new(Array1::from_vec(vec![0.0, 1.0]))
        ).unwrap();
        let contract: Contract = Contract::build(
            ContractType::Future, 1.5, 100.0, 0.03, 2.0, 101.0, compounding_type, financial_instrument_id,
            asset_historical_data, None
        ).unwrap();
        // Theoretical value
        let forward_price: f64 = 101.0 * (0.03 * 2.0_f64).exp();
        let pv: f64 = (forward_price - 100.0) * (-0.03 * 2.0_f64).exp();
        assert!((contract.present_value().unwrap() - pv).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_option() -> () {
        // Option definition
        let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
            instrument_type: FinancialInstrumentType::DerivativeInstrument, asset_class: AssetClass::EquityBasedInstrument,
            identifier: String::from("32198407128904"),
        };
        let asset_historical_data: AssetHistData = AssetHistData::build(
            Array1::from_vec(vec![0.4, 0.5]), Array1::from_vec(vec![0.0, 0.0]),
            Time::new(Array1::from_vec(vec![0.0, 1.0]))
        ).unwrap();
        let payoff: Box<LongCall> = Box::new(LongCall { k: 100.0, cost: 1.5 });
        let option_pricing_method: OptionPricingMethod = OptionPricingMethod::BlackScholes { type_: BlackScholesType::Call };
        let option: OptionContract = OptionContract::build(
            99.0, 100.0, 0.02, 0.0, 3.0, 0.2, OptionType::European, payoff, 1.5, option_pricing_method,
            financial_instrument_id, asset_historical_data, None
        ).unwrap();
        // Theoretical value
        let normal_dist: NormalDistribution = NormalDistribution::build(0.0, 1.0).unwrap();
        let d_1_numerator: f64 = (99.0 / 100.0_f64).ln() + 3.0  * (0.02 - 0.0 + 0.5 * 0.2_f64.powi(2));
        let d_1: f64 = d_1_numerator / (0.2 * 3.0_f64.sqrt());
        let d_2: f64 = d_1 - 0.2 * 3.0_f64.sqrt();
        let component_1: f64 = 99.0 * (-0.0 * 3.0_f64).exp() * normal_dist.cdf(d_1).unwrap();
        let component_2: f64 = 100.0 * (-0.02 * 3.0_f64).exp() * normal_dist.cdf(d_2).unwrap();
        let pv: f64 = component_1 - component_2;
        assert!((option.present_value().unwrap() - pv).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_option_pricing_all() -> () {
        use crate::lattice_models::LatticeModel;
        use crate::lattice_models::trinomial_models::BrownianMotionTrinomialModel;
        use crate::lattice_models::binomial_models::BrownianMotionBinomialModel;
        use crate::financial_instruments::derivatives::{BlackScholesType, black_scholes_formula};
        use crate::stochastic_processes::standard_stochastic_models::GeometricBrownianMotion;
        use crate::monte_carlo::monte_carlo_simulation;
        use crate::financial_instruments::LongCall;
        let payoff: LongCall = LongCall { k: 105.0, cost: 0.0, };
        // Binomial model
        let bmbm: BrownianMotionBinomialModel = BrownianMotionBinomialModel::build(
            Box::new(payoff.clone()), 100.0, 1.0, 0.02, 0.15, 0.0, 1000
        ).unwrap();
        let predicted_value: f64 = bmbm.european().unwrap();
        println!("Binomial: {}", predicted_value);
        // Trinomial model
        let bmtm: BrownianMotionTrinomialModel = BrownianMotionTrinomialModel::build(
            Box::new(payoff.clone()), 100.0, 1.0, 0.02, 0.15, 0.0, 1000
        ).unwrap();
        let predicted_value: f64 = bmtm.european().unwrap();
        println!("Trinomial: {}", predicted_value);
        // Black-Scholes
        let predicted_value: f64 = black_scholes_formula(100.0, 105.0, 0.15, 1.0, 0.02, 0.0, &BlackScholesType::Call).unwrap();
        println!("Black-Scholes: {}", predicted_value);
        // Monte-Carlo
        // GBM definition (`mu` is set to the risk-free rate to account for the future value of assets's price)
        let gbm: GeometricBrownianMotion = GeometricBrownianMotion::new(0.02, 0.15, 1_000, 200, 1.0, 100.0);
        let predicted_value: f64 = monte_carlo_simulation(&gbm, &payoff, 0.02, &Some(vec![false; 200])).unwrap();
        println!("Monte-Carlo: {}", predicted_value);
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_option_plot() -> () {
        use plotly::Plot;
        use crate::financial_instruments::derivatives::{plot_present_value_surface, PresentValueSurface};
        // Option definition
        let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
            instrument_type: FinancialInstrumentType::DerivativeInstrument, asset_class: AssetClass::EquityBasedInstrument,
            identifier: String::from("32198407128904"),
        };
        let asset_historical_data: AssetHistData = AssetHistData::build(
            Array1::from_vec(vec![0.4, 0.5]), Array1::from_vec(vec![0.0, 0.0]),
            Time::new(Array1::from_vec(vec![0.0, 1.0]))
        ).unwrap();
        let payoff: Box<LongCall> = Box::new(LongCall { k: 100.0, cost: 1.5 });
        let option_pricing_method: OptionPricingMethod = OptionPricingMethod::Binomial { n_steps: 50 };
        let mut option: OptionContract = OptionContract::build(
            99.0, 100.0, 0.02, 0.0, 3.0, 0.2, OptionType::European, payoff, 1.5, option_pricing_method,
            financial_instrument_id, asset_historical_data, None
        ).unwrap();
        // Plot
        let surface: PresentValueSurface = option.present_value_surface(70.0, 130.0, 61, 30).unwrap();
        let plot: Plot = plot_present_value_surface(&surface);
        plot.show();
    }
}