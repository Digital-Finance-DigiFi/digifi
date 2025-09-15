//! # Financial Instruments
//! 
//! Provides utilies for pricing different types of financial instruments. This modules contains multiple payoff functions for pricing options and 
//! financial instrument structs like `Bond`, `Contract`, `OptionContract`, `ForwardRateAgreement` and `Stock` that implement methods for pricing
//! these instruments and performing their analysis. 


// Re-Exports
pub use self::bonds::{bootstrap, YtMMethod, BondType, Bond};
pub use self::derivatives::{
    minimum_variance_hedge_ratio, black_scholes_formula, ContractType, OptionType, BlackScholesType, OptionPricingMethod, PresentValueSurface, Contract,
    OptionContract,
};
pub use self::rates_and_swaps::ForwardRateAgreement;
pub use self::stocks::{QuoteValues, StockValuationType, ValuationByComparablesParams, Stock};


pub mod bonds;
pub mod derivatives;
pub mod rates_and_swaps;
pub mod stocks;


use ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
#[cfg(feature = "plotly")]
use plotly::{Plot, Scatter, Layout, layout::Axis};
use crate::error::{DigiFiError, ErrorTitle};
use crate::portfolio_applications::AssetHistData;
use crate::stochastic_processes::StochasticProcess;


#[derive(Clone, Copy, Debug)]
pub enum FinancialInstrumentType {
    /// Value of the instrument is determined by the markets (e.g., stocks, bonds, deposits, loans, checks)
    CashInstrument,
    /// Value of the instrument is determined by another instrument (e.g., options, futures, forwards)
    DerivativeInstrument,
}


#[derive(Clone, Copy, Debug)]
pub enum AssetClass {
    /// Loans, certificates of deposit, bank deposits, futures, forwards, options, bonds, mortgage-backed securities
    DebtBasedInstrument,
    /// Stocks - common and preferred, some ETFs, mutual funds, REITs, checks, limited partnerships
    EquityBasedInstrument,
    /// Forwards, futures, options, CFDs, swaps
    ForeignExchangeInstrument,
}


#[derive(Clone, Debug)]
/// Struct with general financial instrument data.
pub struct FinancialInstrumentId {
    /// Type of the financial instrument (i.e., cash instrument or derivative instrument)
    pub instrument_type: FinancialInstrumentType,
    /// Asset class of the instrument (i.e., debt-based, equity-based or foreign exchange instrument)
    pub asset_class: AssetClass,
    /// Financial instrument identifier (e.g., Financial Instrument Global Identifier (FIGI), International Securities Identification Number (ISIN))
    pub identifier: String,
}


pub trait FinancialInstrument {
    /// Computes the present value of the financial instrument.
    fn present_value(&self) -> Result<f64, DigiFiError>;

    /// Computes the net present value of the financial instrument.
    fn net_present_value(&self) -> Result<f64, DigiFiError>;

    /// Computes the future value of the financial instrument.
    fn future_value(&self) -> Result<f64, DigiFiError>;

    /// Returns asset's historical data.
    fn historical_data(&self) -> &AssetHistData;

    /// Updates historical data of the asset with the newly generated data.
    fn update_historical_data(&mut self, new_data: &AssetHistData) -> ();

    /// Returns a mutable reference to the stochastic process that simulates price action.
    fn stochastic_model(&mut self) -> &mut Option<Box<dyn StochasticProcess>>;

    /// Updates the number of paths the stochastic model will produce when called.
    /// 
    /// # Input
    /// - `n_paths`: New number of paths to use
    fn update_n_stochastic_paths(&mut self, n_paths: usize) -> () {
        match self.stochastic_model() {
            Some(sm) => sm.update_n_paths(n_paths),
            None => (),
        }
    }

    /// Simulates the paths of price action for the financial instrument.
    fn stochastic_simulation(&mut self) -> Result<Option<Vec<Array1<f64>>>, DigiFiError> {
        let paths: Vec<Array1<f64>> = match self.stochastic_model() {
            Some(sm) => sm.get_paths()?,
            None => return Ok(None),
        };
        Ok(Some(paths))
    }

    /// Genrates and updates the historica data about the asset.
    /// 
    /// # Input
    /// - `in_place`: If true, uses generated data to update the asset history data 
    fn generate_historic_data(&mut self, in_place: bool) -> Result<Option<AssetHistData>, DigiFiError> {
        match self.stochastic_model() {
            Some(sm) => {
                let prices: Array1<f64> = sm.get_paths()?.remove(0);
                let length: usize = prices.len();
                let new_data: AssetHistData = AssetHistData::build(
                    prices, Array1::from_vec(vec![0.0; length]), self.historical_data().time.clone(),
                )?;
                if in_place {
                    self.update_historical_data(&new_data);
                }
                Ok(Some(new_data))
            },
            None => Ok(None)
        }
    }
}


pub trait PayoffClone {
    fn clone_box(&self) -> Box<dyn Payoff>;
}

impl<T> PayoffClone for T
where
    T: 'static + Payoff + Clone,
{
    fn clone_box(&self) -> Box<dyn Payoff> {
        Box::new(self.clone())
    }
}


pub trait Payoff: PayoffClone {
    /// Payoff function.
    fn payoff(&self, s: f64) -> f64;

    /// Payoff function applied to the array of values.
    fn payoff_iter(&self, s: &Array1<f64>) -> Array1<f64> {
        s.map(|s| self.payoff(*s) )
    }

    /// Validation of payoff object to satisfy the computational requirements.
    /// 
    /// # Input
    /// - `length_value`: Number of test data points to validate payoff method on
    fn validate_payoff(&self, val_length: usize) -> Result<(), DigiFiError> {
        let s: Array1<f64> = Array1::from_vec(vec![1.0; val_length]);
        let result: Array1<f64> = self.payoff_iter(&s);
        if result.len() != val_length {
            return Err(DigiFiError::CustomFunctionLengthVal { title: "Validate Payoff".to_owned(), });
        }
        Ok(())
    }

    /// Profit function.
    /// 
    /// Profit = Payoff - Cost
    fn profit(&self, s: f64) -> f64;

    /// Profit function applied to the array of values.
    fn profit_iter(&self, s: &Array1<f64>) -> Array1<f64> {
        s.map(|s| self.profit(*s) )
    }

    /// Updates the cost/premium of the instrument.
    ///
    /// # Input
    /// - `new_cost`: Updated cost/premium of the instrument
    fn update_cost(&mut self, new_cost: f64) -> ();
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Long call payoff and profit.
pub struct LongCall {
    /// Strike price
    pub k: f64,
    /// Initial option price (cost)
    pub cost: f64,
}

impl Payoff for LongCall {
    /// Long call option payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the long call option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Long Call Payoff} = max(S_{t}-K, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, LongCall};
    ///
    /// let s: Array1<f64> = array![10.0, 9.0, 13.0];
    ///
    /// let long_call: LongCall = LongCall { k: 10.0, cost: 1.0 };
    ///
    /// assert!((long_call.payoff_iter(&s) - Array1::from_vec(vec![0.0, 0.0, 3.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        (s - self.k).max(0.0)
    }

    /// Long call option profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the long call option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Long Call Profit} = max(S_{t}-K, 0) - \\textit{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Short call payoff and profit.
pub struct ShortCall {
    /// Strike price
    pub k: f64,
    /// Initial option price (cost)
    pub cost: f64,
}

impl Payoff for ShortCall {
    /// Short call option payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the short call option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Short Call Payoff} = min(K-S_{t}, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, ShortCall};
    ///
    /// let s: Array1<f64> = array![10.0, 9.0, 13.0];
    ///
    /// let short_call: ShortCall = ShortCall { k: 10.0, cost: 1.0 };
    ///
    /// assert!((short_call.payoff_iter(&s) - Array1::from_vec(vec![0.0, 0.0, -3.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        (self.k - s).min(0.0)
    }

    /// Short call option profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the short call option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Short Call Profit} = min(K-S_{t}, 0) + \\textit{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) + self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Long put payoff and profit.
pub struct LongPut {
    /// Strike price
    pub k: f64,
    /// Initial option price (cost)
    pub cost: f64,
}

impl Payoff for LongPut {
    /// Long put option payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the long put option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Long Put Payoff} = max(K-S_{t}, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, LongPut};
    ///
    /// let s: Array1<f64> = array![10.0, 9.0, 13.0];
    ///
    /// let long_put: LongPut = LongPut { k: 10.0, cost: 1.0 };
    ///
    /// assert!((long_put.payoff_iter(&s) - Array1::from_vec(vec![0.0, 1.0, 0.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        (self.k - s).max(0.0)
    }

    /// Long put option profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the long put option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Long Put Profit} = max(K-S_{t}, 0) - \\textit{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Short put payoff and profit.
pub struct ShortPut {
    /// Strike price
    pub k: f64,
    /// Initial option price (cost)
    pub cost: f64,
}

impl Payoff for ShortPut {
    /// Short put option payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the short put option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Short Put Payoff} = min(S_{t}-K, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, ShortPut};
    ///
    /// let s: Array1<f64> = array![10.0, 9.0, 13.0];
    ///
    /// let short_put: ShortPut = ShortPut { k: 10.0, cost: 1.0 };
    ///
    /// assert!((short_put.payoff_iter(&s) - Array1::from_vec(vec![0.0, -1.0, 0.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        (s - self.k).min(0.0)
    }

    /// Short put option profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the short put option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Short Put Profit} = min(S_{t}-K, 0) + \\textit{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) + self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Bull collar strategy payoff and profit.
/// 
/// Bull Collar = Asset + Long Put + Short Call.
pub struct BullCollar {
    /// Long put option strike price
    k_p: f64,
    /// Short call option strike price
    k_c: f64,
    /// Initial option price (cost)
    cost: f64,
    /// Initial asset cost
    cost_s: f64,
}

impl BullCollar {
    /// Creates a new `BullCollar` instance.
    /// 
    /// # Input
    /// - `k_p`: Long put option strike price
    /// - `k_c`: Short call option strike price
    /// - `cost`: Initial option price (cost)
    /// - `cost_s`: Initial asset price (cost)
    /// 
    /// # Errors
    /// - Returns an error if the long put strike price is larger than the short call strike price.
    pub fn build(k_p: f64, k_c: f64, cost: f64, cost_s: f64) -> Result<Self, DigiFiError> {
        if k_c < k_p {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `k_p` must be smaller or equal to `k_c`.".to_owned(),
            });
        }
        Ok(BullCollar { k_p, k_c, cost, cost_s })
    }
}

impl ErrorTitle for BullCollar {
    fn error_title() -> String {
        String::from("Bull Collar")
    }
}

impl Payoff for BullCollar {
    /// Bull collar strategy payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the bull collar strategy
    /// 
    /// # LaTeX Formula
    /// - \\textit{Bull Collar Payoff} = S_{t} - S_{0} + max(K_{p}-S_{t}, 0) + min(K_{c}-S_{t}, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, BullCollar};
    ///
    /// let s: Array1<f64> = Array1::range(3.0, 7.0, 0.5);
    ///
    /// let bull_collar: BullCollar = BullCollar::build(4.0, 6.0, 0.0, 5.0).unwrap();
    ///
    /// assert!((bull_collar.payoff_iter(&s) - Array1::from_vec(vec![-1.0, -1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        s - self.cost_s + (self.k_p - s).max(0.0) + (self.k_c - s).min(0.0)
    }

    /// Bull collar strategy profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the bull collar strategy
    /// 
    /// # LaTeX Formula
    /// - \\textit{Bull Collar Profit} = S_{t} - S_{0} + max(K_{p}-S_{t}, 0) + min(K_{c}-S_{t}, 0) - \\textit{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Bear collar strategy payoff and profit.
/// 
/// Bear Collar = - Asset + Short Put + Long Call.
pub struct BearCollar {
    /// Short put option strike price
    k_p: f64,
    /// Long call option strike price
    k_c: f64,
    /// Initial option price (cost)
    cost: f64,
    /// Initial asset cost
    cost_s: f64,
}

impl BearCollar {
    /// Creates a new `BearCollar` instance.
    /// 
    /// # Input
    /// - `k_p`: Short put option strike price
    /// - `k_c`: Long call option strike price
    /// - `cost`: Initial option price (cost)
    /// - `cost_s`: Initial asset price (cost)
    /// 
    /// # Errors
    /// - Returns an error if the short put strike price is larger than the long call strike price.
    pub fn build(k_p: f64, k_c: f64, cost: f64, cost_s: f64) -> Result<Self, DigiFiError> {
        if k_c < k_p {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(), 
                constraint: "The argument `k_p` must be smaller or equal to `k_c`.".to_owned(),
            });
        }
        Ok(BearCollar { k_p, k_c, cost, cost_s })
    }
}

impl ErrorTitle for BearCollar {
    fn error_title() -> String {
        String::from("Bear Collar")
    }
}

impl Payoff for BearCollar {
    /// Bear collar strategy payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the bear collar strategy
    /// 
    /// # LaTeX Formula
    /// - \\textit{Bear Collar Payoff} = S_{0} - S_{t} + min(S_{t}-K_{p}, 0) + max(S_{t}-K_{c}, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, BearCollar};
    ///
    /// let s: Array1<f64> = Array1::range(3.0, 7.0, 0.5);
    ///
    /// let bear_collar: BearCollar = BearCollar::build(4.0, 6.0, 0.0, 5.0).unwrap();
    ///
    /// assert!((bear_collar.payoff_iter(&s) - Array1::from_vec(vec![1.0, 1.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        self.cost_s - s + (s - self.k_p).min(0.0) + (s - self.k_c).max(0.0)
    }

    /// Bear collar strategy profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the bear collar strategy
    /// 
    /// # LaTeX Formula
    /// - \\textit{Bear Collar Profit} = S_{0} - S_{t} + min(S_{t}-K_{p}, 0) + max(S_{t}-K_{c}, 0) - \\textit{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Bull spread option payoff and profit.
/// 
/// Bull Spread = Long Call + Short Call
pub struct BullSpread {
    /// Long call option strike price
    k_l: f64,
    /// Short call option strike price
    k_s: f64,
    /// Initial oprion price (cost)
    cost: f64,
}

impl BullSpread {
    /// Creates a new `BullSpread` instance.
    /// 
    /// # Input
    /// - `k_l`: Long call option strike price
    /// - `k_s`: Short call option strike price
    /// - `cost`: Initial option price (cost)
    /// 
    /// # Errors
    /// - Returns an error if the short call strike price is smaller than the long call strike price.
    pub fn build(k_l: f64, k_s: f64, cost: f64) -> Result<Self, DigiFiError> {
        if k_s < k_l {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `k_l` must be smaller or equal to `k_s`.".to_owned(),
            });
        }
        Ok(BullSpread { k_l, k_s, cost })
    }
}

impl ErrorTitle for BullSpread {
    fn error_title() -> String {
        String::from("Bull Spread")
    }
}

impl Payoff for BullSpread {
    /// Bull spread option payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the bull spread option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Bull Spread Payoff} = max(S_{t}-K_{l}, 0) + min(K_{s}-S_{t}, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, BullSpread};
    ///
    /// let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    ///
    /// let bull_spread: BullSpread = BullSpread::build(4.0, 6.0, 1.0).unwrap();
    ///
    /// assert!((bull_spread.payoff_iter(&s) - Array1::from_vec(vec![0.0, 0.0, 1.0, 2.0, 2.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        (s - self.k_l).max(0.0) + (self.k_s - s).min(0.0)
    }

    /// Bull spread option profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the bull spread option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Bull Spread Profit} = max(S_{t}-K_{l}, 0) + min(K_{s}-S_{t}, 0) - \\textit{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Bear spread option payoff and profit.
/// 
/// Bear Spread = Long Short + Short Short
pub struct BearSpread {
    /// Long put option strike price
    k_l: f64,
    /// Short put option strike price
    k_s: f64,
    /// Initial option price (cost)
    cost: f64,
}

impl BearSpread {
    /// Creates a new `BearSpread` instance.
    /// 
    /// # Input
    /// - `k_l`: Long put option strike price
    /// - `k_s`: Short put option strike price
    /// - `cost`: Initial option price (cost)
    /// 
    /// # Errors
    /// - Returns an error if the long put strike price is smaller than the short put strike price.
    pub fn build(k_l: f64, k_s: f64, cost: f64) -> Result<Self, DigiFiError> {
        if k_l < k_s {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `k_s` must be smaller or equal to `k_l`.".to_owned(),
            });
        }
        Ok(BearSpread { k_l, k_s, cost })
    }
}

impl ErrorTitle for BearSpread {
    fn error_title() -> String {
        String::from("Bear Spread")
    }
}

impl Payoff for BearSpread {
    /// Bear spread option payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the bear spread option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Bear Spread Payoff} = max(K_{l}-S_{t}, 0) + min(S_{t}-K_{s}, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, BearSpread};
    ///
    /// let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    ///
    /// let bear_spread: BearSpread = BearSpread::build(6.0, 4.0, 1.0).unwrap();
    ///
    /// assert!((bear_spread.payoff_iter(&s) - Array1::from_vec(vec![2.0, 2.0, 1.0, 0.0, 0.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        (self.k_l - s).max(0.0) + (s - self.k_s).min(0.0)
    }

    /// Bear spread option profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the bear spread option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Bear Spread Profit} = max(K_{l}-S_{t}, 0) + min(S_{t}-K_{s}, 0) - \\textit{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Butterfly spread option payoff and profit.
/// 
/// Buttefly Spread = Smaller Long Call + 2*Short Call + Larger Long Call
pub struct LongButterfly {
    /// Smaller long call option strike price
    k1_c: f64,
    /// Larger long call option strike price
    k2_c: f64,
    /// Short put option strike price
    k_p: f64,
    /// Initial option price (cost)
    cost: f64,
}

impl LongButterfly {
    /// Creates a new `LongButterfly` instance.
    /// 
    /// # Input
    /// - `k1_c`: Smaller long call option strike price
    /// - `k2_c`: Larger long call option strike price
    /// - `k_p`: Short put option strike price
    /// - `cost`: Initial option price (cost)
    /// 
    /// # Errors
    /// - Returns an error if the long put strike price is smaller than the short put strike price.
    pub fn build(k1_c: f64, k2_c: f64, k_p: f64, cost: f64) -> Result<Self, DigiFiError> {
        // Check that k1_c <= k_p <= k2_c
        if k2_c < k_p {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `k_p` must be smaller or equal to `k1_c`.".to_owned(),
            });
        }
        if k_p < k1_c {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `k1_c` must be smaller or equal to `k_p`.".to_owned(),
            });
        }
        Ok(LongButterfly { k1_c, k2_c, k_p, cost })
    }
}

impl ErrorTitle for LongButterfly {
    fn error_title() -> String {
        String::from("Long Butterfly")
    }
}

impl Payoff for LongButterfly {
    /// Long butterfly option payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the long butterfly option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Butterfly Spread Payoff} = max(S_{t}-K1_{c}, 0) + 2min(K_{p}-S_{t}, 0) + max(S_{t}-K2_{c}, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, LongButterfly};
    ///
    /// let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    ///
    /// let long_butterfly: LongButterfly = LongButterfly::build(4.0, 6.0, 5.0, 1.0).unwrap();
    ///
    /// assert!((long_butterfly.payoff_iter(&s) - Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        (s - self.k1_c).max(0.0) + 2.0*(self.k_p - s).min(0.0) + (s - self.k2_c).max(0.0)
    }

    /// Long butterfly option profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the long butterfly option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Butterfly Spread Payoff} = max(S_{t}-K1_{c}, 0) + 2min(K_{p}-S_{t}, 0) + max(S_{t}-K2_{c}, 0) - \\textit{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Box spread option payoff and profit.
/// 
/// Box Spread = Long Call + Short Call + Long Put + Short Put = Bull Spread + Bear Spread
pub struct BoxSpread {
    /// Smaller option strike price
    k_1: f64,
    /// Larger option strike price
    k_2: f64,
    /// Initial option price (cost)
    cost: f64,
}

impl BoxSpread {
    /// Creates a new `BoxSpread` instance.
    /// 
    /// # Input
    /// - `k_1`: Smaller option strike price
    /// - `k_2`: Larger option strike price
    /// - `cost`: Initial option price (cost)
    /// 
    /// # Errors
    /// - Returns an error if the k_2 strike price is smaller than the k_1 strike price.
    pub fn build(k_1: f64, k_2: f64, cost: f64) -> Result<Self, DigiFiError> {
        if k_2 < k_1 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `k_1` must be smaller or equal to `k_2`.".to_owned(),
            });
        }
        Ok(BoxSpread { k_1, k_2, cost })
    }
}

impl ErrorTitle for BoxSpread {
    fn error_title() -> String {
        String::from("Box Spread")
    }
}

impl Payoff for BoxSpread {
    /// Box spread option payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the box spread option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Box Spread Payoff} = max(S_{t}-K_{1}, 0) + min(K_{2}-S_{t}, 0) + max(K_{2}-S_{t}, 0) + min(S_{t}-K_{1}, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, BoxSpread};
    ///
    /// let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    ///
    /// let box_spread: BoxSpread = BoxSpread::build(4.0, 6.0, 1.0).unwrap();
    ///
    /// assert!((box_spread.payoff_iter(&s) - Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0, 2.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        (s - self.k_1).max(0.0) + (self.k_2 - s).min(0.0) + (self.k_2 - s).max(0.0) + (s - self.k_1).min(0.0)
    }

    /// Box spread option profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the box spread option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Box Spread Payoff} = max(S_{t}-K_{1}, 0) + min(K_{2}-S_{t}, 0) + max(K_{2}-S_{t}, 0) + min(S_{t}-K_{1}, 0) - \\textit{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Straddle option payoff.
/// 
/// Straddle = Long Call + Long Put
pub struct Straddle {
    /// Option strike price
    pub k: f64,
    /// Initial option price (cost)
    pub cost: f64,
}

impl Payoff for Straddle {
    /// Straddle option payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the straddle option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Straddle Payoff} = max(S_{t}-K, 0) + max(K-S_{t}, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, Straddle};
    ///
    /// let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    ///
    /// let straddle: Straddle = Straddle { k: 5.0, cost: 1.0 };
    ///
    /// assert!((straddle.payoff_iter(&s) - Array1::from_vec(vec![2.0, 1.0, 0.0, 1.0, 2.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        (s - self.k).max(0.0) + (self.k - s).max(0.0)
    }

    /// Straddle option profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the straddle option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Straddle Profit} = max(S_{t}-K, 0) + max(K-S_{t}, 0) - \\textit{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Strangle option payoff.
/// 
/// Strangle = Long Call + Long Put
pub struct Strangle {
    /// Call option strike price
    k_c: f64,
    /// Put option strike price
    k_p: f64,
    /// Initial option price (cost)
    cost: f64,
}

impl Strangle {
    /// Creates a new `Strangle` instance.
    /// 
    /// # Input
    /// - `k_c`: Call option strike price
    /// - `k_p`: Put option strike price
    /// - `cost`: Initial option price (cost)
    /// 
    /// # Errors
    /// - Returns an error if the `k_c` strike price is smaller or equal to the `k_p` strike price.
    pub fn build(k_c: f64, k_p: f64, cost: f64) -> Result<Self, DigiFiError> {
        if k_c <= k_p {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `k_p` must be smaller than `k_c`.".to_owned(),
            });
        }
        Ok(Strangle { k_c, k_p, cost })
    }
}

impl ErrorTitle for Strangle {
    fn error_title() -> String {
        String::from("Strangle")
    }
}

impl Payoff for Strangle {
    /// Strangle option payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the strangle option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Strangle Payoff} = max(S_{t}-K_{c}, 0) + max(K_{p}-S_{t}, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, Strangle};
    ///
    /// let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    ///
    /// let strangle: Strangle = Strangle::build(6.0, 4.0, 1.0).unwrap();
    ///
    /// assert!((strangle.payoff_iter(&s) - Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        (s - self.k_c).max(0.0) + (self.k_p - s).max(0.0)
    }

    /// Strangle option profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the strangle option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Strangle Profit} = max(S_{t}-K_{c}, 0) + max(K_{p}-S_{t}, 0) - \\textit{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Strip option payoff.
/// 
/// Strip = Long Call + 2*Long Put
pub struct Strip {
    /// Option strike price
    pub k: f64,
    /// Initial option price (cost)
    pub cost: f64,
}

impl Payoff for Strip {
    /// Strip option payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the strip option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Strip Payoff} = max(S_{t}-K, 0) + 2max(K-S_{t}, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, Strip};
    ///
    /// let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    ///
    /// let strip: Strip = Strip { k: 5.0, cost: 1.0 };
    ///
    /// assert!((strip.payoff_iter(&s) - Array1::from_vec(vec![4.0, 2.0, 0.0, 1.0, 2.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        (s - self.k).max(0.0) + 2.0*(self.k - s).max(0.0)
    }

    /// Strip option profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the strip option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Strip Profit} = max(S_{t}-K, 0) + 2max(K-S_{t}, 0) - \\text{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Strap option payoff.
/// 
/// Strap = 2*Long Call + Long Put
pub struct Strap {
    /// Option strike price
    pub k: f64,
    /// Initial option price (cost)
    pub cost: f64,
}

impl Payoff for Strap {
    /// Strap option payoff.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the strap option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Strap Payoff} = 2max(S_{t}-K, 0) + max(K-S_{t}, 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, array};
    /// use digifi::utilities::TEST_ACCURACY;
    /// use digifi::financial_instruments::{Payoff, Strap};
    ///
    /// let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    ///
    ///  let strap: Strap = Strap { k: 5.0, cost: 1.0 };
    ///
    ///  assert!((strap.payoff_iter(&s) - Array1::from_vec(vec![2.0, 1.0, 0.0, 2.0, 4.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: f64) -> f64 {
        2.0*(s - self.k).max(0.0) + (self.k - s).max(0.0)
    }

    /// Strap option profit.
    /// 
    /// # Input
    /// - `s`: Underlying asset price
    /// 
    /// # Output
    /// - Profit of the strap option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Strap Profit} = 2max(S_{t}-K, 0) + max(K-S_{t}, 0) - \\text{Cost}
    fn profit(&self, s: f64) -> f64 {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[cfg(feature = "plotly")]
/// Plots the payoff function.
///
/// # Input
/// - `payoff_obj`: An instrument with a payoff function
/// - `start_price`: Start of the price range (i.e., the start point of the x-axis)
/// - `stop_price`: End of the price range (i.e., the end point of the x-axis)
/// - `n_points`: Number of points to produce on the plot
///
/// # Output
/// - Plot of the payoff function
///
/// # Examples
///
/// ```rust,ignore
/// use digifi::financial_instruments::Strangle;
///
/// #[cfg(feature = "plotly")]
/// fn test_plot_payoff() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_payoff;
///     
///     let payoff_obj: Strangle = Strangle::build(6.0, 4.0, 1.0).unwrap();
///     let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
///     payoff_plot.show();
/// }
/// ```
pub fn plot_payoff(payoff_obj: &impl Payoff, start_price: f64, stop_price: f64, n_points: usize) -> Plot {
    let prices: Array1<f64> = Array1::linspace(start_price, stop_price, n_points);
    let payoff: Array1<f64> = payoff_obj.payoff_iter(&prices);
    let mut plot: Plot = Plot::new();
    plot.add_trace(Scatter::new(prices.to_vec(), payoff.to_vec()));
    let x_axis: Axis = Axis::new().title("Price at Maturity");
    let y_axis: Axis = Axis::new().title("Payoff at Maturity");
    let layout: Layout = Layout::new().title("<b>Payoff Plot</b>").x_axis(x_axis).y_axis(y_axis);
    plot.set_layout(layout);
    plot
}


#[cfg(feature = "plotly")]
/// Plots the profit function.
///
/// # Input
/// - `payoff_obj`: An instrument with a profit function
/// - `start_price`: Start of the price range (i.e., the start point of the x-axis)
/// - `stop_price`: End of the price range (i.e., the end point of the x-axis)
/// - `n_points`: Number of points to produce on the plot
///
/// # Output
/// - Plot of the profit function
///
/// # Examples
///
/// ```rust,ignore
/// use digifi::financial_instruments::Strangle;
///
/// #[cfg(feature = "plotly")]
/// fn test_plot_payoff() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_profit;
///     
///     let payoff_obj: Strangle = Strangle::build(6.0, 4.0, 1.0).unwrap();
///     let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
///     profit_plot.show();
/// }
/// ```
pub fn plot_profit(payoff_obj: &impl Payoff, start_price: f64, stop_price: f64, n_points: usize) -> Plot {
    let prices: Array1<f64> = Array1::linspace(start_price, stop_price, n_points);
    let profit: Array1<f64> = payoff_obj.profit_iter(&prices);
    let mut plot: Plot = Plot::new();
    plot.add_trace(Scatter::new(prices.to_vec(), profit.to_vec()));
    let x_axis: Axis = Axis::new().title("Price at Maturity");
    let y_axis: Axis = Axis::new().title("Profit at Maturity");
    let layout: Layout = Layout::new().title("<b>Profit Plot</b>").x_axis(x_axis).y_axis(y_axis);
    plot.set_layout(layout);
    plot
}


#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};
    #[cfg(feature = "plotly")]
    use plotly::Plot;
    use crate::utilities::TEST_ACCURACY;
    use crate::financial_instruments::Payoff;

    #[test]
    fn unit_test_long_call() -> () {
        use crate::financial_instruments::LongCall;
        let s: Array1<f64> = array![10.0, 9.0, 13.0];
        let long_call: LongCall = LongCall { k: 10.0, cost: 1.0 };
        assert!((long_call.payoff_iter(&s) - Array1::from_vec(vec![0.0, 0.0, 3.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_short_call() -> () {
        use crate::financial_instruments::ShortCall;
        let s: Array1<f64> = array![10.0, 9.0, 13.0];
        let short_call: ShortCall = ShortCall { k: 10.0, cost: 1.0 };
        assert!((short_call.payoff_iter(&s) - Array1::from_vec(vec![0.0, 0.0, -3.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_long_put() -> () {
        use crate::financial_instruments::LongPut;
        let s: Array1<f64> = array![10.0, 9.0, 13.0];
        let long_put: LongPut = LongPut { k: 10.0, cost: 1.0 };
        assert!((long_put.payoff_iter(&s) - Array1::from_vec(vec![0.0, 1.0, 0.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_short_put() -> () {
        use crate::financial_instruments::ShortPut;
        let s: Array1<f64> = array![10.0, 9.0, 13.0];
        let short_put: ShortPut = ShortPut { k: 10.0, cost: 1.0 };
        assert!((short_put.payoff_iter(&s) - Array1::from_vec(vec![0.0, -1.0, 0.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_bull_collar() -> () {
        use crate::financial_instruments::BullCollar;
        let s: Array1<f64> = Array1::range(3.0, 7.0, 0.5);
        let bull_collar: BullCollar = BullCollar::build(4.0, 6.0, 0.0, 5.0).unwrap();
        assert!((bull_collar.payoff_iter(&s) - Array1::from_vec(vec![-1.0, -1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_bear_collar() -> () {
        use crate::financial_instruments::BearCollar;
        let s: Array1<f64> = Array1::range(3.0, 7.0, 0.5);
        let bear_collar: BearCollar = BearCollar::build(4.0, 6.0, 0.0, 5.0).unwrap();
        assert!((bear_collar.payoff_iter(&s) - Array1::from_vec(vec![1.0, 1.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_bull_spread() -> () {
        use crate::financial_instruments::BullSpread;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let bull_spread: BullSpread = BullSpread::build(4.0, 6.0, 1.0).unwrap();
        assert!((bull_spread.payoff_iter(&s) - Array1::from_vec(vec![0.0, 0.0, 1.0, 2.0, 2.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_bear_spread() -> () {
        use crate::financial_instruments::BearSpread;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let bear_spread: BearSpread = BearSpread::build(6.0, 4.0, 1.0).unwrap();
        assert!((bear_spread.payoff_iter(&s) - Array1::from_vec(vec![2.0, 2.0, 1.0, 0.0, 0.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_long_butterfly() -> () {
        use crate::financial_instruments::LongButterfly;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let long_butterfly: LongButterfly = LongButterfly::build(4.0, 6.0, 5.0, 1.0).unwrap();
        assert!((long_butterfly.payoff_iter(&s) - Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_box_spread() -> () {
        use crate::financial_instruments::BoxSpread;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let box_spread: BoxSpread = BoxSpread::build(4.0, 6.0, 1.0).unwrap();
        assert!((box_spread.payoff_iter(&s) - Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0, 2.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_straddle() -> () {
        use crate::financial_instruments::Straddle;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let straddle: Straddle = Straddle { k: 5.0, cost: 1.0 };
        assert!((straddle.payoff_iter(&s) - Array1::from_vec(vec![2.0, 1.0, 0.0, 1.0, 2.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_strangle() -> () {
        use crate::financial_instruments::Strangle;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let strangle: Strangle = Strangle::build(6.0, 4.0, 1.0).unwrap();
        assert!((strangle.payoff_iter(&s) - Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_strip() -> () {
        use crate::financial_instruments::Strip;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let strip: Strip = Strip { k: 5.0, cost: 1.0 };
        assert!((strip.payoff_iter(&s) - Array1::from_vec(vec![4.0, 2.0, 0.0, 1.0, 2.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_strap() -> () {
        use crate::financial_instruments::Strap;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let strap: Strap = Strap { k: 5.0, cost: 1.0 };
        assert!((strap.payoff_iter(&s) - Array1::from_vec(vec![2.0, 1.0, 0.0, 2.0, 4.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_long_call_plot() -> () {
        use crate::financial_instruments::{LongCall, plot_payoff, plot_profit};
        let payoff_obj: LongCall = LongCall { k: 50.0, cost: 1.0 };
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 30.0, 70.0, 41);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 30.0, 70.0, 41);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_short_call_plot() -> () {
        use crate::financial_instruments::{ShortCall, plot_payoff, plot_profit};
        let payoff_obj: ShortCall = ShortCall { k: 50.0, cost: 1.0 };
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 30.0, 70.0, 41);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 30.0, 70.0, 41);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_long_put_plot() -> () {
        use crate::financial_instruments::{LongPut, plot_payoff, plot_profit};
        let payoff_obj: LongPut = LongPut { k: 50.0, cost: 1.0 };
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 30.0, 70.0, 41);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 30.0, 70.0, 41);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_short_put_plot() -> () {
        use crate::financial_instruments::{ShortPut, plot_payoff, plot_profit};
        let payoff_obj: ShortPut = ShortPut { k: 50.0, cost: 1.0 };
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 30.0, 70.0, 41);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 30.0, 70.0, 41);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_bull_collar_plot() -> () {
        use crate::financial_instruments::{BullCollar, plot_payoff, plot_profit};
        let payoff_obj: BullCollar = BullCollar::build(4.0, 6.0, 1.0, 5.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_bear_collar_plot() -> () {
        use crate::financial_instruments::{BearCollar, plot_payoff, plot_profit};
        let payoff_obj: BearCollar = BearCollar::build(4.0, 6.0, 1.0, 5.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_bull_spread_plot() -> () {
        use crate::financial_instruments::{BullSpread, plot_payoff, plot_profit};
        let payoff_obj: BullSpread = BullSpread::build(4.0, 6.0, 1.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_bear_spread_plot() -> () {
        use crate::financial_instruments::{BearSpread, plot_payoff, plot_profit};
        let payoff_obj: BearSpread = BearSpread::build(6.0, 4.0, 1.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_long_butterfly_plot() -> () {
        use crate::financial_instruments::{LongButterfly, plot_payoff, plot_profit};
        let payoff_obj: LongButterfly = LongButterfly::build(4.0, 6.0, 5.0, 1.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_box_spread_plot() -> () {
        use crate::financial_instruments::{BoxSpread, plot_payoff, plot_profit};
        let payoff_obj: BoxSpread = BoxSpread::build(4.0, 6.0, 1.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_straddle_plot() -> () {
        use crate::financial_instruments::{Straddle, plot_payoff, plot_profit};
        let payoff_obj: Straddle = Straddle { k: 5.0, cost: 1.0 };
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_strangle_plot() -> () {
        use crate::financial_instruments::{Strangle, plot_payoff, plot_profit};
        let payoff_obj: Strangle = Strangle::build(6.0, 4.0, 1.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_strip_plot() -> () {
        use crate::financial_instruments::{Strip, plot_payoff, plot_profit};
        let payoff_obj: Strip = Strip { k: 5.0, cost: 1.0 };
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_strap_plot() -> () {
        use crate::financial_instruments::{Strap, plot_payoff, plot_profit};
        let payoff_obj: Strap = Strap { k: 5.0, cost: 1.0 };
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }
}