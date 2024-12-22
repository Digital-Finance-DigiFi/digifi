// Re-Exports
pub use self::bonds::{bootstrap, YtMMethod, BondType, Bond};
pub use self::derivatives::{
    minimum_variance_hedge_ratio, ContractType, OptionType, BlackScholesType, OptionPricingMethod, PresentValueSurface, Contract, OptionContract,
};
pub use self::rates_and_swaps::ForwardRateAgreement;
pub use self::stocks::{QuoteValues, StockValuationType, ValuationByComparablesParams, Stock};


pub mod bonds;
pub mod derivatives;
pub mod rates_and_swaps;
pub mod stocks;


use std::io::Error;
use ndarray::Array1;
#[cfg(feature = "plotly")]
use plotly::{Plot, Scatter, Layout, layout::Axis};
use crate::utilities::{input_error, data_error};
use crate::portfolio_applications::AssetHistData;


pub enum FinancialInstrumentType {
    /// Value of the instrument is determined by the markets (e.g., stocks, bonds, deposits, loans, checks)
    CashInstrument,
    /// Value of the instrument is determined by another instrument (e.g., options, futures, forwards)
    DerivativeInstrument,
}


pub enum AssetClass {
    /// Loans, certificates of deposit, bank deposits, futures, forwards, options, bonds, mortgage-backed securities
    DebtBasedInstrument,
    /// Stocks - common and preferred, some ETFs, mutual funds, REITs, checks, limited partnerships
    EquityBasedInstrument,
    /// Forwards, futures, options, CFDs, swaps
    ForeignExchangeInstrument,
}


/// # Description
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
    /// # Description
    /// Computes the present value of the financial instrument.
    fn present_value(&self) -> Result<f64, Box<dyn std::error::Error>>;

    /// # Description
    /// Computes the net present value of the financial instrument.
    fn net_present_value(&self) -> Result<f64, Box<dyn std::error::Error>>;

    /// # Description
    /// Computes the future value of the financial instrument.
    fn future_value(&self) -> Result<f64, Box<dyn std::error::Error>>;

    /// # Description
    /// Returns an array of asset prices.
    ///
    /// # Input
    /// - `end_index`: Time index beyond which no data will be returned
    /// - `start_index`: Time index below which no data will be returned
    fn get_prices(&self, end_index: usize, start_index: Option<usize>) -> Result<Array1<f64>, Error>;

    /// # Description
    /// Returns an array of predictable incomes for an asset (i.e., dividends, coupons, etc.).
    ///
    /// # Input
    /// - `end_index`: Time index beyond which no data will be returned
    /// - `start_index`: Time index below which no data will be returned
    fn get_predictable_income(&self, end_index: usize, start_index: Option<usize>) -> Result<Array1<f64>, Error>;

    /// # Description
    /// Returns an array of time steps at which the asset price and predictable_income are recorded.
    fn get_time_array(&self) -> Array1<f64>;

    /// # Description
    /// Updates the number of paths the stochastic model will produce when called.
    fn update_n_stochastic_paths(&mut self, n_paths: usize) -> ();

    /// # Description
    /// Simulates the paths of price action for the financial instrument.
    fn stochastic_simulation(&self) -> Result<Vec<Array1<f64>>, Error>;

    /// # Description
    /// Genrates and updates the historica data about the asset.
    fn generate_historic_data(&mut self, in_place: bool) -> Result<AssetHistData, Error>;
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
    /// # Description
    /// Payoff function.
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64>;

    /// # Description
    /// Validation of payoff object to satisfy the computational requirements.
    /// 
    /// # Input
    /// - `length_value`: Number of test data points to validate payoff method on
    fn validate_payoff(&self, val_length: usize) -> Result<(), Error> {
        let s: Array1<f64> = Array1::from_vec(vec![1.0; val_length]);
        let result: Array1<f64> = self.payoff(&s);
        if result.len() != val_length {
            return Err(data_error(format!("The payoff does not produce an array of length {}.", val_length)));
        }
        Ok(())
    }

    /// # Description
    /// Profit function.
    /// 
    /// Profit = Payoff - Cost
    fn profit(&self, s: &Array1<f64>) -> Array1<f64>;

    /// # Description
    /// Updates the cost/premium of the instrument.
    ///
    /// # Input
    /// - `new_cost`: Updated cost/premium of the instrument
    fn update_cost(&mut self, new_cost: f64) -> ();
}


#[derive(Clone)]
/// # Description
/// Long call payoff and profit.
pub struct LongCall {
    /// Strike price
    pub k: f64,
    /// Initial option price (cost)
    pub cost: f64,
}

impl Payoff for LongCall {
    //// # Description
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
    /// assert!((long_call.payoff(&s) - Array1::from_vec(vec![0.0, 0.0, 3.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (s - self.k).map(| p | p.max(0.0))
    }

    /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone)]
/// # Description
/// Short call payoff and profit.
pub struct ShortCall {
    /// Strike price
    pub k: f64,
    /// Initial option price (cost)
    pub cost: f64,
}

impl Payoff for ShortCall {
    /// # Description
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
    /// assert!((short_call.payoff(&s) - Array1::from_vec(vec![0.0, 0.0, -3.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (self.k - s).map(| p | p.min(0.0))
    }

    /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) + self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone)]
/// # Description
/// Long put payoff and profit.
pub struct LongPut {
    /// Strike price
    pub k: f64,
    /// Initial option price (cost)
    pub cost: f64,
}

impl Payoff for LongPut {
    /// # Description
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
    /// assert!((long_put.payoff(&s) - Array1::from_vec(vec![0.0, 1.0, 0.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (self.k - s).map(| p | p.max(0.0))
    }

    /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone)]
/// # Description
/// Short put payoff and profit.
pub struct ShortPut {
    /// Strike price
    pub k: f64,
    /// Initial option price (cost)
    pub cost: f64,
}

impl Payoff for ShortPut {
    /// # Description
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
    /// assert!((short_put.payoff(&s) - Array1::from_vec(vec![0.0, -1.0, 0.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (s - self.k).map(| p | p.min(0.0))
    }

    /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) + self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone)]
/// # Description
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
    /// # Description
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
    pub fn new(k_p: f64, k_c: f64, cost: f64, cost_s: f64) -> Result<Self, Error> {
        if k_c < k_p {
            return Err(input_error("Bull Collar: The argument k_p must be smaller or equal to k_c."));
        }
        Ok(BullCollar { k_p, k_c, cost, cost_s })
    }
}

impl Payoff for BullCollar {
    /// # Description
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
    /// let bull_collar: BullCollar = BullCollar::new(4.0, 6.0, 0.0, 5.0).unwrap();
    ///
    /// assert!((bull_collar.payoff(&s) - Array1::from_vec(vec![-1.0, -1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        s - self.cost_s + (self.k_p - s).map(| p | p.max(0.0)) + (self.k_c - s).map(| p | p.min(0.0))
    }

    /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone)]
/// # Description
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
    /// # Description
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
    pub fn new(k_p: f64, k_c: f64, cost: f64, cost_s: f64) -> Result<Self, Error> {
        if k_c < k_p {
            return Err(input_error("Bear Collar: The argument k_p must be smaller or equal to k_c."));
        }
        Ok(BearCollar { k_p, k_c, cost, cost_s })
    }
}

impl Payoff for BearCollar {
    /// # Description
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
    /// let bear_collar: BearCollar = BearCollar::new(4.0, 6.0, 0.0, 5.0).unwrap();
    ///
    /// assert!((bear_collar.payoff(&s) - Array1::from_vec(vec![1.0, 1.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        self.cost_s - s + (s - self.k_p).map(| p | p.min(0.0)) + (s - self.k_c).map(| p | p.max(0.0))
    }

     /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone)]
/// # Description
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
    /// # Description
    /// Creates a new `BullSpread` instance.
    /// 
    /// # Input
    /// - `k_l`: Long call option strike price
    /// - `k_s`: Short call option strike price
    /// - `cost`: Initial option price (cost)
    /// 
    /// # Errors
    /// - Returns an error if the short call strike price is smaller than the long call strike price.
    pub fn new(k_l: f64, k_s: f64, cost: f64) -> Result<Self, Error> {
        if k_s < k_l {
            return Err(input_error("Bull Spread: The argument k_l must be smaller or equal to k_s."));
        }
        Ok(BullSpread { k_l, k_s, cost })
    }
}

impl Payoff for BullSpread {
    /// # Description
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
    /// let bull_spread: BullSpread = BullSpread::new(4.0, 6.0, 1.0).unwrap();
    ///
    /// assert!((bull_spread.payoff(&s) - Array1::from_vec(vec![0.0, 0.0, 1.0, 2.0, 2.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (s - self.k_l).map(| p | p.max(0.0)) + (self.k_s - s).map(| p | p.min(0.0))
    }

    /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone)]
/// # Description
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
    /// # Description
    /// Creates a new `BearSpread` instance.
    /// 
    /// # Input
    /// - `k_l`: Long put option strike price
    /// - `k_s`: Short put option strike price
    /// - `cost`: Initial option price (cost)
    /// 
    /// # Errors
    /// - Returns an error if the long put strike price is smaller than the short put strike price.
    pub fn new(k_l: f64, k_s: f64, cost: f64) -> Result<Self, Error> {
        if k_l < k_s {
            return Err(input_error("Bear Spread: The argument k_s must be smaller or equal to k_l."));
        }
        Ok(BearSpread { k_l, k_s, cost })
    }
}

impl Payoff for BearSpread {
    /// # Description
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
    /// let bear_spread: BearSpread = BearSpread::new(6.0, 4.0, 1.0).unwrap();
    ///
    /// assert!((bear_spread.payoff(&s) - Array1::from_vec(vec![2.0, 2.0, 1.0, 0.0, 0.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (self.k_l - s).map(| p | p.max(0.0)) + (s - self.k_s).map(| p | p.min(0.0))
    }

    /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone)]
/// # Description
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
    /// # Description
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
    pub fn new(k1_c: f64, k2_c: f64, k_p: f64, cost: f64) -> Result<Self, Error> {
        // Check that k1_c <= k_p <= k2_c
        if k2_c < k_p {
            return Err(input_error("Long Butterfly: The argument k_p must be smaller or equal to k1_c."));
        }
        if k_p < k1_c {
            return Err(input_error("Long Butterfly: The argument k1_c must be smaller or equal to k_p."));
        }
        Ok(LongButterfly { k1_c, k2_c, k_p, cost })
    }
}

impl Payoff for LongButterfly {
    /// # Description
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
    /// let long_butterfly: LongButterfly = LongButterfly::new(4.0, 6.0, 5.0, 1.0).unwrap();
    ///
    /// assert!((long_butterfly.payoff(&s) - Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (s - self.k1_c).map(| p | p.max(0.0)) + 2.0*(self.k_p - s).map(| p | p.min(0.0)) + (s - self.k2_c).map(| p | p.max(0.0))
    }

    /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone)]
/// # Description
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
    /// # Description
    /// Creates a new `BoxSpread` instance.
    /// 
    /// # Input
    /// - `k_1`: Smaller option strike price
    /// - `k_2`: Larger option strike price
    /// - `cost`: Initial option price (cost)
    /// 
    /// # Errors
    /// - Returns an error if the k_2 strike price is smaller than the k_1 strike price.
    pub fn new(k_1: f64, k_2: f64, cost: f64) -> Result<Self, Error> {
        if k_2 < k_1 {
            return Err(input_error("Box Spread: The argument k_1 must be smaller or equal to k_2."));
        }
        Ok(BoxSpread { k_1, k_2, cost })
    }
}

impl Payoff for BoxSpread {
    /// # Description
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
    /// let box_spread: BoxSpread = BoxSpread::new(4.0, 6.0, 1.0).unwrap();
    ///
    /// assert!((box_spread.payoff(&s) - Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0, 2.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (s - self.k_1).map(| p | p.max(0.0)) + (self.k_2 - s).map(| p | p.min(0.0)) + (self.k_2 - s).map(| p | p.max(0.0)) + (s - self.k_1).map(| p | p.min(0.0))
    }

    /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone)]
/// # Description
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
    /// # Description
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
    /// assert!((straddle.payoff(&s) - Array1::from_vec(vec![2.0, 1.0, 0.0, 1.0, 2.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (s - self.k).map(| p | p.max(0.0)) + (self.k - s).map(| p | p.max(0.0))
    }

    /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone)]
/// # Description
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
    /// # Description
    /// Creates a new `Strangle` instance.
    /// 
    /// # Input
    /// - `k_c`: Call option strike price
    /// - `k_p`: Put option strike price
    /// - `cost`: Initial option price (cost)
    /// 
    /// # Errors
    /// - Returns an error if the `k_c` strike price is smaller or equal to the `k_p` strike price.
    pub fn new(k_c: f64, k_p: f64, cost: f64) -> Result<Self, Error> {
        if k_c <= k_p {
            return Err(input_error("Strangle: The argument k_p must be smaller than k_c."));
        }
        Ok(Strangle { k_c, k_p, cost })
    }
}

impl Payoff for Strangle {
    /// # Description
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
    /// let strangle: Strangle = Strangle::new(6.0, 4.0, 1.0).unwrap();
    ///
    /// assert!((strangle.payoff(&s) - Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (s - self.k_c).map(| p | p.max(0.0)) + (self.k_p - s).map(| p | p.max(0.0))
    }

    /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone)]
/// # Description
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
    /// # Description
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
    /// assert!((strip.payoff(&s) - Array1::from_vec(vec![4.0, 2.0, 0.0, 1.0, 2.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (s - self.k).map(| p | p.max(0.0)) + 2.0*(self.k - s).map(| p | p.max(0.0))
    }

    /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[derive(Clone)]
/// # Description
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
    /// # Description
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
    ///  assert!((strap.payoff(&s) - Array1::from_vec(vec![2.0, 1.0, 0.0, 2.0, 4.0])).sum().abs() < TEST_ACCURACY);
    /// ```
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        2.0*(s - self.k).map(| p | p.max(0.0)) + (self.k - s).map(| p | p.max(0.0))
    }

    /// # Description
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
    fn profit(&self, s: &Array1<f64>) -> Array1<f64> {
        self.payoff(s) - self.cost
    }

    fn update_cost(&mut self, new_cost: f64) -> () {
        self.cost = new_cost;
    }
}


#[cfg(feature = "plotly")]
/// # Description
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
/// ```rust
/// use digifi::financial_instruments::Strangle;
///
/// #[cfg(feature = "plotly")]
/// fn test_plot_payoff() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_payoff;
///     
///     let payoff_obj: Strangle = Strangle::new(6.0, 4.0, 1.0).unwrap();
///     let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
///     payoff_plot.show();
/// }
/// ```
pub fn plot_payoff(payoff_obj: &impl Payoff, start_price: f64, stop_price: f64, n_points: usize) -> Plot {
    let prices: Array1<f64> = Array1::linspace(start_price, stop_price, n_points);
    let payoff: Array1<f64> = payoff_obj.payoff(&prices);
    let mut plot: Plot = Plot::new();
    plot.add_trace(Scatter::new(prices.to_vec(), payoff.to_vec()));
    let x_axis: Axis = Axis::new().title("Price at Maturity");
    let y_axis: Axis = Axis::new().title("Payoff at Maturity");
    let layout: Layout = Layout::new().title("<b>Payoff Plot</b>").x_axis(x_axis).y_axis(y_axis);
    plot.set_layout(layout);
    plot
}


#[cfg(feature = "plotly")]
/// # Description
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
/// ```rust
/// use digifi::financial_instruments::Strangle;
///
/// #[cfg(feature = "plotly")]
/// fn test_plot_payoff() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_profit;
///     
///     let payoff_obj: Strangle = Strangle::new(6.0, 4.0, 1.0).unwrap();
///     let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
///     profit_plot.show();
/// }
/// ```
pub fn plot_profit(payoff_obj: &impl Payoff, start_price: f64, stop_price: f64, n_points: usize) -> Plot {
    let prices: Array1<f64> = Array1::linspace(start_price, stop_price, n_points);
    let profit: Array1<f64> = payoff_obj.profit(&prices);
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
        assert!((long_call.payoff(&s) - Array1::from_vec(vec![0.0, 0.0, 3.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_short_call() -> () {
        use crate::financial_instruments::ShortCall;
        let s: Array1<f64> = array![10.0, 9.0, 13.0];
        let short_call: ShortCall = ShortCall { k: 10.0, cost: 1.0 };
        assert!((short_call.payoff(&s) - Array1::from_vec(vec![0.0, 0.0, -3.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_long_put() -> () {
        use crate::financial_instruments::LongPut;
        let s: Array1<f64> = array![10.0, 9.0, 13.0];
        let long_put: LongPut = LongPut { k: 10.0, cost: 1.0 };
        assert!((long_put.payoff(&s) - Array1::from_vec(vec![0.0, 1.0, 0.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_short_put() -> () {
        use crate::financial_instruments::ShortPut;
        let s: Array1<f64> = array![10.0, 9.0, 13.0];
        let short_put: ShortPut = ShortPut { k: 10.0, cost: 1.0 };
        assert!((short_put.payoff(&s) - Array1::from_vec(vec![0.0, -1.0, 0.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_bull_collar() -> () {
        use crate::financial_instruments::BullCollar;
        let s: Array1<f64> = Array1::range(3.0, 7.0, 0.5);
        let bull_collar: BullCollar = BullCollar::new(4.0, 6.0, 0.0, 5.0).unwrap();
        println!("{}", bull_collar.profit(&s));
        assert!((bull_collar.payoff(&s) - Array1::from_vec(vec![-1.0, -1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_bear_collar() -> () {
        use crate::financial_instruments::BearCollar;
        let s: Array1<f64> = Array1::range(3.0, 7.0, 0.5);
        let bear_collar: BearCollar = BearCollar::new(4.0, 6.0, 0.0, 5.0).unwrap();
        assert!((bear_collar.payoff(&s) - Array1::from_vec(vec![1.0, 1.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_bull_spread() -> () {
        use crate::financial_instruments::BullSpread;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let bull_spread: BullSpread = BullSpread::new(4.0, 6.0, 1.0).unwrap();
        assert!((bull_spread.payoff(&s) - Array1::from_vec(vec![0.0, 0.0, 1.0, 2.0, 2.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_bear_spread() -> () {
        use crate::financial_instruments::BearSpread;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let bear_spread: BearSpread = BearSpread::new(6.0, 4.0, 1.0).unwrap();
        assert!((bear_spread.payoff(&s) - Array1::from_vec(vec![2.0, 2.0, 1.0, 0.0, 0.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_long_butterfly() -> () {
        use crate::financial_instruments::LongButterfly;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let long_butterfly: LongButterfly = LongButterfly::new(4.0, 6.0, 5.0, 1.0).unwrap();
        assert!((long_butterfly.payoff(&s) - Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_box_spread() -> () {
        use crate::financial_instruments::BoxSpread;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let box_spread: BoxSpread = BoxSpread::new(4.0, 6.0, 1.0).unwrap();
        assert!((box_spread.payoff(&s) - Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0, 2.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_straddle() -> () {
        use crate::financial_instruments::Straddle;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let straddle: Straddle = Straddle { k: 5.0, cost: 1.0 };
        assert!((straddle.payoff(&s) - Array1::from_vec(vec![2.0, 1.0, 0.0, 1.0, 2.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_strangle() -> () {
        use crate::financial_instruments::Strangle;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let strangle: Strangle = Strangle::new(6.0, 4.0, 1.0).unwrap();
        assert!((strangle.payoff(&s) - Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_strip() -> () {
        use crate::financial_instruments::Strip;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let strip: Strip = Strip { k: 5.0, cost: 1.0 };
        assert!((strip.payoff(&s) - Array1::from_vec(vec![4.0, 2.0, 0.0, 1.0, 2.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_strap() -> () {
        use crate::financial_instruments::Strap;
        let s: Array1<f64> = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let strap: Strap = Strap { k: 5.0, cost: 1.0 };
        assert!((strap.payoff(&s) - Array1::from_vec(vec![2.0, 1.0, 0.0, 2.0, 4.0])).sum().abs() < TEST_ACCURACY);
    }

    #[cfg(feature = "plotly")]
    #[test]
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
    fn unit_test_bull_collar_plot() -> () {
        use crate::financial_instruments::{BullCollar, plot_payoff, plot_profit};
        let payoff_obj: BullCollar = BullCollar::new(4.0, 6.0, 1.0, 5.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    fn unit_test_bear_collar_plot() -> () {
        use crate::financial_instruments::{BearCollar, plot_payoff, plot_profit};
        let payoff_obj: BearCollar = BearCollar::new(4.0, 6.0, 1.0, 5.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    fn unit_test_bull_spread_plot() -> () {
        use crate::financial_instruments::{BullSpread, plot_payoff, plot_profit};
        let payoff_obj: BullSpread = BullSpread::new(4.0, 6.0, 1.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    fn unit_test_bear_spread_plot() -> () {
        use crate::financial_instruments::{BearSpread, plot_payoff, plot_profit};
        let payoff_obj: BearSpread = BearSpread::new(6.0, 4.0, 1.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    fn unit_test_long_butterfly_plot() -> () {
        use crate::financial_instruments::{LongButterfly, plot_payoff, plot_profit};
        let payoff_obj: LongButterfly = LongButterfly::new(4.0, 6.0, 5.0, 1.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
    fn unit_test_box_spread_plot() -> () {
        use crate::financial_instruments::{BoxSpread, plot_payoff, plot_profit};
        let payoff_obj: BoxSpread = BoxSpread::new(4.0, 6.0, 1.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
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
    fn unit_test_strangle_plot() -> () {
        use crate::financial_instruments::{Strangle, plot_payoff, plot_profit};
        let payoff_obj: Strangle = Strangle::new(6.0, 4.0, 1.0).unwrap();
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }

    #[cfg(feature = "plotly")]
    #[test]
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
    fn unit_test_strap_plot() -> () {
        use crate::financial_instruments::{Strap, plot_payoff, plot_profit};
        let payoff_obj: Strap = Strap { k: 5.0, cost: 1.0 };
        let payoff_plot: Plot = plot_payoff(&payoff_obj, 0.0, 10.0, 21);
        payoff_plot.show();
        let profit_plot: Plot = plot_profit(&payoff_obj, 0.0, 10.0, 21);
        profit_plot.show();
    }
}