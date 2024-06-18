use ndarray::Array1;


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
    fn present_value(&self) -> f64;

    /// # Description
    /// Computes the net present value of the financial instrument.
    fn net_present_value(&self) -> f64;

    /// # Description
    /// Computes the future value of the financial instrument.
    fn future_value(&self) -> f64;

}

pub trait Payoff {
    /// # Description
    /// Payoff function.
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64>;

    /// # Description
    /// Validation of payoff object to satisfy the computational requirements.
    /// 
    /// # Input
    /// - length_value: Number of test data points to validate payoff method on
    fn validate_payoff(&self, val_length: usize) -> () {
        let s: Array1<f64> = Array1::from_vec(vec![1.0; val_length]);
        let result: Array1<f64> = self.payoff(&s);
        if result.len() != val_length {
            panic!("The payoff does not produce an array of length {}.", val_length);
        }
    }
}


/// # Description
/// Long call payoff.
pub struct LongCall {
    /// Strike price
    pub k: f64,
}

impl Payoff for LongCall {
    //// # Description
    /// Long call option payoff.
    /// 
    /// # Input
    /// - s: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the long call option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Long Call Payoff} = max(S_{t}-K, 0)
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (s - self.k).map(| p | p.max(0.0))
    }
}


/// # Description
/// Short call payoff.
pub struct ShortCall {
    /// Strike price
    pub k: f64,
}

impl Payoff for ShortCall {
    /// # Description
    /// Short call option payoff.
    /// 
    /// # Input
    /// - s: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the short call option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Short Call Payoff} = min(K-S_{t}, 0)
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (self.k - s).map(| p | p.min(0.0))
    }
}


/// # Description
/// Long put payoff.
pub struct LongPut {
    /// Strike price
    pub k: f64,
}

impl Payoff for LongPut {
    /// # Description
    /// Long put option payoff.
    /// 
    /// # Input
    /// - s: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the long put option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Long Put Payoff} = max(K-S_{t}, 0)
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (self.k - s).map(| p | p.max(0.0))
    }
}


/// # Description
/// Short put payoff.
pub struct ShortPut {
    /// Strike price
    pub k: f64,
}

impl Payoff for ShortPut {
    /// # Description
    /// Short put option payoff.
    /// 
    /// # Input
    /// - s: Underlying asset price
    /// 
    /// # Output
    /// - Payoff of the short put option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Short Put Payoff} = min(S_{t}-K, 0)
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        (s - self.k).map(| p | p.min(0.0))
    }
}


/// # Description
/// Bull collar payoff.
pub struct BullCollar {
    /// Long put option strike price
    k_p: f64,
    /// Short call option strike price
    k_c: f64,
}

impl Payoff for BullCollar {
    /// # Description
    /// Bull collar option payoff.\n
    /// Bull Collar = Asset + Long Put + Short Call.
    /// 
    /// # Input
    /// - s_t: Underlying asset price
    /// - k_p: Long put option strike price
    /// - k_c: Short call option strike price
    /// 
    /// # Output
    /// - Payoff of the bull collar option
    /// 
    /// # LaTeX Formula
    /// - \\textit{Bull Collar Payoff} = S_{t} - S_{t-1} + max(K_{p}-S_{t}, 0) + min(K_{c}-S_{t}, 0)
    fn payoff(&self, s: &Array1<f64>) -> Array1<f64> {
        // TODO: Add asset payoff
        s + (self.k_p - s).map(| p | p.max(0.0)) + (self.k_c - s).map(| p | p.min(0.0))
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};
    use crate::utilities::TEST_ACCURACY;
    use crate::financial_instruments::Payoff;

    #[test]
    fn unit_test_long_call() -> () {
        use crate::financial_instruments::LongCall;
        let s: Array1<f64> = array![10.0, 9.0, 13.0];
        let long_call: LongCall = LongCall { k: 10.0 };
        assert!((long_call.payoff(&s) - Array1::from_vec(vec![0.0, 0.0, 3.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_short_call() -> () {
        use crate::financial_instruments::ShortCall;
        let s: Array1<f64> = array![10.0, 9.0, 13.0];
        let short_call: ShortCall = ShortCall { k: 10.0 };
        assert!((short_call.payoff(&s) - Array1::from_vec(vec![0.0, 0.0, -3.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_long_put() -> () {
        use crate::financial_instruments::LongPut;
        let s: Array1<f64> = array![10.0, 9.0, 13.0];
        let long_put: LongPut = LongPut { k: 10.0 };
        assert!((long_put.payoff(&s) - Array1::from_vec(vec![0.0, 1.0, 0.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_short_put() -> () {
        use crate::financial_instruments::ShortPut;
        let s: Array1<f64> = array![10.0, 9.0, 13.0];
        let short_put: ShortPut = ShortPut { k: 10.0 };
        assert!((short_put.payoff(&s) - Array1::from_vec(vec![0.0, -1.0, 0.0])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_bull_collar() -> () {
        use crate::financial_instruments::BullCollar;
        let s: Array1<f64> = array![10.0, 9.0, 13.0];
        let bull_collar: BullCollar = BullCollar { k_p: 9.0, k_c: 11.0 };
        assert!((bull_collar.payoff(&s) - Array1::from_vec(vec![10.0, 9.0, 11.0])).sum().abs() < TEST_ACCURACY);
    }
}