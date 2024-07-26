use std::io::Error;
use ndarray::Array1;
use crate::utilities::{input_error, time_value_utils::{CompoundingType, Compounding}};
use crate::financial_instruments::{FinancialInstrument, FinancialInstrumentId};
use crate::portfolio_applications::AssetHistData;
use crate::stochastic_processes::StochasticProcess;


/// # Description
/// Forward interest rate for the period between time_1 and time_2.
/// 
/// # Input
/// - zero_rate_1: Zero rate at time step 1
/// - time_1: Time step 1
/// - zero_rate_2: Zero rate at time step 2
/// - time_2: Time step 2
/// 
/// # Output
/// - Forward rate from zero rate at time step 1 to zero rate at time step 2
/// 
/// # LaTeX Formula
/// - R_{f} = \\frac{R_{2}T_{2} - R_{1}T_{1}}{T_{2} - T_{1}}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Forward_rate
/// - Original Source: N/A
pub fn forward_interest_rate(zero_rate_1: f64, time_1: f64, zero_rate_2: f64, time_2: f64) -> f64 {
    (zero_rate_2*time_2 - zero_rate_1*time_1) / (time_2 - time_1)
}


/// # Description
/// Zero rate defined through the previous zero rate and current forward rate.
/// 
/// # Input
/// - zero_rate_1 (float): Zero rate at time step 1
/// - time_1 (float): Time step 1
/// - time_2 (float): Time step 2
/// - forward_rate (float): Current forward rate
/// 
/// # Output
/// - Zero rate at time step 2
/// 
/// # LaTeX Formula
/// - R_{2} = \\frac{R_{F}(T_{2}-T_{1}) + R_{1}T_{1}}{T_{2}}
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Forward_rate
/// - Original Source: N/A
pub fn future_zero_rate(zero_rate_1: f64, time_1: f64, time_2: f64, forward_rate: f64) -> f64 {
    (forward_rate*(time_2 - time_1) + zero_rate_1*time_1) / time_2
}


/// # Description
/// Forward rate agreement and its methods.
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Forward_rate_agreement
/// - Original Source: N/A
pub struct ForwardRate {
    /// Agreed fixed rate of the contract
    agreed_fixed_rate: f64,
    /// Current market-derived forward rate of similar contracts
    current_forward_rate: f64,
    /// Time to maturity of the contract
    time_to_maturity: f64,
    /// Principal of the forward rate agreement
    principal: f64,
    /// Initial price of the forward rate contract
    initial_price: f64,
    /// Compounding type used to discount cashflows
    compounding_type: CompoundingType,
    /// Parameters for defining regulatory categorization of an instrument
    _financial_instrument_id: FinancialInstrumentId,
    /// Time series asset data
    _asset_historical_data: AssetHistData,
    /// Stochastic model to use for price paths generation
    stochastic_model: Box<dyn StochasticProcess>,
}

impl ForwardRate {
    /// # Description
    /// Creates a new ForwardRate instance.
    /// 
    /// # Input
    /// - agreed_fixed_rate: Agreed fixed rate of the contract
    /// - current_forward_rate: Current market-derived forward rate of similar contracts
    /// - time_to_maturity: Time to maturity of the contract
    /// - principal: Principal of the forward rate agreement
    /// - initial_price: Initial price of the forward rate contract
    /// - compounding_type: Compounding type used to discount cashflows
    /// - financial_instrument_id: arameters for defining regulatory categorization of an instrument
    /// - asset_historical_data: Time series asset data
    /// - stochastic_model: Stochastic model to use for price paths generation
    pub fn new(agreed_fixed_rate: f64, current_forward_rate: f64, time_to_maturity: f64, principal: f64, initial_price: f64,
               compounding_type: CompoundingType, _financial_instrument_id: FinancialInstrumentId, _asset_historical_data: AssetHistData,
               stochastic_model: Box<dyn StochasticProcess>) -> Self {
        // TODO: Add default stochastic model for the case when the user doesn't provide one
        ForwardRate { agreed_fixed_rate, current_forward_rate, time_to_maturity, principal, initial_price, compounding_type,
                      _financial_instrument_id, _asset_historical_data, stochastic_model }
    }

    /// # Description
    /// Update forward rate of the contract.
    /// 
    /// Helper method to update current_forward_rate during calculations.
    /// 
    /// # Input
    /// - new_forward_rate: Updated market forward rate
    pub fn update_forward_rate(&mut self, new_forward_rate: f64) -> () {
        self.current_forward_rate = new_forward_rate;
    }

    /// # Description
    /// Update time to maturity.
    /// 
    /// Helper method to update time_to_maturity during calculations.
    /// 
    /// # Input
    /// - new_time_to_maturity: Updated time too maturity
    pub fn latest_time_maturity(&mut self, new_time_to_maturity: f64) -> () {
        self.time_to_maturity = new_time_to_maturity;
    }

    /// # Description
    /// Forward interest rate for the period between time_1 and time_2.
    /// 
    /// # Input
    /// - zero_rate_1 (float): Zero rate at time step 1
    /// - time_1 (float): Time step 1
    /// - zero_rate_2 (float): Zero rate at time step 2
    /// - time_2 (float): Time step 2
    /// - in_place (bool): Overwrite current_forward_rate with the obtained forward rate
    /// 
    /// # Output
    /// - Forward rate from zero rate at time step 1 to zero rate at time step 2
    /// 
    /// # LaTeX Formula
    /// - R_{f} = \\frac{R_{2}T_{2} - R_{1}T_{1}}{T_{2} - T_{1}}
    /// 
    /// # Links
    /// - Wikipedia: https://en.wikipedia.org/wiki/Forward_rate
    /// - Original Source: N/A
    pub fn future_rate_from_zero_rates(&mut self, zero_rate_1: f64, time_1: f64, zero_rate_2: f64, time_2: f64, in_place: bool) -> f64 {
        let forward_rate: f64 = forward_interest_rate(zero_rate_1, time_1, zero_rate_2, time_2);
        if in_place {
            self.current_forward_rate = forward_rate
        }
        forward_rate
    }

    /// # Description
    /// Zero rate defined through the previous zero rate and current forward rate.
    /// 
    /// # Input
    /// - zero_rate_1: Zero rate at time step 1
    /// - time_1: Time step 1
    /// - time_2: Time step 2
    /// 
    /// # Output
    /// - Zero rate at time step 2
    /// 
    /// # LaTeX Formula
    /// - R_{2} = \\frac{R_{F}(T_{2}-T_{1}) + R_{1}T_{1}}{T_{2}}
    /// 
    /// # Links
    /// - Wikipedia: https://en.wikipedia.org/wiki/Forward_rate
    /// - Original Source: N/A
    pub fn zero_rate_from_forward_rate(&self, zero_rate_1: f64, time_1: f64, time_2: f64) -> f64 {
        future_zero_rate(zero_rate_1, time_1, time_2, self.current_forward_rate)
    }

    /// # Description
    /// Adjustment of the forward rate based on futures rate.
    /// 
    /// Forward Rate = Futures Rate - Convexity Adjustment.
    /// 
    /// # Input
    /// - futures_rate (float): Current futures contract rate
    /// - convexity_sdjustment (float): Convexity adjustment constant
    /// - in_place (bool): Overwrite current_forward_rate with the obtained forward rate
    /// 
    /// # Output
    /// - Forward rate of the contract
    /// 
    /// # LaTeX Formula
    /// - \\textit{Forward Rate} = \\textit{Futures Rate} - \\textit{C}
    pub fn rate_adjustment(&mut self, futures_rate: f64, convexity_adjustment: f64, in_place: bool) -> Result<f64, Error> {
        if convexity_adjustment <= 0.0 {
            return Err(input_error("The argument convexity adjustment must be positive."));
        }
        let forward_rate: f64 = futures_rate - convexity_adjustment;
        if in_place {
            self.current_forward_rate = forward_rate;
        }
        Ok(forward_rate)
    }
}

impl FinancialInstrument for ForwardRate {
    /// # Description
    /// Present values of the forward rate agreement.
    /// 
    /// # Output
    /// - Present value of the forward rate agreement
    /// 
    /// # LaTeX Formula
    /// - PV = \\tau(R_{F} - R_{K})L
    /// 
    /// # Links
    /// - Wikipedia: https://en.wikipedia.org/wiki/Forward_rate_agreement#Valuation_and_pricing
    /// - Original Source: N/A
    fn present_value(&self) -> f64 {
        let discount_term: Compounding = Compounding::new(self.current_forward_rate, self.compounding_type.clone());
        (self.time_to_maturity * (self.current_forward_rate - self.agreed_fixed_rate) * self.principal) * discount_term.compounding_term(self.time_to_maturity)
    }

    /// # Description
    /// Net present value of the forward rate agreement.
    /// 
    /// # Output
    /// - Present value of the forward rate agreement minus the initial price it took to purchase the contract
    fn net_present_value(&self) -> f64 {
        -self.initial_price + self.present_value()
    }

    /// # Description
    /// Future value of the forward rate agreement.
    /// 
    /// # Output
    /// - Future value of the forward rate agreement at it maturity (Computed from the present value of the forward rate agreement)
    fn future_value(&self) -> f64 {
        let discount_term: Compounding = Compounding::new(self.current_forward_rate, self.compounding_type.clone());
        self.present_value() / discount_term.compounding_term(self.time_to_maturity)
    }

    /// # Description
    /// Simulated stochastic paths of the forward rate agreement.
    /// 
    /// # Output
    /// - Simulated spot prices of the forward rate agreement
    fn stochastic_simulation(&self) -> Result<Vec<Array1<f64>>, Error> {
        self.stochastic_model.get_paths()
    }

    /// # Description
    /// Generates an array of prices and predictable income, and updates the asset_historical_data.
    fn generate_historic_data(&self) -> () {
        // TODO: Implement data generation
    }
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::{financial_instruments::FinancialInstrument, utilities::TEST_ACCURACY};

//     #[test]
//     fn unit_test_forward_rate() -> () {
//         use crate::utilities::time_value_utils::CompoundingType;
//         use crate::financial_instruments::{FinancialInstrumentId, FinancialInstrumentType, AssetClass};
//         use crate::financial_instruments::rates_and_swaps::ForwardRate;
//         use crate::portfolio_applications::AssetHistData;
//         let compounding_type: CompoundingType = CompoundingType::Continuous;
//         let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {instrument_type: FinancialInstrumentType::DerivativeInstrument,
//                                                                                     asset_class: AssetClass::ForeignExchangeInstrument,
//                                                                                     identifier: String::from("32198407128904") };
//         let asset_historical_data: AssetHistData = AssetHistData::new(Array1::from_vec(vec![10.0]),
//                                                                       Array1::from_vec(vec![0.0]),
//                                                                       Array1::from_vec(vec![0.0]));
//         let forward_rate: ForwardRate = ForwardRate::new(0.04, 0.05, 1.0, 1000.0, 10.0,
//                                                          compounding_type, financial_instrument_id, asset_historical_data);
//         assert!((forward_rate.present_value() - 1.0*0.01*1000.0*(-0.05*1.0_f64).exp()).abs() < TEST_ACCURACY);
//     }
}