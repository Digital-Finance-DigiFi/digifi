use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::time_value_utils::{CompoundingType, Compounding, forward_rate};
use crate::financial_instruments::{FinancialInstrument, FinancialInstrumentId};
use crate::portfolio_applications::{AssetHistData, PortfolioInstrument};
use crate::stochastic_processes::StochasticProcess;


/// Forward rate agreement and its methods.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Forward_rate_agreement>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, Time, CompoundingType};
/// use digifi::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass, ForwardRateAgreement};
/// use digifi::portfolio_applications::AssetHistData;
///
/// let compounding_type: CompoundingType = CompoundingType::Continuous;
/// let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
///     instrument_type: FinancialInstrumentType::DerivativeInstrument, asset_class: AssetClass::ForeignExchangeInstrument,
///     identifier: String::from("32198407128904"),
/// };
/// let asset_historical_data: AssetHistData = AssetHistData::build(
///     Array1::from_vec(vec![0.4, 0.5]),Array1::from_vec(vec![0.0, 0.0]), Time::new(Array1::from_vec(vec![0.0, 1.0]))
/// ).unwrap();
/// let forward_rate: ForwardRateAgreement = ForwardRateAgreement::new(
///     0.04, 0.05, 1.0, 1000.0, 10.0, compounding_type, financial_instrument_id, asset_historical_data, None
/// );
///
/// assert!((forward_rate.present_value().unwrap() - 1.0*0.01*1000.0*(-0.05*1.0_f64).exp()).abs() < TEST_ACCURACY);
/// ```
pub struct ForwardRateAgreement {
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
    financial_instrument_id: FinancialInstrumentId,
    /// Time series asset data
    asset_historical_data: AssetHistData,
    /// Stochastic model to use for price paths generation
    stochastic_model: Option<Box<dyn StochasticProcess>>,
}

impl ForwardRateAgreement {
    /// Creates a new `ForwardRateAgreement` instance.
    /// 
    /// # Input
    /// - `agreed_fixed_rate`: Agreed fixed rate of the contract
    /// - `current_forward_rate`: Current market-derived forward rate of similar contracts
    /// - `time_to_maturity`: Time to maturity of the contract
    /// - `principal`: Principal of the forward rate agreement
    /// - `initial_price`: Initial price of the forward rate contract
    /// - `compounding_type`: Compounding type used to discount cashflows
    /// - `financial_instrument_id`: Parameters for defining regulatory categorization of an instrument
    /// - `asset_historical_data`: Time series asset data
    /// - `stochastic_model`: Stochastic model to use for price paths generation
    pub fn new(
        agreed_fixed_rate: f64, current_forward_rate: f64, time_to_maturity: f64, principal: f64, initial_price: f64,
        compounding_type: CompoundingType, financial_instrument_id: FinancialInstrumentId, asset_historical_data: AssetHistData,
        stochastic_model: Option<Box<dyn StochasticProcess>>
    ) -> Self {
        Self {
            agreed_fixed_rate, current_forward_rate, time_to_maturity, principal, initial_price, compounding_type, financial_instrument_id,
            asset_historical_data, stochastic_model: stochastic_model,
        }
    }

    /// Update forward rate of the contract.
    /// 
    /// Helper method to update current_forward_rate during calculations.
    /// 
    /// # Input
    /// - new_forward_rate: Updated market forward rate
    pub fn update_forward_rate(&mut self, new_forward_rate: f64) -> () {
        self.current_forward_rate = new_forward_rate;
    }

    /// Update time to maturity.
    /// 
    /// Helper method to update time_to_maturity during calculations.
    /// 
    /// # Input
    /// - `new_time_to_maturity`: Updated time too maturity
    pub fn latest_time_maturity(&mut self, new_time_to_maturity: f64) -> () {
        self.time_to_maturity = new_time_to_maturity;
    }

    /// Forward interest rate for the period between time_1 and time_2.
    /// 
    /// # Input
    /// - `time_1`: Time step 1 (i.e., `current_forward_rate` in `ForwardRateAgreement` is defined for this time step)
    /// - `future_compounding_term`: Compounding term that defines the forward rate for time step 2
    /// - `time_2`: Time step 2
    /// 
    /// # Output
    /// - Forward rate from zero rate at time step 1 to zero rate at time step 2
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Forward_rate>
    /// - Original Source: N/A
    pub fn future_rate(&self, time_1: f64, future_compounding_term: &Compounding, time_2: f64) -> Result<f64, DigiFiError> {
        let current_compounding_term: Compounding = Compounding::new(self.current_forward_rate, &self.compounding_type);
        forward_rate(&current_compounding_term, time_1, future_compounding_term, time_2)
    }

    /// Adjustment of the forward rate based on futures rate.
    /// 
    /// Forward Rate = Futures Rate - Convexity Adjustment.
    /// 
    /// # Input
    /// - `futures_rate`: Current futures contract rate
    /// - `convexity_adjustment`: Convexity adjustment constant
    /// - `in_place`: Overwrite current_forward_rate with the obtained forward rate
    /// 
    /// # Output
    /// - Forward rate of the contract
    /// 
    /// # LaTeX Formula
    /// - \\textit{Forward Rate} = \\textit{Futures Rate} - \\textit{C}
    pub fn rate_adjustment(&mut self, futures_rate: f64, convexity_adjustment: f64, in_place: bool) -> Result<f64, DigiFiError> {
        if convexity_adjustment <= 0.0 {
            return Err(DigiFiError::ParameterConstraint { title: Self::error_title(), constraint: "The argument `convexity_adjustment` must be positive.".to_owned(), });
        }
        let forward_rate: f64 = futures_rate - convexity_adjustment;
        if in_place {
            self.current_forward_rate = forward_rate;
        }
        Ok(forward_rate)
    }
}

impl ErrorTitle for ForwardRateAgreement {
    fn error_title() -> String {
        String::from("Forward Rate Agreement")
    }
}

impl FinancialInstrument for ForwardRateAgreement {
    /// Present values of the forward rate agreement.
    /// 
    /// # Output
    /// - Present value of the forward rate agreement
    /// 
    /// # LaTeX Formula
    /// - PV = \\tau(R_{F} - R_{K})L
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Forward_rate_agreement#Valuation_and_pricing>
    /// - Original Source: N/A
    fn present_value(&self) -> Result<f64, DigiFiError> {
        let discount_term: Compounding = Compounding::new(self.current_forward_rate, &self.compounding_type);
        Ok((self.time_to_maturity * (self.current_forward_rate - self.agreed_fixed_rate) * self.principal) * discount_term.compounding_term(self.time_to_maturity))
    }

    /// Net present value of the forward rate agreement.
    /// 
    /// # Output
    /// - Present value of the forward rate agreement minus the initial price it took to purchase the contract
    fn net_present_value(&self) -> Result<f64, DigiFiError> {
        Ok(-self.initial_price + self.present_value()?)
    }

    /// Future value of the forward rate agreement.
    /// 
    /// # Output
    /// - Future value of the forward rate agreement at it maturity (Computed from the present value of the forward rate agreement)
    fn future_value(&self) -> Result<f64, DigiFiError> {
        let discount_term: Compounding = Compounding::new(self.current_forward_rate, &self.compounding_type);
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

impl PortfolioInstrument for ForwardRateAgreement {

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

    #[test]
    fn unit_test_forward_rate() -> () {
        use crate::utilities::{Time, time_value_utils::CompoundingType};
        use crate::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass};
        use crate::financial_instruments::rates_and_swaps::ForwardRateAgreement;
        use crate::portfolio_applications::AssetHistData;
        let compounding_type: CompoundingType = CompoundingType::Continuous;
        let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
            instrument_type: FinancialInstrumentType::DerivativeInstrument, asset_class: AssetClass::ForeignExchangeInstrument,
            identifier: String::from("32198407128904"),
        };
        let asset_historical_data: AssetHistData = AssetHistData::build(
            Array1::from_vec(vec![0.4, 0.5]), Array1::from_vec(vec![0.0, 0.0]), Time::new(Array1::from_vec(vec![0.0, 1.0]))
        ).unwrap();
        let forward_rate: ForwardRateAgreement = ForwardRateAgreement::new(
            0.04, 0.05, 1.0, 1000.0, 10.0, compounding_type, financial_instrument_id, asset_historical_data, None
        );
        assert!((forward_rate.present_value().unwrap() - 1.0*0.01*1000.0*(-0.05*1.0_f64).exp()).abs() < TEST_ACCURACY);
    }
}