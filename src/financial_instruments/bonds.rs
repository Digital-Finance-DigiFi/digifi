use std::convert::From;
use ndarray::Array1;
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::{compare_len, Time, ParameterType, time_value_utils::{internal_rate_of_return, CompoundingType, Compounding, Cashflow}};
use crate::financial_instruments::{FinancialInstrument, FinancialInstrumentId};
use crate::portfolio_applications::{AssetHistData, PortfolioInstrument};
use crate::stochastic_processes::StochasticProcess;


/// Spot rate computation for the given list of bonds.
///
/// # Input
/// - `principals`: Array of bond principals
/// - `maturities`: An array of bond maturities
/// - `coupons`: An array of bond coupons (Coupns are assumed to remain constant with respect to future payoffs)
/// - `prices`: An array of current market prices of bonds
/// - `coupon_dt`: An array of differences between times of coupon payments (e.g., for semi-annual coupon `coupon_dt`=0.5)
///
/// # Output
/// - Spot rates of the bonds provided
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Bootstrapping_(finance)>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::financial_instruments::bonds::bootstrap;
///
/// let principals: Array1<f64> = Array1::from_vec(vec![100.0, 100.0, 100.0, 100.0, 100.0]);
/// let maturities: Array1<f64> = Array1::from_vec(vec![0.25, 0.5, 1.0, 1.5, 2.0]);
/// let coupons: Array1<f64> = Array1::from_vec(vec![0.0, 0.0, 0.0, 4.0, 5.0]);
/// let prices: Array1<f64> = Array1::from_vec(vec![99.6, 99.0, 97.8, 102.5, 105.0]);
/// let coupon_dt: Array1<f64> = Array1::from_vec(vec![0.25, 0.5, 1.0, 0.5, 0.5]);
///
/// let spot_rates: Array1<f64> = bootstrap(principals, maturities, coupons, prices, coupon_dt).unwrap();
/// let theor: Array1<f64> = Array1::from_vec(vec![0.01603, 0.0201, 0.02225, 0.02284, 0.02416]);
///
/// assert!((spot_rates - theor).map(|v| v.abs() ).sum() < 10_000.0 * TEST_ACCURACY);
/// ```
pub fn bootstrap(principals: Array1<f64>, maturities: Array1<f64>, coupons: Array1<f64>, prices: Array1<f64>, coupon_dt: Array1<f64>) -> Result<Array1<f64>, DigiFiError> {
    compare_len(&principals.iter(), &maturities.iter(), "principals", "maturities")?;
    compare_len(&principals.iter(), &coupons.iter(), "principals", "coupons")?;
    compare_len(&principals.iter(), &prices.iter(), "principals", "prices")?;
    compare_len(&principals.iter(), &coupon_dt.iter(), "principals", "coupon_dt")?;
    for dt in &coupon_dt {
        if (dt < &0.0) || (&1.0 < dt) {
            return Err(DigiFiError::ParameterConstraint {
                title: "Bootstrap".to_owned(),
                constraint: "The argument `coupon_dt` must contain values in the range `[0, 1]`.".to_owned(),
            });
        }
    }
    let mut spot_rates: Vec<f64> = Vec::with_capacity(principals.len());
    for i in 0..principals.len() {
        let time: Time = Time::new_from_range(coupon_dt[i], maturities[i] - coupon_dt[i], coupon_dt[i]);
        let discount_term: f64 = time.time_array().iter().fold(0.0, |discount_term, time_step| {
            // Find the correct choice of spot rate based on time step
            let mut spot_rate_choice: usize = 0;
            for j in 0..maturities.len() {
                if maturities[j] == *time_step {
                    spot_rate_choice = j;
                    break;
                } else if *time_step < maturities[j] {
                    spot_rate_choice = j-1;
                    break;
                }
            }
            discount_term + (-time_step * spot_rates[spot_rate_choice]).exp()
        } );
        let spot_rate: f64 = -((prices[i] - coupons[i] * coupon_dt[i] * discount_term) / (principals[i] + coupons[i] * coupon_dt[i])).ln() / maturities[i];
        spot_rates.push(spot_rate);
    }
    Ok(Array1::from_vec(spot_rates))
}


#[derive(Clone, Copy, Debug)]
/// Method for computing yield-to-maturity of a bond.
pub enum YtMMethod {
    /// Uses internal rate of return numerical solver
    Numerical,
    /// Uses the YtM approximation formula
    Approximation,
}


#[derive(Clone, Copy, Debug)]
/// Type of bond that specifies the structure of the cashflow payoffs.
pub enum BondType {
    /// Bond with a regularly paid coupon
    AnnuityBond { principal: f64, coupon_rate: f64, maturity: f64, first_coupon_time: Option<f64>, },
    /// Bond with a regularly paid coupon hat grows at a predetermined rate
    GrowingAnnuityBond { principal: f64, coupon_rate: f64, maturity: f64, coupon_growth_rate: f64, first_coupon_time: Option<f64>, },
    /// Zero-coupon bond
    ZeroCouponBond { principal: f64, maturity: f64, },
}


struct BondTypeInfo {
    principal: f64,
    coupon_rate: f64,
    maturity: f64,
    coupon_growth_rate: f64,
    first_coupon_time: f64,
}

impl From<&BondType> for BondTypeInfo {
    fn from(value: &BondType) -> Self {
        match value {
            BondType::AnnuityBond { principal, coupon_rate, maturity, first_coupon_time } => {
                BondTypeInfo {
                    principal: *principal, coupon_rate: *coupon_rate, maturity: *maturity,
                    coupon_growth_rate: 0.0, first_coupon_time: match first_coupon_time { Some(v) => *v, None => 1.0, },
                }
            },
            BondType::GrowingAnnuityBond { principal, coupon_rate, maturity, coupon_growth_rate, first_coupon_time } => {
                BondTypeInfo {
                    principal: *principal, coupon_rate: *coupon_rate, maturity: *maturity,
                    coupon_growth_rate: *coupon_growth_rate, first_coupon_time: match first_coupon_time { Some(v) => *v, None => 1.0, },
                }
            },
            BondType::ZeroCouponBond { principal, maturity } => {
                BondTypeInfo { principal: *principal, coupon_rate: 0.0, maturity: *maturity, coupon_growth_rate: 0.0, first_coupon_time: 0.0, }
            },
        }
    }
}


/// Bond financial instrument and its methods.
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Bond_(finance)>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, ParameterType, Time, CompoundingType, present_value};
/// use digifi::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass, BondType, Bond};
/// use digifi::portfolio_applications::AssetHistData;
///
/// // Bond definition
/// let compounding_type: CompoundingType = CompoundingType::Continuous;
/// let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
///     instrument_type: FinancialInstrumentType::CashInstrument, asset_class: AssetClass::DebtBasedInstrument, identifier: String::from("32198407128904"),
/// };
/// let asset_historical_data: AssetHistData = AssetHistData::build(
///     Array1::from_vec(vec![0.4, 0.5]), Array1::from_vec(vec![0.0, 0.0]), Time::new(Array1::from_vec(vec![0.0, 1.0]))
/// ).unwrap();
/// let bond_type: BondType = BondType::AnnuityBond { principal: 100.0, coupon_rate: 0.05, maturity: 5.0, first_coupon_time: Some(1.0) };
/// let discount_rate: ParameterType = ParameterType::Value { value: 0.02 };
/// let bond: Bond = Bond::build(bond_type, discount_rate, 101.0, compounding_type, Some(0.0), financial_instrument_id, asset_historical_data, None).unwrap();
///
/// // Theoretical value
/// let cashflow: Vec<f64> = vec![5.0, 5.0, 5.0, 5.0, 105.0];
/// let time: Time = Time::new_from_range(1.0, 5.0, 1.0);
/// let pv: f64 = present_value(cashflow.iter(), &time, ParameterType::Value { value: 0.02 }, &CompoundingType::Continuous).unwrap();
///
/// assert!((bond.present_value().unwrap() - pv).abs() < TEST_ACCURACY);
/// ```
pub struct Bond {
    /// Type of bond
    bond_type: BondType,
    /// Principal (i.e., par or face value) of the bond
    principal: f64,
    /// Discount rate(s) for discounting future coupons
    discount_rate: Array1<f64>,
    /// Maturity of the bond
    maturity: f64,
    /// Price at which the bond was initially purchased
    initial_price: f64,
    /// Compounding type used to discount cashflows
    compounding_type: CompoundingType,
    /// Compounding frequency of bond's payoff (e.g., for a bond with 3% semi-annual coupon rate, the compounding frequency is 2)
    compounding_frequency: f64,
    /// Parameters for defining regulatory categorization of an instrument
    financial_instrument_id: FinancialInstrumentId,
    /// Time series asset data
    asset_historical_data: AssetHistData,
    /// Stochastic model to use for price paths generation
    stochastic_model: Option<Box<dyn StochasticProcess>>,
    /// Nominal coupon value
    coupon: f64,
    /// Time series of cashflow payoffs (Excluding principal)
    cashflow: Cashflow,
}

impl Bond {

    /// Creates a new `Bond` instance.
    /// 
    /// # Input
    /// - `bond_type`: Type of bond
    /// - `discount_rate`: Discount rate(s) for discounting future coupons
    /// - `initial_price`: Price at which the bond was initially purchased
    /// - `compounding_type`: Compounding type used to discount cashflows
    /// - `inflation_rate`: Inflation rate with which the cashflows will be discounted (Applies multiplicatively on top of `discount_rates`)
    /// - `financial_instrument_id`: Parameters for defining regulatory categorization of an instrument
    /// - `asset_historical_data`: Time series asset data
    /// - `stochastic_model`: Stochastic model to use for price paths generation
    pub fn build(bond_type: BondType, discount_rate: ParameterType, initial_price: f64, compounding_type: CompoundingType, inflation_rate: Option<f64>,
        financial_instrument_id: FinancialInstrumentId, asset_historical_data: AssetHistData, stochastic_model: Option<Box<dyn StochasticProcess>>
    ) -> Result<Self, DigiFiError> {
        let bond_info: BondTypeInfo = (&bond_type).into();
        let compounding_frequency: f64 = compounding_type.frequency() as f64;
        let coupon: f64 = bond_info.principal * bond_info.coupon_rate / compounding_frequency;
        let time: Time = Time::new_from_range(bond_info.first_coupon_time, bond_info.maturity, 1.0 / compounding_frequency);
        let inflation_rate: f64 = inflation_rate.unwrap_or(0.0);
        let cashflow: Cashflow = Cashflow::build(ParameterType::Value { value: coupon }, time, bond_info.coupon_growth_rate, inflation_rate)?;
        let discount_rate: Array1<f64> = discount_rate.into_array(cashflow.len())?;
        Ok(Bond {
            bond_type, principal: bond_info.principal, discount_rate, maturity: bond_info.maturity, initial_price: initial_price, compounding_type, compounding_frequency,
            financial_instrument_id, asset_historical_data: asset_historical_data, stochastic_model: stochastic_model, coupon, cashflow,
        })
    }

    /// Estimated total rate of return of the bond from current time to its maturity.
    ///
    /// # Input
    /// - `ytm_method`: Method for evaluating yield-to-maturity (i.e., Numerical - numerical solution of the IRR, Approximation - approximation formula)
    ///
    /// # Output
    /// - Yield-to-Maturity (YtM)
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Yield_to_maturity>
    /// - Original Source: N/A
    pub fn yield_to_maturity(&self, ytm_method: YtMMethod) -> Result<f64, DigiFiError> {
        match ytm_method {
            YtMMethod::Approximation => {
                Ok((self.coupon + (self.principal - self.initial_price) / self.maturity) / (0.5 * (self.principal * self.initial_price)))
            },
            YtMMethod::Numerical => {
                let mut cashflow: Array1<f64> = self.cashflow.cashflow().clone();
                let last_index: usize = cashflow.len() - 1;
                cashflow[last_index] = cashflow[last_index] + self.principal;
                internal_rate_of_return(self.initial_price, cashflow.iter(), self.cashflow.time(), &self.compounding_type)
            },
        }
    }

    /// Estimated total rate of return of a callable bond until it is called.
    ///
    /// # Input
    /// - `call_price`: Strike price of the callable bond
    ///
    /// # Output
    /// - Yield-to-Call
    pub fn yield_to_call(&self, call_price: f64) -> Result<f64, DigiFiError> {
        let mut cashflow: Array1<f64> = self.cashflow.cashflow().clone();
        let last_index: usize = cashflow.len() - 1;
        cashflow[last_index] = cashflow[last_index] + call_price;
        internal_rate_of_return(0.0, cashflow.iter(), self.cashflow.time(), &self.compounding_type)
    }

    /// Spot rate of a bond computed based on cashflows of the bond discounted with the provided spot rates.
    ///
    /// # Input
    /// - `spot_rates`: Spot rates for bonds with smaller maturities (Must be the same length as the number of coupons of the bond)
    ///
    /// # Output
    /// - Spot rate of the bond
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Spot_contract#Bonds_and_swaps>
    /// - Original Source: N/A
    pub fn spot_rate(&self, spot_rates: Option<Array1<f64>>) -> Result<f64, DigiFiError> {
        match self.bond_type {
            BondType::ZeroCouponBond { principal, maturity } => {
                Ok((self.initial_price / principal).ln() / maturity)
            },
            _ => {
                let spot_rates: Array1<f64> = spot_rates.ok_or(DigiFiError::ValidationError {
                    title: Self::error_title(),
                    details: "The argument `spot_rates` must be an array if the bond type is not `ZeroCouponBond`.".to_owned(),
                })?;
                let last_cashflow_i: usize = self.cashflow.len() - 1;
                if spot_rates.len() != last_cashflow_i {
                    return Err(DigiFiError::UnmatchingLength { array_1: "spot_rate".to_owned(), array_2: "cashflow (without the last element)".to_owned(), });
                }
                let discounted_coupons: f64 = (0..last_cashflow_i).into_iter().fold(0.0, |discounted_coupons, i| {
                    let discount_term: Compounding = Compounding::new(spot_rates[i], &self.compounding_type);
                    discounted_coupons + self.cashflow.cashflow()[i] * discount_term.compounding_term(self.cashflow.time_array()[i])
                } );
                match self.compounding_type {
                    CompoundingType::Continuous => {
                        Ok(-((self.initial_price - discounted_coupons) / (self.principal + self.cashflow.cashflow()[last_cashflow_i])).ln() / self.maturity)
                    },
                    CompoundingType::Periodic { frequency } => {
                        let frequency: f64 = frequency as f64;
                        let power: f64 = -1.0 / (self.maturity * frequency);
                        Ok(frequency * ((self.initial_price - discounted_coupons) / (self.principal + self.cashflow.cashflow()[last_cashflow_i])).powf(power) - frequency)
                    },
                }
            }
        }
    }

    /// Estimated rate of return of the bond assuming that its market price is equal to its principal.
    ///
    /// # Output
    /// - Par yield of the bond
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Par_yield#>
    /// - Original Source: N/A
    pub fn par_yield(&self) -> f64 {
        let discount_terms: f64 = self.discount_rate.iter().zip(self.cashflow.time_array().iter()).fold(0.0, |discount, (r, t)| {
            let discount_term: Compounding = Compounding::new(*r, &self.compounding_type);
            discount + discount_term.compounding_term(*t)
        } );
        self.compounding_frequency * (self.principal * (1.0 - discount_terms)) / discount_terms
    }

    /// Weigthted average of the times when bond's payoffs are made.
    ///
    /// Note: This method computes Macaulay duration.
    ///
    /// # Input
    /// - `modified`: Modifying the duration for periodic compounding
    ///
    /// # Output
    /// - Duration of the bond
    ///
    /// # LaTeX Formula
    ///- B = \\sum^{n}_{i=1}c_{i}e^{-yt_{i}}
    ///
    /// # Links
    /// - Wikipedia:<https://en.wikipedia.org/wiki/Bond_duration>
    /// - Original Source: N/A
    pub fn duration(&self, modified: bool) -> Result<f64, DigiFiError> {
        let ytm: f64 = self.yield_to_maturity(YtMMethod::Numerical)?;
        let last_index: usize = self.cashflow.len() - 1;
        let weighted_cashflows: f64 = (0..self.cashflow.len()).into_iter().fold(0.0, |weighted_cashflows, i| {
            let discount_term: Compounding = Compounding::new(ytm, &self.compounding_type);
            let cashflow: f64 = if i == last_index {
                self.cashflow.cashflow()[i] + self.principal
            } else {
                self.cashflow.cashflow()[i]
            };
            let time_step: f64 = self.cashflow.time_array()[i];
            weighted_cashflows + time_step * cashflow * discount_term.compounding_term(time_step)
        } );
        let duration: f64 = weighted_cashflows / self.initial_price;
        match self.compounding_type {
            CompoundingType::Periodic { .. } if modified => Ok(duration / (1.0 + ytm / self.compounding_frequency)),
            _ => Ok(duration), 
        }
    }

    /// Measure of the curvature in relationship between bond's prices and yields.
    ///
    /// # Input
    /// - `modified`: Modifying the convexity for periodic compounding
    ///
    /// # Output
    /// - Convexity of the bond
    ///
    /// # LaTeX Formula
    /// - C = \\frac{1}{B}\\sum^{n}_{i=1}c_{i}t^{2}_{i}e^{-yt_{i}}
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Bond_convexity>
    /// - Original Source: N/A
    pub fn convexity(&self, modified: bool) -> Result<f64, DigiFiError> {
        let ytm: f64 = self.yield_to_maturity(YtMMethod::Numerical)?;
        let last_index: usize = self.cashflow.len() - 1;
        let weighted_cashflows: f64 = (0..self.cashflow.len()).into_iter().fold(0.0, |weighted_cashflows, i| {
            let discount_term: Compounding = Compounding::new(ytm, &self.compounding_type);
            let cashflow: f64 = if i == last_index {
                self.cashflow.cashflow()[i] + self.principal
            } else {
                self.cashflow.cashflow()[i]
            };
            let time_step: f64 = self.cashflow.time_array()[i];
            let mut time: f64 = time_step.powi(2);
            // Convexity modified for periodic compounding
            if let CompoundingType::Periodic { .. } = self.compounding_type {
                if modified {
                    time += time_step / self.compounding_frequency;
                }
            }
            weighted_cashflows + time * cashflow * discount_term.compounding_term(time_step)
        } );
        let convexity: f64 = weighted_cashflows / self.initial_price;
        match self.compounding_type {
            CompoundingType::Periodic { .. } if modified => Ok(convexity / (1.0 + ytm / self.compounding_frequency).powi(2)),
            _ => Ok(convexity)
        }
    }

    /// Amount of coupon accumulated before the purchase of the bond.
    ///
    /// # Input
    /// - `time_since_last_coupon`: Period of time passed from last coupon payment
    /// - `time_separation`: Period of time between consecutive bond payments
    ///
    /// # Output
    /// - Accrued interest of the bond
    ///
    /// # LaTeX Formula
    /// - \\textit{Accrued Interest} = \\frac{Coupon}{\\textit{Compounding Frequency}}\\frac{\\textit{Time since last coupon payment}}{\\textit{Time separating coupon payments}}
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Accrued_interest>
    /// - Original Source: N/A
    pub fn accrued_interest(&self, time_since_last_coupon: f64, time_separation: f64) -> f64 {
        (self.coupon / self.compounding_frequency) * (time_since_last_coupon / time_separation)
    }

    /// Frequency with which the failure of bond repayment occurs.
    ///
    /// Note: It is assumed that the excess yield is the compensation for the possibility of default.
    ///
    ///
    /// # Input
    /// - `yield_spread`: Yield spread of the bond
    /// - `recovery_rate`: Assumed amount of bond's value that will be recovered
    ///
    /// # Output
    /// - Hazard rate of the bond
    ///
    /// # LaTeX Formula
    /// - \\textit{Hazard Rate} = \\frac{\\textit{Yield Spread}}{1 - \\text{Recovery Rate}}
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Failure_rate>
    /// - Original Source: N/A
    pub fn hazard_rate(&self, yield_spread: f64, recovery_rate: f64) -> f64 {
        yield_spread / (1.0 + recovery_rate)
    }

    /// Bond pricing via Taylor series expansion of bond price assuming it only depends on the yield.
    /// This method prices the bond at the next time step (i.e., next time step of coupon or principal payout).
    ///
    /// # Input
    /// - `current_price`: Current market price of the bond
    /// - `yield_spread`: Expected change in bond's yield between current and future time step
    ///
    /// # Output
    /// - Price of the bond in the next time step
    /// 
    /// # LaTeX Formula
    /// - B_{t}-B_{t-1} = \Delta B_{t} = \\frac{dB}{dy}\\Delta y + 0.5\\frac{d^{2}B}{dy^{2}}\Delta y^{2}
    pub fn bond_price_from_yield_spread(&self, current_price: f64, yield_spread: f64) -> Result<f64, DigiFiError> {
        let (duration, convexity) = match self.compounding_type {
            CompoundingType::Periodic { .. } => (self.duration(true)?, self.convexity(true)?),
            CompoundingType::Continuous => (self.duration(false)?, self.convexity(false)?),
        };
        let db_dy: f64 = -current_price * duration;
        let d2b_dy2: f64 = current_price * convexity;
        Ok(current_price + db_dy * yield_spread + 0.5 * d2b_dy2 * yield_spread.powi(2))
    }
}

impl ErrorTitle for Bond {
    fn error_title() -> String {
        String::from("Bond")
    }
}

impl FinancialInstrument for Bond {

    /// Present values of the bond.
    ////
    /// # Output
    /// - Present value of the bond's coupons and principal
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Bond_valuation>
    /// - Original Source: N/A
    fn present_value(&self) -> Result<f64, DigiFiError> {
        // Present value of coupon payments
        let coupon_pv: f64 = self.discount_rate.iter().zip(self.cashflow.cashflow().iter().zip(self.cashflow.time_array().iter()))
            .fold(0.0, |pv, (r, (cash, t))| {
                let discount_term: Compounding = Compounding::new(*r, &self.compounding_type);
                pv + cash * discount_term.compounding_term(*t)
            } );

        // Present value of principal and coupon payments
        let last_index: usize = self.cashflow.len() - 1;
        let principal_pv: f64 = self.principal * Compounding::new(self.discount_rate[last_index], &self.compounding_type).compounding_term(self.cashflow.time_array()[last_index]);
        Ok(coupon_pv + principal_pv)
    }

    /// Net present value of the bond.
    ///
    /// # Output
    /// - Present value of the bond minus the initial price it took to purchase the bond
    fn net_present_value(&self) -> Result<f64, DigiFiError> {
        Ok(self.present_value()? - self.initial_price)
    }

    /// Future value of the bond.
    ///
    /// # Output
    /// - Future value of the bond at it maturity (Computed from the present value of the bond)
    fn future_value(&self) -> Result<f64, DigiFiError> {
        let future_multiplicator: f64 = self.discount_rate.iter().zip(self.cashflow.time_array().iter())
            .fold(0.0, |fm, (r, t)| {
                let discount_term: Compounding = Compounding::new(*r, &self.compounding_type);
                fm / discount_term.compounding_term(*t)
            } );
        Ok(self.present_value()? * future_multiplicator)
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

impl PortfolioInstrument for Bond {

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
    fn unit_test_bootstrap() -> () {
        use crate::financial_instruments::bonds::bootstrap;
        let principals: Array1<f64> = Array1::from_vec(vec![100.0, 100.0, 100.0, 100.0, 100.0]);
        let maturities: Array1<f64> = Array1::from_vec(vec![0.25, 0.5, 1.0, 1.5, 2.0]);
        let coupons: Array1<f64> = Array1::from_vec(vec![0.0, 0.0, 0.0, 4.0, 5.0]);
        let prices: Array1<f64> = Array1::from_vec(vec![99.6, 99.0, 97.8, 102.5, 105.0]);
        let coupon_dt: Array1<f64> = Array1::from_vec(vec![0.25, 0.5, 1.0, 0.5, 0.5]);
        let spot_rates: Array1<f64> = bootstrap(principals, maturities, coupons, prices, coupon_dt).unwrap();
        let theor: Array1<f64> = Array1::from_vec(vec![0.01603, 0.0201, 0.02225, 0.02284, 0.02416]);
        assert!((spot_rates - theor).map(|v| v.abs() ).sum() < 10_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_bond() -> () {
        use crate::utilities::{ParameterType, Time, time_value_utils::{CompoundingType, present_value}};
        use crate::financial_instruments::{FinancialInstrument, FinancialInstrumentId, FinancialInstrumentType, AssetClass};
        use crate::financial_instruments::bonds::{BondType, Bond};
        use crate::portfolio_applications::AssetHistData;
        // Bond definition
        let compounding_type: CompoundingType = CompoundingType::Continuous;
        let financial_instrument_id: FinancialInstrumentId = FinancialInstrumentId {
            instrument_type: FinancialInstrumentType::CashInstrument, asset_class: AssetClass::DebtBasedInstrument, identifier: String::from("32198407128904"),
        };
        let asset_historical_data: AssetHistData = AssetHistData::build(
            Array1::from_vec(vec![0.4, 0.5]), Array1::from_vec(vec![0.0, 0.0]),
            Time::new(Array1::from_vec(vec![0.0, 1.0]))
        ).unwrap();
        let bond_type: BondType = BondType::AnnuityBond { principal: 100.0, coupon_rate: 0.05, maturity: 5.0, first_coupon_time: Some(1.0) };
        let discount_rate: ParameterType = ParameterType::Value { value: 0.02 };
        let bond: Bond = Bond::build(
            bond_type, discount_rate, 101.0, compounding_type, Some(0.0), financial_instrument_id,
            asset_historical_data, None
        ).unwrap();
        // Theoretical value
        let cashflow: Vec<f64> = vec![5.0, 5.0, 5.0, 5.0, 105.0];
        let time: Time = Time::new_from_range(1.0, 5.0, 1.0);
        let pv: f64 = present_value(cashflow.iter(), &time, ParameterType::Value { value: 0.02 }, &CompoundingType::Continuous).unwrap();
        assert!((bond.present_value().unwrap() - pv).abs() < TEST_ACCURACY);
    }
}