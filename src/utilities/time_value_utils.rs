use std::borrow::Borrow;
use ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::{ParameterType, Time, compare_len, loss_functions::{LossFunction, MSE}, numerical_engines::nelder_mead};


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Type of discount rate compounding.
pub enum CompoundingType {
    Continuous,
    Periodic {
        /// Frequency at which compounding happens (e.g., for semi-annual compounding `frequency=0.5`)
        frequency: u32,
    },
}

impl CompoundingType {
    /// Returns the  compounding frequency for the compounding type.
    /// 
    /// Note: For continuous compounding the frequency is set to `1` by default.
    pub fn frequency(&self) -> u32 {
        match self {
            Self::Periodic { frequency } => *frequency,
            Self::Continuous => 1,
        }
    }
}


/// Present value of the cashflow discounted at a certain rate for every time period.
/// 
/// # Input
/// - `cashflow`: Array of cashflows
/// - `time`: Time settings
/// - `rate`: Value (array) of discount rate(s)
/// - `compounding_type`: Compounding type used to discount cashflows
/// 
/// # Output
/// - Present value of series of cashflows
/// 
/// # Errors
/// - Returns an error if any rate is defined outside the `(-1,1)` interval.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Present_value>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, ParameterType, Time, CompoundingType, present_value};
///
/// let cashflow: Vec<f64> = vec![10.0, 10.0, 10.0];
/// let time: Time = Time::new(Array1::from_vec(vec![1.0, 2.0, 3.0]));
/// let rate: ParameterType = ParameterType::Value { value: 0.02 };
///
/// // Periodic compounding
/// let periodic_compounding: CompoundingType = CompoundingType::Periodic { frequency: 1 };
/// let pv_periodic: f64 = present_value(cashflow.iter(), &time, rate.clone(), &periodic_compounding).unwrap();
/// assert!((pv_periodic - 10.0*(1.0/1.02 + 1.0/1.02_f64.powf(2.0) + 1.0/1.02_f64.powf(3.0))).abs() < TEST_ACCURACY);
///
/// // Continuous compounding
/// let continuous_compounding: CompoundingType = CompoundingType::Continuous;
/// let pv_continuous: f64 = present_value(cashflow.iter(), &time, rate, &continuous_compounding).unwrap();
/// assert!((pv_continuous - 10.0*((-0.02_f64).exp() + (-0.02*2.0_f64).exp() + (-0.02*3.0_f64).exp())).abs() < TEST_ACCURACY);
/// ```
pub fn present_value<T, I>(cashflow: T, time: &Time, rate: ParameterType, compounding_type: &CompoundingType) -> Result<f64, DigiFiError>
where
    T: Iterator<Item = I> + ExactSizeIterator,
    I: Borrow<f64>
{
    let error_title: String = String::from("Present Value");
    compare_len(&cashflow, &time.time_array().iter(), "cashflow", "time_array")?;
    let rates: Array1<f64> = match rate {
        ParameterType::Value { value } => {
            if (value <= -1.0) || (1.0 <= value) {
                return Err(DigiFiError::ParameterConstraint {
                    title: error_title, 
                    constraint: "The argument `rate` must be defined in the interval `(-1,1)`.".to_owned(),
                });
            }
            Array1::from_vec(vec![value; cashflow.len()])
        },
        ParameterType::TimeSeries { values } => {
            if values.mapv(|r| { if (r <= -1.0) || (1.0 <= r) {1.0} else {0.0} }).sum() != 0.0 {
                return Err(DigiFiError::ParameterConstraint { title: error_title, constraint: "All `rates` must be in the range `(-1,1)`.".to_owned(), });
            }
            compare_len(&cashflow, &values.iter(), "cashflow", "rate")?;
            values
        },
    };
    Ok(rates.iter().zip(time.time_array().iter().zip(cashflow))
        .fold(0.0, |pv, (rate, (time, cash))| {
            let discount_term: Compounding = Compounding::new(*rate, &compounding_type);
            pv + cash.borrow() * discount_term.compounding_term(*time)
        } )
    )
}


/// Net present value of the series of cashflows.
/// 
/// # Input
/// - `initial_cashflow`: Initial cashflow
/// - `cashflow`: Array of cashflows
/// - `time`: Time settings
/// - `rate`: Value (array) of discount rate(s)
/// - `compounding_type`: Compounding type used to discount cashflows
/// 
/// # Output
/// - Present value of series of cashflows minus the initial cashflow
/// 
/// # Errors
/// - Returns an error if any rate is defined outside the `(-1,1)` interval.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Present_value#Net_present_value_of_a_stream_of_cash_flows>
/// - Original Source: N/A
pub fn net_present_value<T, I>(initial_cashflow: f64, cashflow: T, time: &Time, rate: ParameterType, compounding_type: &CompoundingType) -> Result<f64, DigiFiError>
where
    T: Iterator<Item = I> + ExactSizeIterator,
    I: Borrow<f64>,
{
    Ok(-initial_cashflow + present_value(cashflow, time, rate, compounding_type)?)
}


/// Future value of the cashflow with a certain interest rate at a specific time.
/// 
/// # Input
/// - `current_value`: Present value
/// - `rate`: Discount rate
/// - `time`: Time for which the future value is evaluated
/// - `compounding_type`: Compounding type used to discount cashflows
/// 
/// # Output
/// - Future value of the current cashflow
/// 
/// # Errors
/// - Returns an error if the rate is defined outside the `(-1,1)` interval.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Future_value>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::{TEST_ACCURACY, CompoundingType, future_value};
///
/// // Periodic compounding
/// let periodic_compounding: CompoundingType = CompoundingType::Periodic { frequency: 1 };
/// let fv_periodic: f64 = future_value(100.0, 0.03, 3.0, periodic_compounding).unwrap();
/// assert!((fv_periodic - 100.0*(1.03_f64).powf(3.0)).abs() < TEST_ACCURACY);
///
/// // Continuous compounding
/// let continuous_compounding: CompoundingType = CompoundingType::Continuous;
/// let fv_continuous: f64 = future_value(100.0, 0.03, 3.0, continuous_compounding).unwrap();
/// assert!((fv_continuous - 100.0*(0.03*3.0_f64).exp()).abs() < TEST_ACCURACY);
/// ```
pub fn future_value(current_value: f64, rate: f64, time: f64, compounding_type: CompoundingType) -> Result<f64, DigiFiError> {
    if (rate <= -1.0) || (1.0 <= rate) {
        return Err(DigiFiError::ParameterConstraint {
            title: "Future Value".to_owned(),
            constraint: "The argument `rate` must be defined in the interval `(-1,1)`.".to_owned(),
        });
    }
    let discount_term: Compounding = Compounding::new(rate, &compounding_type);
    Ok(current_value / discount_term.compounding_term(time))
}


/// Computes the internal rate of return under a certain compounding for the given series of cashflows.
/// 
/// # Input
/// - `initial_cashflow`: Initial cashflow
/// - `cashflow`: Array of cashflows
/// - `time`: Time settings
/// - `compounding_type`: Compounding type used to discount cashflows
/// 
/// # Output
/// - Internal rate of return that yields the initial cashflow by discounting future cashflows
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Internal_rate_of_return>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, Time, CompoundingType, internal_rate_of_return};
///
/// let cashflow: Vec<f64> = vec![200.0, 200.0, 900.0];
/// let time: Time = Time::new(Array1::from_vec(vec![1.0, 2.0, 3.0]));
/// let compounding_type: CompoundingType = CompoundingType::Continuous;
///
/// let rate: f64 = internal_rate_of_return(1000.0, cashflow.iter(), &time, &compounding_type).unwrap();
/// assert!((rate - 0.10459277343750006).abs() < 1_000.0 * TEST_ACCURACY);
/// ```
pub fn internal_rate_of_return<T, I>(initial_cashflow: f64, cashflow: T, time: &Time, compounding_type: &CompoundingType) -> Result<f64, DigiFiError>
where
    T: Iterator<Item = I> + ExactSizeIterator + Clone,
    I: Borrow<f64>,
{
    compare_len(&cashflow, &time.time_array().iter(), "cashflow", "time_array")?;
    let pv_closure = | rate: &[f64] | {
        let present_value: f64 = cashflow.clone().zip(time.time_array().iter())
            .fold(0.0, |pv, (cash, time)| {
                let discount_term: Compounding = Compounding::new(rate[0], &compounding_type);
                pv + cash.borrow() * discount_term.compounding_term(*time)
            } );
        MSE.loss(present_value, initial_cashflow)
    };
    let rate: Array1<f64> = nelder_mead(pv_closure, vec![0.0], Some(1_000), Some(1_000), None, None, None)?;
    Ok(rate[0])
}


/// A conversion from nominal interest rate to real interest rate based on inflation.
/// 
/// # Input
/// - `nominal_interest_rate`: Nominal interest rate
/// - `inflation_rate`: Rate of inflation
/// 
/// # Output
/// - Real interest rate
/// 
/// # Errors
/// - Returns an error if inflation rate is defined as `-1.0`.
pub fn real_interest_rate(nominal_interest_rate: f64, inflation_rate: f64) -> Result<f64, DigiFiError> {
    if inflation_rate == -1.0 {
        return Err(DigiFiError::ParameterConstraint {
            title: "Real Interest Rate".to_owned(),
            constraint: "The argument `inflation_rate` has a residual at `-1.0`.".to_owned(),
        });
    }
    Ok((1.0 + nominal_interest_rate)/(1.0 + inflation_rate) - 1.0)
}


/// Periodic-to-periodic compounding transformation between different compounding frequencies.
/// 
/// # Input
/// - `current_rate`: Current periodic discount rate
/// - `current_frequency`: Current compounding frequency
/// - `new_frequency`: New compounding frequency
/// 
/// # Output
/// - New periodic discount rate
pub fn ptp_compounding_transformation(current_rate: f64, current_frequency: u32, new_frequency: u32) -> f64 {
    if current_frequency == new_frequency { return current_rate; }
    let (current_frequency, new_frequency) = (current_frequency as f64, new_frequency as f64);
    new_frequency * ((1.0 + current_rate / current_frequency).powf(current_frequency / new_frequency) - 1.0)
}


/// Periodic-to-continuous compounding transformation.
/// 
/// # Input
/// - `periodic_rate`: Current periodic discount rate
/// - `periodic_frequency`: Current compounding frequency
/// 
/// # Output
/// - Continuous discount rate
pub fn ptc_compounding_transformation(periodic_rate: f64, periodic_frequency: u32) -> f64 {
    let periodic_frequency: f64 = periodic_frequency as f64;
    periodic_frequency * (1.0 + periodic_rate / periodic_frequency).ln()
}


/// Continuous-to-periodic compounding transformation.
/// 
/// # Input
/// - `continuous_rate`: Current continuous discount rate
/// - `periodic_frequency`: Periodic compounding frequency
/// 
/// # Output
/// - Periodic discount rate
pub fn ctp_compounding_transformation(continuous_rate: f64, periodic_frequency: u32) -> f64 {
    let periodic_frequency: f64 = periodic_frequency as f64;
    periodic_frequency * ((continuous_rate / periodic_frequency).exp() - 1.0)
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Different compounding techniques and methods.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Compound_interest>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, CompoundingType, Compounding, ptc_compounding_transformation};
///
/// let mut compounding: Compounding = Compounding::new(0.03, &CompoundingType::Periodic{ frequency: 2 });
/// let periodic_compounding_term: f64 = compounding.compounding_term(3.0);
///
/// assert!((periodic_compounding_term - (1.0_f64 + 0.03/2.0).powf(-2.0*3.0)).abs() < TEST_ACCURACY);
///
/// compounding.compounding_transformation(CompoundingType::Continuous);
/// assert!((compounding.compounding_term(3.0) - (-ptc_compounding_transformation(0.03, 2)*3.0_f64).exp()).abs() < TEST_ACCURACY);
///
/// compounding.compounding_transformation(CompoundingType::Periodic { frequency: 2 });
/// assert!((periodic_compounding_term - compounding.compounding_term(3.0)).abs() < TEST_ACCURACY);
/// ```
pub struct Compounding {
    /// Dicount rate
    rate: f64,
    /// Compounding type used to discount cashflows
    compounding_type: CompoundingType,
}

impl Compounding {
    /// Creates a new `Compounding` instance.
    /// 
    /// # Input
    /// - `rate`: Discount rate
    /// - `compounding_type`: Type of compounding to use when computing the discount term
    pub fn new(rate: f64, compounding_type: &CompoundingType) -> Self {
        Self { rate, compounding_type: *compounding_type }
    }

    /// Defines a compounding term with either continuous or periodic compounding.
    /// 
    /// Note: Compounding term is defined as the discounting terms for future cashflows.
    /// 
    /// # Input
    /// - `time`: Time at which to discount
    /// 
    /// # Output
    /// - Discounting term for future cashflows
    pub fn compounding_term(&self, time: f64) -> f64 {
        match self.compounding_type {
            // Continuous compounding.
            CompoundingType::Continuous => (-self.rate * time).exp(),
            // Periodic compounding.
            CompoundingType::Periodic { frequency } => {
                let frequency: f64 = frequency as f64;
                (1.0 + self.rate / frequency).powf(-frequency * time)
            },
        }
    }

    /// Periodic-to-periodic compopunding transformation.
    fn ptp_compounding_transformation(&mut self, new_frequency: u32) -> () {
        match self.compounding_type {
            CompoundingType::Continuous => (),
            CompoundingType::Periodic { frequency } => {
                self.rate = ptp_compounding_transformation(self.rate, frequency, new_frequency);
                self.compounding_type = CompoundingType::Periodic { frequency: new_frequency };
            },
        }
    }

    /// Periodic-to-continuous compounding transformation.
    fn ptc_compounding_transformation(&mut self) -> () {
        match self.compounding_type {
            CompoundingType::Continuous => (),
            CompoundingType::Periodic { frequency } => {
                self.rate = ptc_compounding_transformation(self.rate, frequency);
                self.compounding_type = CompoundingType::Continuous;
            },
        }
    }

    /// Continuous-to-periodic compounding transformation.
    fn ctp_compounding_transformation(&mut self, new_frequency: u32) -> () {
        match self.compounding_type {
            CompoundingType::Continuous => {
                self.rate = ctp_compounding_transformation(self.rate, new_frequency);
                self.compounding_type = CompoundingType::Periodic { frequency: new_frequency };
            },
            CompoundingType::Periodic { .. } => (),
        }
    }

    /// Converts one compounding type an frequency to another type or frequency.
    /// 
    /// # Input
    /// - `new_compounding_type`: Compounding type to convert to
    pub fn compounding_transformation(&mut self, new_compounding_type: CompoundingType) -> () {
        match (self.compounding_type, new_compounding_type) {
            (CompoundingType::Periodic { .. }, CompoundingType::Continuous) => self.ptc_compounding_transformation(),
            (CompoundingType::Periodic { .. }, CompoundingType::Periodic { frequency }) => self.ptp_compounding_transformation(frequency),
            (CompoundingType::Continuous, CompoundingType::Periodic { frequency }) => self.ctp_compounding_transformation(frequency),
            _ => (),
        }
    }
}


/// Forward interest rate for the period between time_1 and time_2.
/// 
/// # Input
/// - `compounding_1`: Compounding term (i.e., term that defines the rate, compounding type and frequency) at time step 1
/// - `time_1`: Time step 1
/// - `compounding_2`: Compounding term (i.e., term that defines the rate, compounding type and frequency) at time step 2
/// - `time_2`: Time step 2
/// 
/// # Output
/// - Forward rate
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Forward_rate>
/// - Original Source: N/A
pub fn forward_rate(compounding_1: &Compounding, time_1: f64, compounding_2: &Compounding, time_2: f64) -> Result<f64, DigiFiError> {
    let error_title: String = String::from("Forward Rate");
    match (&compounding_1.compounding_type, &compounding_2.compounding_type) {
        (CompoundingType::Continuous, CompoundingType::Continuous) => {
            Ok((compounding_2.rate * time_2 - compounding_1.rate * time_1) / (time_2 - time_1))
        },
        (CompoundingType::Periodic { frequency: frequency_1 }, CompoundingType::Periodic { frequency: frequency_2 }) => {
            if frequency_1 != frequency_2 {
                return Err(DigiFiError::ValidationError {
                    title: error_title,
                    details: "Frequencies of both `compounding_1` and `compounding_2` must be the same.".to_owned(),
                });
            }
            Ok((compounding_1.compounding_term(time_1) / compounding_2.compounding_term(time_2)).powf(1.0 / (time_2 - time_1)) - 1.0)
        },
        _ => Err(DigiFiError::ValidationError {
            title: error_title,
            details: "Compounding types for `compounding_1` and `compounding_2` must have the same `CompoundingType`.".to_owned(),
        }),
    }
}


#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Base class for generating cashflow array with a base cashflow growth rate and inflation rate.
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, ParameterType, Time, Cashflow};
///
/// let cashflow_: ParameterType = ParameterType::Value { value: 10.0 }; 
/// let time: Time = Time::new_from_range(0.0, 2.0, 1.0);
///
/// let cashflow: Cashflow = Cashflow::build(cashflow_, time, 0.02, 0.015).unwrap();
///
/// assert!((cashflow.time().time_array() - Array1::from_vec(vec![0.0, 1.0, 2.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((cashflow.cashflow() - Array1::from_vec(vec![10.0, 10.0*1.02/1.015, 10.0*(1.02/1.015_f64).powf(2.0)])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub struct Cashflow {
    /// Series of cashflows
    cashflow: Array1<f64>,
    /// Time parameters
    time: Time,
}

impl Cashflow {
    /// Creates a new `Cashflow` instance.
    /// 
    /// # Input
    /// - `cashflow`: Cashflow settings
    /// - `time`: Time settings
    /// - `cashflow_growth_rate`: Growth rate of the cashflow
    /// - `inflation_rate`: Inflation rate to discount cashflows by
    /// 
    /// # Errors
    /// - Returns an error if `inflation_rate` is set to `-1.0`. 
    /// - Returns an error if cashflow and time settings do not generate arrays of the same length.
    pub fn build(cashflow: ParameterType, time: Time, cashflow_growth_rate: f64, inflation_rate: f64) -> Result<Self, DigiFiError> {
        if inflation_rate == -1.0 {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The argument `inflation_rate` must not be equal to `-1.0`.".to_owned(),
            });
        }
        match cashflow {
            // Generate cashflow array
            ParameterType::Value { value } => {
                let array_size: usize = time.len();
                let cashflow: Vec<f64> = if (cashflow_growth_rate != 0.0) || (inflation_rate != 0.0) {
                    (0..array_size).into_iter()
                        .fold((Vec::with_capacity(array_size), value), |(mut cash, prev), _| {
                            cash.push(prev);
                            (cash, (1.0 + cashflow_growth_rate) * prev / (1.0 + inflation_rate))
                        } ).0
                } else {
                    vec![value; array_size]
                };
                Ok(Self { cashflow: Array1::from_vec(cashflow), time })
            },
            // Cashflow array is provided
            ParameterType::TimeSeries { values } => {
                compare_len(&values.iter(), &time.time_array().iter(), "cashflow", "time_array")?;
                Ok(Self { cashflow: values, time })
            },
        }
    }

    pub fn len(&self) -> usize {
        self.cashflow.len()
    }

    pub fn cashflow(&self)-> &Array1<f64> {
        &self.cashflow
    }

    pub fn time(&self) -> &Time {
        &self.time
    }

    pub fn time_array(&self) -> &Array1<f64> {
        &self.time.time_array()
    }
}

impl ErrorTitle for Cashflow {
    fn error_title() -> String {
        String::from("Cashflow")
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// A series of fixed income cashflows paid out each time step forever.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Perpetuity>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, CompoundingType, Perpetuity};
///
/// let compounding_type: CompoundingType = CompoundingType::Continuous;
/// let perpetuity: Perpetuity = Perpetuity::build(10.0, 0.03, 0.02, compounding_type).unwrap();
///
/// assert!((perpetuity.present_value() - 10.0*0.02_f64.exp()/(0.01_f64.exp() - 1.0)).abs() < TEST_ACCURACY);
/// ```
pub struct Perpetuity {
    /// Constant cashflow of the perpetuity (Initial cashflow for a perpetuity with non-zero growth rate)
    cashflow: f64,
    /// Discount rate
    rate: f64,
    /// Growth rate of the cashflow at each time step
    growth_rate: f64,
    /// Compounding type used to discount cashflows
    compounding_type: CompoundingType,
}

impl Perpetuity {
    /// Creates a new `Perpetuity` instance.
    /// 
    /// # Input
    /// - `cashflow`: Constant cashflow of the perpetuity (Initial cashflow for a perpetuity with non-zero growth rate)
    /// - `rate`: Discount rate
    /// - `growth_rate`: Growth rate of the cashflow at each time step
    /// - `compounding_type`: Compounding type used to discount cashflows
    /// 
    /// # Errors
    /// - Returns an error if the discount rate is smaller or equal to the perpetuity growth rate.
    pub fn build(cashflow: f64, rate: f64, growth_rate: f64, compounding_type: CompoundingType) -> Result<Self, DigiFiError> {
        if rate <= growth_rate {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The `rate` cannot be smaller or equal to the `perpetuity_growth_rate`.".to_owned(),
            });
        }
        Ok(Self { cashflow, rate, growth_rate, compounding_type })
    }

    /// Present value of the perpetuity.
    /// Note: Compounding frequency for periodic compounding does not affect the calculation.
    /// 
    /// # Output
    /// - Present value of the perpetuity
    pub fn present_value(&self) -> f64 {
        match self.compounding_type {
            CompoundingType::Continuous => self.cashflow * self.growth_rate.exp() / ((self.rate - self.growth_rate).exp() - 1.0),
            CompoundingType::Periodic { .. } => self.cashflow / (self.rate - self.growth_rate),
        }
    }

    /// Net present value of the perpetuity.
    /// Note: Compounding frequency for periodic compounding does not affect the calculation.
    /// 
    /// # Input
    /// - `initial_cashflow`: Initial cashflow
    /// 
    /// # Output
    /// - Net present value of the perpetuity
    pub fn net_present_value(&self, initial_cashflow: f64) -> f64 {
        -initial_cashflow + self.present_value()
    }

    /// Future value of the perpetuity.
    /// Note: Compounding frequency for periodic compounding does not affect the calculation.
    /// 
    /// # Output
    /// - Future value of the perpetuity
    pub fn future_value(&self, final_time: f64) -> f64 {
        match self.compounding_type {
            CompoundingType::Continuous => self.present_value() * (self.rate * final_time).exp(),
            CompoundingType::Periodic { .. } => self.present_value() * (1.0 + self.rate).powf(final_time),
        }
    }
}

impl ErrorTitle for Perpetuity {
    fn error_title() -> String {
        String::from("Perpetuity")
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// A series of fixed income cashflows paid out for a specified number of time periods periods.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Annuity>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, CompoundingType, Annuity};
///
/// let compounding_type: CompoundingType = CompoundingType::Continuous;
/// let annuity: Annuity = Annuity::build(10.0, 0.03, 3.0, 0.02, compounding_type).unwrap();
///
/// assert!((annuity.present_value() - 10.0 * ((-0.02_f64).exp()) / (0.01_f64.exp()-1.0) * (1.0-(-0.01*3.0_f64).exp())).abs() < TEST_ACCURACY);
/// ```
pub struct Annuity {
    /// Constant cashflow of the annuity (Initial cashflow for an annuity with non-zero growth rate)
    cashflow: f64,
    /// Discount rate
    rate: f64,
    /// Final time for annuity cashflows
    t_f: f64,
    /// Growth rate of the cashflow at eadch time step
    growth_rate: f64,
    /// Compounding type used to discount cashflows
    compounding_type: CompoundingType,
}

impl Annuity {
    /// Creates a new `Annuity` instance.
    /// 
    /// # Input
    /// - `cashflow`: Constant cashflow of the annuity (Initial cashflow for an annuity with non-zero growth rate)
    /// - `rate`: Discount rate
    /// - `t_f`: Final time for annuity cashflows
    /// - `growth_rate`: Growth rate of the cashflow at eadch time step
    /// - `compounding_type`: Compounding type used to discount cashflows
    /// 
    /// # Errors
    /// - Returns an error if the discount rate is smaller or equal to the perpetuity growth rate.
    pub fn build(cashflow: f64, rate: f64, t_f: f64, growth_rate: f64, compounding_type: CompoundingType) -> Result<Self, DigiFiError> {
        if rate <= growth_rate {
            return Err(DigiFiError::ParameterConstraint {
                title: Self::error_title(),
                constraint: "The `rate` must be larger the `annuity_growth_rate`.".to_owned(),
            });
        }
        Ok(Self { cashflow, rate, t_f, growth_rate, compounding_type })
    }
    
    /// Present value of the annuity.
    /// 
    /// # Output
    /// - Present value of the annuity
    pub fn present_value(&self) -> f64 {
        match self.compounding_type {
            CompoundingType::Continuous => {
                self.cashflow * (-self.growth_rate).exp() / ((self.rate-self.growth_rate).exp() - 1.0) * (1.0 - ((self.growth_rate - self.rate) * self.t_f).exp())
            },
            CompoundingType::Periodic { .. } => {
                self.cashflow / (self.rate-self.growth_rate) * (1.0 - ((1.0 + self.growth_rate) / (1.0 + self.rate)).powf(self.t_f))
            },
        }
    }

    /// Net present value of the annuity.
    /// 
    /// # Input
    /// - `initial_cashflow`: Initial cashflow
    /// 
    /// # Output
    /// - Net present value of the annuity
    pub fn net_present_value(&self, initial_cashflow: f64) -> f64 {
        - initial_cashflow * self.present_value()
    }

    /// Future value of the annuity.
    /// 
    /// # Output
    /// - Future value of the annuity
    pub fn future_value(&self) -> f64 {
        match self.compounding_type {
            CompoundingType::Continuous => self.present_value() * (self.rate * self.t_f).exp(),
            CompoundingType::Periodic { .. } => self.present_value() * (1.0 + self.rate).powf(self.t_f),
        }
    }
}

impl ErrorTitle for Annuity {
    fn error_title() -> String {
        String::from("Annuity")
    }
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::utilities::{TEST_ACCURACY, ParameterType, Time};
    use crate::utilities::time_value_utils::CompoundingType;

    #[test]
    fn unit_test_present_value() -> () {
        use crate::utilities::time_value_utils::present_value;
        let cashflow: Vec<f64> = vec![10.0, 10.0, 10.0];
        let time: Time = Time::new(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let rate: ParameterType = ParameterType::Value { value: 0.02 };
        // Periodic compounding
        let periodic_compounding: CompoundingType = CompoundingType::Periodic { frequency: 1 };
        let pv_periodic: f64 = present_value(cashflow.iter(), &time, rate.clone(), &periodic_compounding).unwrap();
        assert!((pv_periodic - 10.0*(1.0/1.02 + 1.0/1.02_f64.powf(2.0) + 1.0/1.02_f64.powf(3.0))).abs() < TEST_ACCURACY);
        // Continuous compounding
        let continuous_compounding: CompoundingType = CompoundingType::Continuous;
        let pv_continuous: f64 = present_value(cashflow.iter(), &time, rate, &continuous_compounding).unwrap();
        assert!((pv_continuous - 10.0*((-0.02_f64).exp() + (-0.02*2.0_f64).exp() + (-0.02*3.0_f64).exp())).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_future_value() -> () {
        use crate::utilities::time_value_utils::future_value;
        // Periodic compounding
        let periodic_compounding: CompoundingType = CompoundingType::Periodic { frequency: 1 };
        let fv_periodic: f64 = future_value(100.0, 0.03, 3.0, periodic_compounding).unwrap();
        assert!((fv_periodic - 100.0*(1.03_f64).powf(3.0)).abs() < TEST_ACCURACY);
        // Continuous compounding
        let continuous_compounding: CompoundingType = CompoundingType::Continuous;
        let fv_continuous: f64 = future_value(100.0, 0.03, 3.0, continuous_compounding).unwrap();
        assert!((fv_continuous - 100.0*(0.03*3.0_f64).exp()).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_internal_rate_of_return() -> () {
        use crate::utilities::time_value_utils::internal_rate_of_return;
        let cashflow: Vec<f64> = vec![200.0, 200.0, 900.0];
        let time: Time = Time::new(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        let compounding_type: CompoundingType = CompoundingType::Continuous;
        let rate: f64 = internal_rate_of_return(1000.0, cashflow.iter(), &time, &compounding_type).unwrap();
        assert!((rate - 0.10459277343750006).abs() < 1_000.0 * TEST_ACCURACY);
    }

    #[test]
    fn unit_test_compounding_struct() -> () {
        use crate::utilities::time_value_utils::{Compounding, CompoundingType, ptc_compounding_transformation};
        let mut compounding: Compounding = Compounding::new(0.03, &CompoundingType::Periodic{ frequency: 2 });
        let periodic_compounding_term: f64 = compounding.compounding_term(3.0);
        assert!((periodic_compounding_term - (1.0_f64 + 0.03/2.0).powf(-2.0*3.0)).abs() < TEST_ACCURACY);
        compounding.compounding_transformation(CompoundingType::Continuous);
        assert!((compounding.compounding_term(3.0) - (-ptc_compounding_transformation(0.03, 2)*3.0_f64).exp()).abs() < TEST_ACCURACY);
        compounding.compounding_transformation(CompoundingType::Periodic { frequency: 2 });
        assert!((periodic_compounding_term - compounding.compounding_term(3.0)).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_cashflow_struct() -> () {
        use crate::utilities::time_value_utils::Cashflow;
        let cashflow_: ParameterType = ParameterType::Value { value: 10.0 }; 
        let time: Time = Time::new_from_range(0.0, 2.0, 1.0);
        let cashflow: Cashflow = Cashflow::build(cashflow_, time, 0.02, 0.015).unwrap();
        assert!((cashflow.time_array() - Array1::from_vec(vec![0.0, 1.0, 2.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((cashflow.cashflow() - Array1::from_vec(vec![10.0, 10.0*1.02/1.015, 10.0*(1.02/1.015_f64).powf(2.0)])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_perpetuity_struct() -> () {
        use crate::utilities::time_value_utils::Perpetuity;
        let compounding_type: CompoundingType = CompoundingType::Continuous;
        let perpetuity: Perpetuity = Perpetuity::build(10.0, 0.03, 0.02, compounding_type).unwrap();
        assert!((perpetuity.present_value() - 10.0*0.02_f64.exp()/(0.01_f64.exp() - 1.0)).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_annuity_struct() -> () {
        use crate::utilities::time_value_utils::Annuity;
        let compounding_type: CompoundingType = CompoundingType::Continuous;
        let annuity: Annuity = Annuity::build(10.0, 0.03, 3.0, 0.02, compounding_type).unwrap();
        assert!((annuity.present_value() - 10.0 * ((-0.02_f64).exp()) / (0.01_f64.exp()-1.0) * (1.0-(-0.01*3.0_f64).exp())).abs() < TEST_ACCURACY);
    }
}