use std::io::Error;
use ndarray::{Array1, arr1};
use crate::utilities::{ParameterType, Time, compare_array_len, input_error, data_error, loss_functions::{LossFunction, MSE}, numerical_engines::nelder_mead};


#[derive(Clone)]
/// # Description
/// Type of discount rate compounding.
pub enum CompoundingType {
    Continuous,
    Periodic {
        /// Frequency at which compounding happens (e.g., for semi-annual compounding `frequency=0.5`)
        frequency: u32,
    },
}


/// # Description#
/// Present value of the cashflow discounted at a certain rate for every time period.
/// 
/// # Input
/// - `cashflow`: Array of cashflows
/// - `time`: Time settings
/// - `rate`: Value (array) of discount rate(s)
/// - `compounding_type`: Compounding type used to discount cashflows
/// - `compounding_frequency`: Compounding frequency of cashflows
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
/// let cashflow: Array1<f64> = Array1::from_vec(vec![10.0, 10.0, 10.0]);
/// let time: Time = Time::Sequence { times: Array1::from_vec(vec![1.0, 2.0, 3.0]) };
/// let rate: ParameterType = ParameterType::Value { value: 0.02 };
///
/// // Periodic compounding
/// let periodic_compounding: CompoundingType = CompoundingType::Periodic { frequency: 1 };
/// let pv_periodic: f64 = present_value(&cashflow, &time, rate.clone(), &periodic_compounding).unwrap();
/// assert!((pv_periodic - 10.0*(1.0/1.02 + 1.0/1.02_f64.powf(2.0) + 1.0/1.02_f64.powf(3.0))).abs() < TEST_ACCURACY);
///
/// // Continuous compounding
/// let continuous_compounding: CompoundingType = CompoundingType::Continuous;
/// let pv_continuous: f64 = present_value(&cashflow, &time, rate, &continuous_compounding).unwrap();
/// assert!((pv_continuous - 10.0*((-0.02_f64).exp() + (-0.02*2.0_f64).exp() + (-0.02*3.0_f64).exp())).abs() < TEST_ACCURACY);
/// ```
pub fn present_value(cashflow: &Array1<f64>, time: &Time, rate: ParameterType, compounding_type: &CompoundingType) -> Result<f64, Error> {
    let time_array = time.get_time_array();
    compare_array_len(&cashflow, &time_array, "cashflow", "time_array")?;
    let rates: Array1<f64> = match rate {
        ParameterType::Value { value } => {
            if (value <= -1.0) || (1.0 <= value) {
                return Err(data_error("Present Value: The argument rate must be defined in the interval (-1,1)."));
            }
            Array1::from_vec(vec![value; cashflow.len()])
        },
        ParameterType::TimeSeries { values } => {
            if values.mapv(|r| { if (r <= -1.0) || (1.0 <= r) {1.0} else {0.0} }).sum() != 0.0 {
                return Err(data_error("Present Value: All rates must be in the range (-1,1)."));
            }
            values
        },
    };
    let mut present_value: f64 = 0.0;
    for i in 0..time_array.len() {
        let discount_term: Compounding = Compounding::new(rates[i], &compounding_type);
        present_value += cashflow[i]*discount_term.compounding_term(time_array[i]);
    }
    Ok(present_value)
}


/// # Description
/// Net present value of the series of cashflows.
/// 
/// # Input
/// - `initial_cashflow`: Initial cashflow
/// - `cashflow`: Array of cashflows
/// - `time`: Time settings
/// - `rate`: Value (array) of discount rate(s)
/// - `compounding_type`: Compounding type used to discount cashflows
/// - `compounding_frequency`: Compounding frequency of cashflows
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
pub fn net_present_value(initial_cashflow: f64, cashflow: &Array1<f64>, time: &Time, rate: ParameterType, compounding_type: &CompoundingType) -> Result<f64, Error> {
    Ok(-initial_cashflow + present_value(cashflow, time, rate, compounding_type)?)
}


/// # Description
/// Future value of the cashflow with a certain interest rate at a specific time.
/// 
/// # Input
/// - `current_value`: Present value
/// - `rate`: Discount rate
/// - `time`: Time for which the future value is evaluated
/// - `compounding_type`: Compounding type used to discount cashflows
/// - `compounding_frequency`: Compounding frequency of cashflows
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
pub fn future_value(current_value: f64, rate: f64, time: f64, compounding_type: CompoundingType) -> Result<f64, Error> {
    if (rate <= -1.0) || (1.0 <= rate) {
        return Err(input_error("Future Value: The argument rate must be defined in the interval (-1,1)."));
    }
    let discount_term: Compounding = Compounding::new(rate, &compounding_type);
    Ok(current_value / discount_term.compounding_term(time))
}


/// # Description
/// Computes the internal rate of return under a certain compounding for the given series of cashflows.
/// 
/// # Input
/// - `initial_cashflow`: Initial cashflow
/// - `cashflow`: Array of cashflows
/// - `time`: Time settings
/// - `compounding_type`: Compounding type used to discount cashflows
/// - `compounding_frequency`: Compounding frequency of cashflows
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
/// let cashflow: Array1<f64> = Array1::from_vec(vec![200.0, 200.0, 900.0]);
/// let time: Time = Time::Sequence { times: Array1::from_vec(vec![1.0, 2.0, 3.0]) };
/// let compounding_type: CompoundingType = CompoundingType::Continuous;
///
/// let rate: f64 = internal_rate_of_return(1000.0, &cashflow, &time, &compounding_type).unwrap();
/// assert!((rate - 0.10459277343750006).abs() < 1_000.0 * TEST_ACCURACY);
/// ```
pub fn internal_rate_of_return(initial_cashflow: f64, cashflow: &Array1<f64>, time: &Time, compounding_type: &CompoundingType) -> Result<f64, Error> {
    let time_array = time.get_time_array();
    compare_array_len(&cashflow, &time_array, "cashflow", "time_array")?;
    let pv_closure  = | rate: &[f64] | {
        let mut present_value: f64 = 0.0;
        for i in 0..time_array.len() {
            let discount_term: Compounding = Compounding::new(rate[0], &compounding_type);
            present_value += cashflow[i] * discount_term.compounding_term(time_array[i]);
        }
        MSE.loss_array(&arr1(&[present_value]), &arr1(&[initial_cashflow])).unwrap()
    };
    let rate: Array1<f64> = nelder_mead(pv_closure, vec![0.0], Some(1_000), Some(1_000), None, None, None)?;
    Ok(rate[0])
}


/// # Description
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
pub fn real_interest_rate(nominal_interest_rate: f64, inflation_rate: f64) -> Result<f64, Error> {
    if inflation_rate == -1.0 {
        return Err(input_error("Real Interest Rate: The argument inflation_rate has a residual at -1.0. Change the value of the argument."));
    }
    Ok((1.0 + nominal_interest_rate)/(1.0 + inflation_rate) - 1.0)
}


/// # Descpition
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
    (new_frequency as f64) * ((1.0+current_rate/(current_frequency as f64)).powf((current_frequency as f64)/(new_frequency as f64)) - 1.0)
}


/// # Description
/// Periodic-to-continuous compounding transformation.
/// 
/// # Input
/// - `periodic_rate`: Current periodic discount rate
/// - `periodic_frequency`: Current compounding frequency
/// 
/// # Output
/// - Continuous discount rate
pub fn ptc_compounding_transformation(periodic_rate: f64, periodic_frequency: u32) -> f64 {
    (periodic_frequency as f64) * (1.0 + periodic_rate/(periodic_frequency as f64)).ln()
}


/// # Description
/// Continuous-to-periodic compounding transformation.
/// 
/// # Input
/// - `continuous_rate`: Current continuous discount rate
/// - `periodic_frequency`: Periodic compounding frequency
/// 
/// # Output
/// - Periodic discount rate
pub fn ctp_compounding_transformation(continuous_rate: f64, periodic_frequency: u32) -> f64 {
    (periodic_frequency as f64) * ((continuous_rate/(periodic_frequency as f64)).exp() - 1.0)
}


/// # Description
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

    /// # Description
    /// Creates a new `Compounding` instance.
    /// 
    /// # Input
    /// - `rate`: Discount rate
    /// - `compounding_type`: Type of compounding to use when computing the discount term
    pub fn new(rate: f64, compounding_type: &CompoundingType) -> Self {
        Compounding { rate, compounding_type: compounding_type.clone() }
    }

    /// # Description
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
            CompoundingType::Continuous => {
                (-self.rate * time).exp()
            },
            // Periodic compounding.
            CompoundingType::Periodic { frequency } => {
                (1.0 + self.rate/(frequency as f64)).powf(-(frequency as f64)*time)
            },
        }
    }

    /// # Description
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

    /// # Description
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

    /// # Description
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

    /// # Description
    /// Converts one compounding type an frequency to another type or frequency.
    /// 
    /// # Input
    /// - `new_compounding_type`: Compounding type to convert to
    /// - `new_compounding_frequency`: New compounding frequency
    pub fn compounding_transformation(&mut self, new_compounding_type: CompoundingType) -> () {
        match new_compounding_type {
            CompoundingType::Continuous => {
                match self.compounding_type {
                    CompoundingType::Continuous => (),
                    CompoundingType::Periodic { .. } => {
                        self.ptc_compounding_transformation()
                    },
                }
            },
            CompoundingType::Periodic { frequency } => {
                match self.compounding_type {
                    CompoundingType::Continuous => {
                        self.ctp_compounding_transformation(frequency)
                    },
                    CompoundingType::Periodic { .. } => {
                        self.ptp_compounding_transformation(frequency)
                    },
                }
            },
        }
    }
}


/// # Description
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
pub fn forward_rate(compounding_1: &Compounding, time_1: f64, compounding_2: &Compounding, time_2: f64) -> Result<f64, Error> {
    match (&compounding_1.compounding_type, &compounding_2.compounding_type) {
        (CompoundingType::Continuous, CompoundingType::Continuous) => {
            Ok((compounding_2.rate * time_2 - compounding_1.rate * time_1) / (time_2 - time_1))
        },
        (CompoundingType::Periodic { frequency: frequency_1 }, CompoundingType::Periodic { frequency: frequency_2 }) => {
            if frequency_1 != frequency_2 {
                return Err(data_error("Forward Rate: Frequencies of both compounding_1 and compounding_2 must be the same."));
            }
            Ok((compounding_1.compounding_term(time_1) / compounding_2.compounding_term(time_2)).powf(1.0 / (time_2 - time_1)) - 1.0)
        },
        _ => Err(data_error("Forward Rate: Compounding types for compounding_1 and compounding_2 must be the same.")),
    }
}


/// # Description
/// Base class for generating cashflow array with a base cashflow growth rate and inflation rate.
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, ParameterType, Time, Cashflow};
///
/// let cashflow_: ParameterType = ParameterType::Value { value: 10.0 }; 
/// let time: Time = Time::Range { initial_time: 0.0, final_time: 2.0, time_step: 1.0 };
///
/// let cashflow: Cashflow = Cashflow::new(cashflow_, time, 0.02, 0.015).unwrap();
///
/// assert!((cashflow.time_array() - Array1::from_vec(vec![0.0, 1.0, 2.0])).sum().abs() < TEST_ACCURACY);
/// assert!((cashflow.cashflow() - Array1::from_vec(vec![10.0, 10.0*1.02/1.015, 10.0*(1.02/1.015_f64).powf(2.0)])).sum().abs() < TEST_ACCURACY);
/// ```
pub struct Cashflow {
    /// Series of cashflows
    cashflow: Array1<f64>,
    /// Time parameters
    time: Time,
}

impl Cashflow {

    /// # Description
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
    pub fn new(cashflow: ParameterType, time: Time, cashflow_growth_rate: f64, inflation_rate: f64) -> Result<Self, Error> {
        if inflation_rate == -1.0 {
            return Err(input_error("Cashflow: The argument inflation_rate must not be equal to -1.0."));
        }
        let time_array: Array1<f64> = time.get_time_array();
        match cashflow {
            // Generate cashflow array
            ParameterType::Value { value } => {
                let array_size: usize = time_array.len();
                let mut cashflow_array: Array1<f64> = Array1::from_vec(vec![value; array_size]);
                if (cashflow_growth_rate != 0.0) || (inflation_rate != 0.0) {
                    for i in 1..array_size {
                        cashflow_array[i] = (1.0 + cashflow_growth_rate)*cashflow_array[i-1]/(1.0 + inflation_rate);
                    }
                }
                Ok(Cashflow { cashflow: cashflow_array, time })
            },
            // Cashflow array is provided
            ParameterType::TimeSeries { values } => {
                compare_array_len(&values, &time_array, "cashflow", "time_array")?;
                Ok(Cashflow { cashflow: values, time })
            },
        }
    }

    pub fn cashflow(&self) -> &Array1<f64> {
        &self.cashflow
    }

    pub fn time(&self) -> &Time {
        &self.time
    }

    pub fn time_array(&self) -> Array1<f64> {
        self.time.get_time_array()
    }
}


/// # Description
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
/// let perpetuity: Perpetuity = Perpetuity::new(10.0, 0.03, 0.02, compounding_type).unwrap();
///
/// assert!((perpetuity.present_value() - 10.0*0.02_f64.exp()/(0.01_f64.exp() - 1.0)).abs() < TEST_ACCURACY);
/// ```
pub struct Perpetuity {
    /// Constant cashflow of the perpetuity (Initial cashflow for a perpetuity with non-zero growth rate)
    perpetuity_cashflow: f64,
    /// Constant cashflow of the perpetuity (Initial cashflow for a perpetuity with non-zero growth rate)
    rate: f64,
    /// Growth rate of the cashflow at eadch time step
    perpetuity_growth_rate: f64,
    /// Compounding type used to discount cashflows
    compounding_type: CompoundingType,
}

impl Perpetuity {
    /// # Description
    /// Creates a new `Perpetuity` instance.
    /// 
    /// # Input
    /// - `perpetuity_cashflow`: Constant cashflow of the perpetuity (Initial cashflow for a perpetuity with non-zero growth rate)
    /// - `rate`: Constant cashflow of the perpetuity (Initial cashflow for a perpetuity with non-zero growth rate)
    /// - `perpetuity_growth_rate`: Growth rate of the cashflow at eadch time step
    /// - `compounding_type`: Compounding type used to discount cashflows
    /// 
    /// # Errors
    /// - Returns an error if the discount rate is smaller or equal to the perpetuity growth rate.
    pub fn new(perpetuity_cashflow: f64, rate: f64, perpetuity_growth_rate: f64, compounding_type: CompoundingType) -> Result<Self, Error> {
        if rate <= perpetuity_growth_rate {
            return Err(data_error("Perpetuity: The rate cannot be smaller or equal to the perpetuity growth rate."));
        }
        Ok(Perpetuity { perpetuity_cashflow, rate, perpetuity_growth_rate, compounding_type })
    }

    /// # Description
    /// Present value of the perpetuity.
    /// Note: Compounding frequency for periodic compounding does not affect the calculation.
    /// 
    /// # Output
    /// - Present value of the perpetuity
    pub fn present_value(&self) -> f64 {
        match self.compounding_type {
            CompoundingType::Continuous => {
                self.perpetuity_cashflow * self.perpetuity_growth_rate.exp() / ((self.rate - self.perpetuity_growth_rate).exp() - 1.0)
            },
            CompoundingType::Periodic { .. } => {
                self.perpetuity_cashflow / (self.rate - self.perpetuity_growth_rate)
            },
        }
    }

    /// # Description
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

    /// # Description
    /// Future value of the perpetuity.
    /// Note: Compounding frequency for periodic compounding does not affect the calculation.
    /// 
    /// # Output
    /// - Future value of the perpetuity
    pub fn future_value(&self, final_time: f64) -> f64 {
        match self.compounding_type {
            CompoundingType::Continuous => {
                self.present_value() * (self.rate*final_time).exp()
            },
            CompoundingType::Periodic { .. } => {
                self.present_value() * (1.0 + self.rate).powf(final_time)
            },
        }
    }
}


/// # Description
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
/// let annuity: Annuity = Annuity::new(10.0, 0.03, 3.0, 0.02, compounding_type).unwrap();
///
/// assert!((annuity.present_value() - 10.0 * ((-0.02_f64).exp()) / (0.01_f64.exp()-1.0) * (1.0-(-0.01*3.0_f64).exp())).abs() < TEST_ACCURACY);
/// ```
pub struct Annuity {
    /// Constant cashflow of the annuity (Initial cashflow for an annuity with non-zero growth rate)
    annuity_cashflow: f64,
    /// Discount rate
    rate: f64,
    /// Final time for annuity cashflows
    final_time_step: f64,
    /// Growth rate of the cashflow at eadch time step
    annuity_growth_rate: f64,
    /// Compounding type used to discount cashflows
    compounding_type: CompoundingType,
}

impl Annuity {
    /// # Description
    /// Creates a new `Annuity` instance.
    /// 
    /// # Input
    /// - `annuity_cashflow`: Constant cashflow of the annuity (Initial cashflow for an annuity with non-zero growth rate)
    /// - `rate`: Discount rate
    /// - `annuity_growth_rate`: Growth rate of the cashflow at eadch time step
    /// - `compounding_type`: Compounding type used to discount cashflows
    /// 
    /// # Errors
    /// - Returns an error if the discount rate is smaller or equal to the perpetuity growth rate.
    pub fn new(annuity_cashflow: f64, rate: f64, final_time_step: f64, annuity_growth_rate: f64, compounding_type: CompoundingType) -> Result<Self, Error> {
        if rate <= annuity_growth_rate {
            return Err(data_error("Annuity: The rate cannot be smaller or equal to the annuity growth rate."));
        }
        Ok(Annuity { annuity_cashflow, rate, final_time_step, annuity_growth_rate, compounding_type })
    }
    
    /// # Description
    /// Present value of the annuity.
    /// 
    /// # Output
    /// - Present value of the annuity
    pub fn present_value(&self) -> f64 {
        match self.compounding_type {
            CompoundingType::Continuous => {
                self.annuity_cashflow * (-self.annuity_growth_rate).exp() / ((self.rate-self.annuity_growth_rate).exp() - 1.0) * (1.0 - ((self.annuity_growth_rate-self.rate)*self.final_time_step).exp())
            },
            CompoundingType::Periodic { .. } => {
                self.annuity_cashflow / (self.rate-self.annuity_growth_rate) * (1.0 - ((1.0+self.annuity_growth_rate)/(1.0+self.rate)).powf(self.final_time_step))
            },
        }
    }

    /// # Description
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

    /// # Description
    /// Future value of the annuity.
    /// 
    /// # Output
    /// - Future value of the annuity
    pub fn future_value(&self) -> f64 {
        match self.compounding_type {
            CompoundingType::Continuous => {
                self.present_value() * (self.rate * self.final_time_step).exp()
            },
            CompoundingType::Periodic { .. } => {
                self.present_value() * (1.0 + self.rate).powf(self.final_time_step)
            },
        }
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
        let cashflow: Array1<f64> = Array1::from_vec(vec![10.0, 10.0, 10.0]);
        let time: Time = Time::Sequence { times: Array1::from_vec(vec![1.0, 2.0, 3.0]) };
        let rate: ParameterType = ParameterType::Value { value: 0.02 };
        // Periodic compounding
        let periodic_compounding: CompoundingType = CompoundingType::Periodic { frequency: 1 };
        let pv_periodic: f64 = present_value(&cashflow, &time, rate.clone(), &periodic_compounding).unwrap();
        assert!((pv_periodic - 10.0*(1.0/1.02 + 1.0/1.02_f64.powf(2.0) + 1.0/1.02_f64.powf(3.0))).abs() < TEST_ACCURACY);
        // Continuous compounding
        let continuous_compounding: CompoundingType = CompoundingType::Continuous;
        let pv_continuous: f64 = present_value(&cashflow, &time, rate, &continuous_compounding).unwrap();
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
        let cashflow: Array1<f64> = Array1::from_vec(vec![200.0, 200.0, 900.0]);
        let time: Time = Time::Sequence { times: Array1::from_vec(vec![1.0, 2.0, 3.0]) };
        let compounding_type: CompoundingType = CompoundingType::Continuous;
        let rate: f64 = internal_rate_of_return(1000.0, &cashflow, &time, &compounding_type).unwrap();
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
        let time: Time = Time::Range { initial_time: 0.0, final_time: 2.0, time_step: 1.0 };
        let cashflow: Cashflow = Cashflow::new(cashflow_, time, 0.02, 0.015).unwrap();
        assert!((cashflow.time_array() - Array1::from_vec(vec![0.0, 1.0, 2.0])).sum().abs() < TEST_ACCURACY);
        assert!((cashflow.cashflow() - Array1::from_vec(vec![10.0, 10.0*1.02/1.015, 10.0*(1.02/1.015_f64).powf(2.0)])).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_perpetuity_struct() -> () {
        use crate::utilities::time_value_utils::Perpetuity;
        let compounding_type: CompoundingType = CompoundingType::Continuous;
        let perpetuity: Perpetuity = Perpetuity::new(10.0, 0.03, 0.02, compounding_type).unwrap();
        assert!((perpetuity.present_value() - 10.0*0.02_f64.exp()/(0.01_f64.exp() - 1.0)).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_annuity_struct() -> () {
        use crate::utilities::time_value_utils::Annuity;
        let compounding_type: CompoundingType = CompoundingType::Continuous;
        let annuity: Annuity = Annuity::new(10.0, 0.03, 3.0, 0.02, compounding_type).unwrap();
        assert!((annuity.present_value() - 10.0 * ((-0.02_f64).exp()) / (0.01_f64.exp()-1.0) * (1.0-(-0.01*3.0_f64).exp())).abs() < TEST_ACCURACY);
    }
}