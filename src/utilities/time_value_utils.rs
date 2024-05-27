use std::f64::consts::E;

#[derive(Debug, PartialEq, Clone, Copy)]
enum CompoundingType {
    Continuous,
    Periodic,
}

fn compare_array_len<T>(array_1: &[T], array_2: &[T], array_1_name: &str, array_2_name: &str) {
    if array_1.len() != array_2.len() {
        panic!("The length of {} and {} do not coincide.", array_1_name, array_2_name);
    }
}

fn type_check<T>(_value: &T, _value_name: &str) {
    // Type checking is implicit in Rust, so this is a no-op.
}

struct Cashflow {
    cashflow: Vec<f64>,
    time_array: Vec<f64>,
    cashflow_growth_rate: f64,
    inflation_rate: f64,
}

impl Cashflow {
    pub fn new(
        cashflow: Vec<f64>,
        final_time: f64,
        start_time: f64,
        time_step: f64,
        cashflow_growth_rate: f64,
        inflation_rate: f64,
    ) -> Cashflow {
        let time_array = Self::generate_time_array(final_time, start_time, time_step);
        Cashflow {
            cashflow,
            time_array,
            cashflow_growth_rate,
            inflation_rate,
        }
    }

    fn generate_time_array(final_time: f64, start_time: f64, time_step: f64) -> Vec<f64> {
        let mut time_array = Vec::new();
        let mut time = start_time;
        while time <= final_time {
            time_array.push(time);
            time += time_step;
        }
        time_array
    }

    fn cashflow_array(&self) -> Vec<f64> {
        let mut cashflow_array = vec![self.cashflow[0]; self.time_array.len()];
        if self.cashflow_growth_rate != 0.0 || self.inflation_rate != 0.0 {
            cashflow_array[0] /= 1.0 + self.inflation_rate;
            for i in 1..self.time_array.len() {
                cashflow_array[i] = (1.0 + self.cashflow_growth_rate) * cashflow_array[i - 1] / (1.0 + self.inflation_rate);
            }
        }
        cashflow_array
    }
}

fn present_value(
    cashflow: Vec<f64>, 
    time_array: Vec<f64>, 
    rate: f64, 
    compounding_type: CompoundingType, 
    compounding_frequency: usize
) -> f64 {
    compare_array_len(&cashflow, &time_array, "cashflow", "time_array");
    let mut present_value = 0.0;
    for i in 0..time_array.len() {
        let discount_term = Compounding::new(rate, compounding_type, compounding_frequency);
        let discount_factor = discount_term.compounding_term(time_array[i]);
        println!("cashflow[{}]: {}, discount_factor: {}", i, cashflow[i], discount_factor); // Print statement
        present_value += cashflow[i] * discount_factor;
    }
    println!("Final Present Value: {}", present_value); // Print statement
    present_value
}

fn net_present_value(
    initial_cashflow: f64, 
    cashflow: Vec<f64>, 
    time_array: Vec<f64>, 
    rate: f64, 
    compounding_type: CompoundingType, 
    compounding_frequency: usize
) -> f64 {
    let pv = present_value(
        cashflow.clone(), 
        time_array.clone(), 
        rate, 
        compounding_type.clone(), 
        compounding_frequency
    );
    println!("Initial Cashflow: {}, Present Value: {}, Net Present Value: {}", initial_cashflow, pv, -initial_cashflow + pv); // Print statement
    -initial_cashflow + pv
}

fn future_value(
    current_value: f64, 
    rate: f64, 
    time: f64, 
    compounding_type: CompoundingType, 
    compounding_frequency: usize
) -> f64 {
    if rate < -1.0 || rate > 1.0 {
        panic!("The argument rate must be defined in in the interval [-1,1].");
    }
    let discount_term = Compounding::new(rate, compounding_type, compounding_frequency);
    let fv = current_value / discount_term.compounding_term(time);
    println!("Current Value: {}, Rate: {}, Time: {}, Future Value: {}", current_value, rate, time, fv); // Print statement
    fv
}

fn internal_rate_of_return(
    initial_cashflow: f64,
    cashflow: &[f64],
    time_array: &[f64],
    compounding_type: CompoundingType,
    compounding_frequency: usize,
) -> f64 {
    fn bisection<F>(mut f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> f64
    where
        F: FnMut(f64) -> f64,
    {
        let mut fa = f(a);
        let mut fb = f(b);
        assert!(fa * fb < 0.0, "The function must have different signs at a and b.");

        for _ in 0..max_iter {
            let c = (a + b) / 2.0;
            let fc = f(c);

            if fc == 0.0 || (b - a) / 2.0 < tol {
                return c;
            }

            if fc * fa < 0.0 {
                b = c;
                fb = fc;
            } else {
                a = c;
                fa = fc;
            }
        }

        (a + b) / 2.0
    }

    let cashflow_series = |rate: f64| -> f64 {
        present_value(cashflow.to_vec(), time_array.to_vec(), rate, compounding_type, compounding_frequency) - initial_cashflow
    };

    bisection(cashflow_series, -1.0, 1.0, 1e-7, 100)
}

fn ptp_compounding_transformation(current_rate: f64, current_frequency: usize, new_frequency: usize) -> f64 {
    new_frequency as f64 * ((1.0 + current_rate / current_frequency as f64).powf(current_frequency as f64 / new_frequency as f64) - 1.0)
}

fn ptc_compounding_transformation(periodic_rate: f64, periodic_frequency: usize) -> f64 {
    periodic_frequency as f64 * (1.0 + periodic_rate / periodic_frequency as f64).ln()
}

fn ctp_compounding_transformation(continuous_rate: f64, periodic_frequency: usize) -> f64 {
    periodic_frequency as f64 * (continuous_rate / periodic_frequency as f64).exp_m1()
}

fn real_interest_rate(nominal_interest_rate: f64, inflation: f64) -> f64 {
    (1.0 + nominal_interest_rate) / (1.0 + inflation) - 1.0
}

struct Compounding {
    rate: f64,
    compounding_type: CompoundingType,
    compounding_frequency: usize,
}

impl Compounding {
    pub fn new(rate: f64, compounding_type: CompoundingType, compounding_frequency: usize) -> Compounding {
        if compounding_type == CompoundingType::Periodic && compounding_frequency <= 0 {
            panic!("For periodic compounding, the compounding_frequency must be defined.");
        }
        Compounding {
            rate,
            compounding_type,
            compounding_frequency,
        }
    }

    fn continuous_compounding_term(&self, time: f64) -> f64 {
        (-self.rate * time).exp()
    }

    fn periodic_compounding_term(&self, time: f64) -> f64 {
        (1.0 + self.rate / self.compounding_frequency as f64).powf(self.compounding_frequency as f64 * time)
    }

    pub fn compounding_term(&self, time: f64) -> f64 {
        match self.compounding_type {
            CompoundingType::Continuous => self.continuous_compounding_term(time),
            CompoundingType::Periodic => self.periodic_compounding_term(time),
        }
    }

    pub fn compounding_transformation(
        &mut self,
        new_compounding_type: CompoundingType,
        new_compounding_frequency: usize,
    ) {
        if new_compounding_type == CompoundingType::Periodic && new_compounding_frequency <= 0 {
            panic!("For periodic compounding, the compounding_frequency must be defined.");
        }
        if new_compounding_type == CompoundingType::Continuous && self.compounding_type == CompoundingType::Periodic {
            self.rate = ptc_compounding_transformation(self.rate, self.compounding_frequency);
            self.compounding_frequency = 1;
        }
        if new_compounding_type == CompoundingType::Periodic && self.compounding_type == CompoundingType::Continuous {
            self.rate = ctp_compounding_transformation(self.rate, new_compounding_frequency);
            self.compounding_frequency = new_compounding_frequency;
        }
        if new_compounding_type == CompoundingType::Periodic && self.compounding_type == CompoundingType::Periodic {
            self.rate = ptp_compounding_transformation(self.rate, self.compounding_frequency, new_compounding_frequency);
            self.compounding_frequency = new_compounding_frequency;
        }
        self.compounding_type = new_compounding_type;
    }
}

struct Perpetuity {
    perpetuity_cashflow: f64,
    rate: f64,
    perpetuity_growth_rate: f64,
    compounding_type: CompoundingType,
}

impl Perpetuity {
    pub fn new(
        perpetuity_cashflow: f64,
        rate: f64,
        perpetuity_growth_rate: f64,
        compounding_type: CompoundingType,
    ) -> Perpetuity {
        if rate <= perpetuity_growth_rate {
            panic!("The rate cannot be smaller than the perpetuity growth rate.");
        }
        Perpetuity {
            perpetuity_cashflow,
            rate,
            perpetuity_growth_rate,
            compounding_type,
        }
    }

    pub fn present_value(&self) -> f64 {
        match self.compounding_type {
            CompoundingType::Periodic => self.perpetuity_cashflow / (self.rate - self.perpetuity_growth_rate),
            CompoundingType::Continuous => {
                self.perpetuity_cashflow * (-self.perpetuity_growth_rate).exp() / ((self.rate - self.perpetuity_growth_rate).exp() - 1.0)
            }
        }
    }

    pub fn net_present_value(&self, initial_cashflow: f64) -> f64 {
        -initial_cashflow + self.present_value()
    }

    pub fn future_value(&self, final_time: f64) -> f64 {
        match self.compounding_type {
            CompoundingType::Periodic => self.present_value() * (1.0 + self.rate).powf(final_time),
            CompoundingType::Continuous => self.present_value() * (self.rate * final_time).exp(),
        }
    }
}

struct Annuity {
    annuity_cashflow: f64,
    rate: f64,
    final_time: f64,
    annuity_growth_rate: f64,
    compounding_type: CompoundingType,
}

impl Annuity {
    pub fn new(
        annuity_cashflow: f64,
        rate: f64,
        final_time: f64,
        annuity_growth_rate: f64,
        compounding_type: CompoundingType,
    ) -> Annuity {
        if rate <= annuity_growth_rate {
            panic!("The rate cannot be smaller than the annuity growth rate.");
        }
        Annuity {
            annuity_cashflow,
            rate,
            final_time,
            annuity_growth_rate,
            compounding_type,
        }
    }

    pub fn present_value(&self) -> f64 {
        match self.compounding_type {
            CompoundingType::Periodic => {
                let pv = self.annuity_cashflow / (self.rate - self.annuity_growth_rate) * (1.0 - ((1.0 + self.annuity_growth_rate) / (1.0 + self.rate)).powf(self.final_time));
                println!("Annuity Present Value: {}", pv); // Print statement
                pv
            },
            CompoundingType::Continuous => {
                let pv = (self.annuity_cashflow * E.powf(-self.annuity_growth_rate) / (E.powf(self.rate - self.annuity_growth_rate) - 1.0)) * (1.0 - E.powf((self.annuity_growth_rate - self.rate) * self.final_time));
                println!("Annuity Present Value (Continuous): {}", pv); // Print statement
                pv
            },
        }
    }

    pub fn net_present_value(&self, initial_cashflow: f64) -> f64 {
        -initial_cashflow + self.present_value()
    }

    pub fn future_value(&self) -> f64 {
        match self.compounding_type {
            CompoundingType::Periodic => self.present_value() * (1.0 + self.rate).powf(self.final_time),
            CompoundingType::Continuous => self.present_value() * (self.rate * self.final_time).exp(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_array_len() {
        let array1 = vec![1, 2, 3];
        let array2 = vec![4, 5, 6];
        compare_array_len(&array1, &array2, "array1", "array2");
    }

    #[test]
    #[should_panic]
    fn test_compare_array_len_panic() {
        let array1 = vec![1, 2, 3];
        let array2 = vec![4, 5];
        compare_array_len(&array1, &array2, "array1", "array2");
    }

    #[test]
    fn test_present_value() {
        let cashflow = vec![100.0, 200.0, 300.0];
        let time_array = vec![1.0, 2.0, 3.0];
        let rate = 0.05;
        let pv = present_value(cashflow, time_array, rate, CompoundingType::Continuous, 1);
        assert!((pv - 545.171).abs() < 0.001);
    }

    #[test]
    fn test_net_present_value() {
        let initial_cashflow = 100.0;
        let cashflow = vec![100.0, 200.0, 300.0];
        let time_array = vec![1.0, 2.0, 3.0];
        let rate = 0.05;
        let npv = net_present_value(initial_cashflow, cashflow, time_array, rate, CompoundingType::Continuous, 1);
        assert!((npv - 445.171).abs() < 0.001);
    }

    #[test]
    fn test_future_value() {
        let current_value = 100.0;
        let rate = 0.05;
        let time = 3.0;
        let fv = future_value(current_value, rate, time, CompoundingType::Continuous, 1);
        assert!((fv - 115.927).abs() < 0.001);
    }

    #[test]
    fn test_annuity_present_value() {
        let annuity = Annuity::new(100.0, 0.05, 10.0, 0.0, CompoundingType::Periodic);
        let pv = annuity.present_value();
        assert!((pv - 925.927).abs() < 0.001);
    }

    #[test]
    fn test_perpetuity_present_value() {
        let perpetuity = Perpetuity::new(100.0, 0.05, 0.0, CompoundingType::Periodic);
        let pv = perpetuity.present_value();
        assert!((pv - 2000.0).abs() < 0.001);
    }
}
