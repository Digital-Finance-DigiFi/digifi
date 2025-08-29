use ndarray::Array1;
use crate::error::DigiFiError;
use crate::lattice_models::LatticeModel;
use crate::financial_instruments::Payoff;


/// Binomial tree with the defined parameters presented as an array of layers.
/// 
/// # Input
/// - `s_0`: Starting value
/// - `u`: Upward movement factor, must be positive
/// - `d`: Downward movement factor, must be positive
/// - `n_steps`: Number of steps in the tree
/// 
/// # Output
/// - List of layers with node values at each step
/// 
/// # Errors
/// - Returns an error if the value of u or d is non-positive.
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::lattice_models::binomial_tree_nodes;
///
/// let tree: Vec<Array1<f64>> = binomial_tree_nodes(10.0, 1.2, 0.9, 2).unwrap();
///
/// assert!((&tree[0] - Array1::from_vec(vec![10.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((&tree[1] - Array1::from_vec(vec![9.0, 12.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((&tree[2] - Array1::from_vec(vec![8.1, 10.8, 14.4])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub fn binomial_tree_nodes(s_0: f64, u: f64, d: f64, n_steps: usize) -> Result<Vec<Array1<f64>>, DigiFiError> {
    if (u <= 0.0) || (d <= 0.0) {
        return Err(DigiFiError::ParameterConstraint {
            title: "Binomial Tree Nodes".to_owned(),
            constraint: "The arguments `u` and `d` must be positive multiplicative factors of the binomial model.".to_owned(),
        });
    }
    let mut binomial_tree: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
    for layer in 0..(n_steps as i32 + 1) {
        let mut current_layer: Vec<f64> = Vec::<f64>::new();
        for i in 0..(layer + 1) {
            let node: f64 = s_0 * u.powi(i) * d.powi(layer-i);
            current_layer.push(node);
        }
        binomial_tree.push(Array1::from_vec(current_layer));
    }
    Ok(binomial_tree)
}


/// General binomial model with custom payoff. It constructs a binomial tree with given parameters;
/// calculates the value at each node considering the custom payoff and probability; 
/// and aggregates these values to determine the final fair value of the option.
/// 
/// Note: The function assumes that there is a payoff at the final time step.
/// 
/// Note: This function does not discount future cashflows.
/// 
/// # Input
/// - `payoff_object`: Custom payoff object defining the payoff at each node
/// - `s_0`: Starting value
/// - `u`: Upward movement factor, must be positive
/// - `d`: Downward movement factor, must be positive
/// - `p_u`: Probability of an upward movement, must be in \[0,1\]
/// - `n_steps`: Number of steps in the tree
/// - `exercise_time_steps`: A vector indicating whether there's a payoff at each time step. Defaults to payoff at every step if None
/// 
/// # Output
/// - The fair value calculated by the binomial model
/// 
/// # Errors
/// - Returns an error if the value of either `u` or `d` is negative.
/// - Returns an error if the value of `p_u` is not in the range \[0,1\].
/// - Returns an error if the length of `exercise_time_steps` is not of length `n_steps`.
///
/// # Examples
///
/// 1. Pricing European Long Call option:
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::lattice_models::binomial_model;
/// use digifi::financial_instruments::LongCall;
///
/// let long_call: LongCall = LongCall { k: 11.0, cost: 0.0 };
/// let predicted_value: f64 = binomial_model(&long_call, 10.0, 1.2, 0.9, 0.5, 2, Some(vec![false, false])).unwrap();
///
/// let analytic_solution: f64 = 0.5 * (0.5*0.0 + 0.5*3.4) + 0.5 * (0.5*0.0 + 0.5*0.0);
/// assert!((predicted_value - analytic_solution).abs() < TEST_ACCURACY);
/// ```
///
/// 2. Pricing European Long Straddle option:
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::lattice_models::binomial_model;
/// use digifi::financial_instruments::Straddle;
///
/// let straddle: Straddle = Straddle { k: 11.0, cost: 0.0 };
/// let predicted_value: f64 = binomial_model(&straddle, 10.0, 1.2, 0.9, 0.5, 2, Some(vec![false, false])).unwrap();
///
/// let analytic_solution: f64 = 0.5 * (0.5*3.4 + 0.5*0.2) + 0.5 * (0.5*0.2 + 0.5*2.9);
/// assert!((predicted_value - analytic_solution).abs() < TEST_ACCURACY);
/// ```
pub fn binomial_model(payoff_object: &dyn Payoff, s_0: f64, u: f64, d: f64, p_u: f64, n_steps: usize, exercise_time_steps: Option<Vec<bool>>) -> Result<f64, DigiFiError> {
    let error_title: String = String::from("Binomial Model");
    // Data validation
    payoff_object.validate_payoff(5)?;
    if (u < 0.0) || (d < 0.0) {
        return Err(DigiFiError::ParameterConstraint {
            title: error_title,
            constraint: "The arguments `u` and `d` must be non-negative.".to_owned(),
        });
    }
    if (p_u <= 0.0) || (1.0 <= p_u) {
        return Err(DigiFiError::ParameterConstraint {
            title: error_title,
            constraint: "The argument `p_u` must be a defined over a range `[0,1]`.".to_owned(),
        });
    }
    let exercise_time_steps_: Vec<bool>;
    match exercise_time_steps {
        Some(exercise_time_steps_vec) => {
            if exercise_time_steps_vec.len() != n_steps {
                return Err(DigiFiError::ParameterConstraint {
                    title: error_title,
                    constraint: "The argument `exercise_time_steps` should be of length `n_steps`.".to_owned(),
                });
            }
            exercise_time_steps_ = exercise_time_steps_vec
        },
        None => { exercise_time_steps_ = vec![true; n_steps]; },
    }
    // Binomial model
    let mut binomial_tree: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
    let mut layer: Vec<f64> = Vec::<f64>::new();
    // Final layer
    for i in 0..(n_steps as i32 + 1) {
        let value: f64 = payoff_object.payoff(&Array1::from_vec(vec![s_0 * u.powi(i) * d.powi(n_steps as i32 - i)]))[0];
        layer.push(value);
    }
    binomial_tree.push(Array1::from_vec(layer));
    // Layers before the final layer
    for j in (0..n_steps).rev() {
        let mut layer: Vec<f64> = Vec::<f64>::new();
        for i in 0..(j+1) {
            let value: f64 = p_u * binomial_tree[binomial_tree.len()-1][i+1] + (1.0-p_u) * binomial_tree[binomial_tree.len()-1][i];
            if exercise_time_steps_[j] {
                let exercise: f64 = payoff_object.payoff(&Array1::from_vec(vec![s_0 * u.powi(i as i32) * d.powi((j-i) as i32)]))[0];
                layer.push(value.max(exercise));
            } else {
                layer.push(value);
            }
        }
        binomial_tree.push(Array1::from_vec(layer));
    }
    Ok(binomial_tree[binomial_tree.len()-1][0])
}


/// Binomial models that are scaled to emulate Brownian motion. This model uses a binomial lattice
/// approach to approximate the continuous path of Brownian motion, specifically for option pricing.
/// 
/// This model calculates the up (`u`) and down (`d`) factors using the volatility and time step (`dt`),
/// ensuring the binomial model aligns with the log-normal distribution of stock prices in the Black-Scholes model.
/// Depending on the payoff type, it sets the appropriate payoff function. For more detailed theory,
/// refer to the Cox-Ross-Rubinstein model and its alignment with the Black-Scholes model in financial literature.
/// 
/// This technique is rooted in the Cox-Ross-Rubinstein (CRR) model, adapting it to mirror the properties of Brownian motion.
/// 
/// # Links:
/// - Wikipedia: <https://en.wikipedia.org/wiki/Binomial_options_pricing_model>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::lattice_models::{LatticeModel, BrownianMotionBinomialModel};
/// use digifi::financial_instruments::LongCall;
///
/// let long_call: LongCall = LongCall { k: 11.0, cost: 0.0 };
/// let bmbm: BrownianMotionBinomialModel = BrownianMotionBinomialModel::build(Box::new(long_call), 10.0, 1.0, 0.02, 0.2, 0.0, 1_000).unwrap();
/// let predicted_value: f64 = bmbm.european().unwrap();
///
/// // Test accuracy depends on the conversion between Brownian-scaled binomial model and Black-Scholes analytic solution
/// assert!((predicted_value - 0.49438669572304805).abs() < 1_000_000.0*TEST_ACCURACY);
/// ```
pub struct BrownianMotionBinomialModel {
    /// Payoff function
    payoff_object: Box<dyn Payoff>,
    /// Initial underlying asset price
    s_0: f64,
    /// Time to maturity
    time_to_maturity: f64,
    /// Risk-free interest rate
    r: f64,
    /// Volatility of the underlying asset
    _sigma: f64,
    /// Dividend yield
    q: f64,
    /// Number of steps in the binomial model
    n_steps: usize,
    /// Time difference between two consequtive time steps
    dt: f64,
    /// Magnitude of step up
    u: f64,
    /// Magnitude of step down
    d: f64,
}

impl BrownianMotionBinomialModel {
    /// Creates a new `BrownianMotionBinomialModel` instance.
    /// 
    /// # Input
    /// - `payoff_object`: Payoff function
    /// - `s_0`: Initial underlying asset price
    /// - `time_to_maturity`: Time to maturity
    /// - `r`: Risk-free interest rate
    /// - `sigma`: Volatility of the underlying asset
    /// - `q`: Dividend yield
    /// - `n_steps`: Number of steps in the binomial model
    pub fn build(payoff_object: Box<dyn Payoff>, s_0: f64, time_to_maturity: f64, r: f64, sigma: f64, q: f64, n_steps: usize) -> Result<Self, DigiFiError> {
        payoff_object.validate_payoff(5)?;
        let dt: f64 = time_to_maturity / (n_steps as f64);
        Ok(BrownianMotionBinomialModel {
            payoff_object, s_0, time_to_maturity, r, _sigma: sigma, q, n_steps, dt, u: (sigma*dt.sqrt()).exp(), d: (-sigma*dt.sqrt()).exp(),
        })
    }
}

impl LatticeModel for BrownianMotionBinomialModel {
    /// Binomial model that computes the payoffs for each path and computes the weighted average of paths based on probability.
    /// 
    /// # Output
    /// - The present value of an instrument with the European exercise style
    fn european(&self) -> Result<f64, DigiFiError> {
        let p_u: f64 = ((-self.q*self.dt).exp() - (-self.r*self.dt).exp()*self.d) / (self.u - self.d);
        let p_d: f64 = (-self.r*self.dt).exp() - p_u;
        let mut binomial_tree: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
        let mut layer: Vec<f64> = Vec::<f64>::new();
        for i in 0..(self.n_steps as i32 + 1) {
            layer.push(self.payoff_object.payoff(&Array1::from_vec(vec![self.s_0 * self.u.powi(i) * self.d.powi(self.n_steps as i32 - i)]))[0]);
        }
        binomial_tree.push(Array1::from_vec(layer));
        for j in (0..self.n_steps).rev() {
            let mut layer: Vec<f64> = Vec::<f64>::new();
            for i in 0..(j + 1) {
                let value: f64 = p_u*binomial_tree[binomial_tree.len()-1][i+1] + p_d*binomial_tree[binomial_tree.len()-1][i];
                layer.push(value);
            }
            binomial_tree.push(Array1::from_vec(layer));
        }
        Ok(binomial_tree[binomial_tree.len()-1][0] * (-self.r * self.time_to_maturity).exp())
    }

    /// Binomial model that computes the payoffs for each node in the binomial tree to determine the initial payoff value.
    /// 
    /// # Output
    /// - The present value of an instrument with the American exercise style
    fn american(&self) -> Result<f64, DigiFiError> {
        let mut exercise_time_steps: Vec<bool> = Vec::<bool>::new();
        for _ in 0..self.n_steps {
            exercise_time_steps.push(true);
        }
        self.bermudan(&exercise_time_steps)
    }

    /// Binomial model that computes the payoffs for each node in the binomial tree to determine the initial payoff value.
    /// 
    /// # Input
    /// - `exercise_time_steps`: Indicators for exercise opportunity at each time step
    /// 
    /// # Output
    /// - The present value of an instrument with the Bermudan exercise style
    /// 
    /// # Errors
    /// - Returns an error if the length of `exercise_time_steps` is not of length `n_steps`
    fn bermudan(&self, exercise_time_steps: &Vec<bool>) -> Result<f64, DigiFiError> {
        if exercise_time_steps.len() != self.n_steps {
            return Err(DigiFiError::ParameterConstraint {
                title: "Brownian Motion Binomial Model".to_owned(),
                constraint: "The argument `exercise_time_steps` should be of length `n_steps`.".to_owned(),
            });
        }
        let p_u: f64 = ((-self.q*self.dt).exp() - (-self.r*self.dt).exp()*self.d) / (self.u - self.d);
        let p_d: f64 = (-self.r*self.dt).exp() - p_u;
        let mut binomial_tree: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
        let mut layer: Vec<f64> = Vec::<f64>::new();
        for i in 0..(self.n_steps as i32 + 1) {
            layer.push(self.payoff_object.payoff(&Array1::from_vec(vec![self.s_0 * self.u.powi(i) * self.d.powi(self.n_steps as i32 - i)]))[0]);
        }
        binomial_tree.push(Array1::from_vec(layer));
        for j in (0..self.n_steps).rev() {
            let mut layer: Vec<f64> = Vec::<f64>::new();
            for i in 0..(j + 1) {
                let value: f64 = p_u*binomial_tree[binomial_tree.len()-1][i+1] + p_d*binomial_tree[binomial_tree.len()-1][i];
                if exercise_time_steps[self.n_steps - j] {
                    let exercise: f64 = self.payoff_object.payoff(&Array1::from_vec(vec![self.s_0 * (self.u.powi(i as i32)) * (self.d.powi((j-i) as i32))]))[0];
                    layer.push(value.max(exercise));
                } else {
                    layer.push(value);
                }
            }
            binomial_tree.push(Array1::from_vec(layer));
        }
        Ok(binomial_tree[binomial_tree.len()-1][0] * (-self.r * self.time_to_maturity).exp())
    }
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_binomial_tree_nodes() -> () {
        use crate::lattice_models::binomial_models::binomial_tree_nodes;
        let tree: Vec<Array1<f64>> = binomial_tree_nodes(10.0, 1.2, 0.9, 2).unwrap();
        assert!((&tree[0] - Array1::from_vec(vec![10.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((&tree[1] - Array1::from_vec(vec![9.0, 12.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((&tree[2] - Array1::from_vec(vec![8.1, 10.8, 14.4])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_binomial_model_1() -> () {
        use crate::lattice_models::binomial_models::binomial_model;
        use crate::financial_instruments::LongCall;
        let long_call: LongCall = LongCall { k: 11.0, cost: 0.0 };
        let predicted_value: f64 = binomial_model(&long_call, 10.0, 1.2, 0.9, 0.5, 2, Some(vec![false, false])).unwrap();
        let analytic_solution: f64 = 0.5 * (0.5*0.0 + 0.5*3.4) + 0.5 * (0.5*0.0 + 0.5*0.0);
        assert!((predicted_value - analytic_solution).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_binomial_model_2() -> () {
        use crate::lattice_models::binomial_models::binomial_model;
        use crate::financial_instruments::Straddle;
        let straddle: Straddle = Straddle { k: 11.0, cost: 0.0 };
        let predicted_value: f64 = binomial_model(&straddle, 10.0, 1.2, 0.9, 0.5, 2, Some(vec![false, false])).unwrap();
        let analytic_solution: f64 = 0.5 * (0.5*3.4 + 0.5*0.2) + 0.5 * (0.5*0.2 + 0.5*2.9);
        assert!((predicted_value - analytic_solution).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_brownian_motion_binomial_model() -> () {
        use crate::lattice_models::binomial_models::BrownianMotionBinomialModel;
        use crate::lattice_models::LatticeModel;
        use crate::financial_instruments::LongCall;
        let long_call: LongCall = LongCall { k: 11.0, cost: 0.0 };
        let bmbm: BrownianMotionBinomialModel = BrownianMotionBinomialModel::build(Box::new(long_call), 10.0, 1.0, 0.02, 0.2, 0.0, 1_000).unwrap();
        let predicted_value: f64 = bmbm.european().unwrap();
        // Test accuracy depends on the conversion between Brownian-scaled binomial model and Black-Scholes analytic solution
        assert!((predicted_value - 0.49438669572304805).abs() < 1_000_000.0*TEST_ACCURACY);
    }
}