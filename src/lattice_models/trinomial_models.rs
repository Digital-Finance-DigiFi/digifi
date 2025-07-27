use ndarray::Array1;
use crate::error::DigiFiError;
use crate::financial_instruments::Payoff;
use crate::lattice_models::LatticeModel;


/// # Description
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
/// - Returns an error if the value of `u` or `d` is non-positive.
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::lattice_models::trinomial_tree_nodes;
///
/// let tree: Vec<Array1<f64>> = trinomial_tree_nodes(10.0, 1.2, 0.9, 2).unwrap();
/// // Sideways movement factor
/// let s: f64 = 1.0392304845;
///
/// assert!((&tree[0] - Array1::from_vec(vec![10.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((&tree[1] - Array1::from_vec(vec![9.0, 10.0*s, 12.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// assert!((&tree[2] - Array1::from_vec(vec![8.1, 9.0*s, 10.0*s*s, 12.0*s, 14.4])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub fn trinomial_tree_nodes(s_0: f64, u: f64, d: f64, n_steps: usize) -> Result<Vec<Array1<f64>>, DigiFiError> {
    if (u <= 0.0) || (d <= 0.0) {
        return Err(DigiFiError::ParameterConstraint { title: "Trinomial Tree Nodes".to_owned(), constraint: "The arguments `u` and `d` must be positive multiplicative factors of the trinomial model.".to_owned(), });
    }
    let s: f64 = (u * d).sqrt();
    let mut trinomial_tree: Vec<Array1<f64>> = vec![Array1::from_vec(vec![s_0])];
    for _ in 1..(n_steps as i32 + 1) {
        let current_layer: Array1<f64> = s * &trinomial_tree[trinomial_tree.len()-1];
        let u_node: f64 = u * trinomial_tree[trinomial_tree.len()-1][current_layer.len()-1];
        let d_node: f64 = d * trinomial_tree[trinomial_tree.len()-1][0];
        let mut current_layer = current_layer.to_vec();
        current_layer.insert(0, d_node);
        current_layer.push(u_node);
        trinomial_tree.push(Array1::from_vec(current_layer));
    }
    Ok(trinomial_tree)
}


/// # Description
/// General trinomial model with custom payoff.
/// 
/// The function assumes that there is a payoff at the final time step.
/// 
/// This function does not discount future cashflows.
/// 
/// # Input
/// - `payoff_object`: Custom payoff object defining the payoff at each node
/// - `s_0`: Initial underlying asset value
/// - `u`: Upward movement factor
/// - `d`: Downward movement factor
/// - `p_u`: Probability of an upward movement
/// - `p_d`: Probability of a downward movement
/// - `n_steps`: Number of steps in the model
/// - `exercise_time_steps`: List of booleans indicating if there's a payoff at each step
/// 
/// # Output
/// - Fair value calculated by the trinomial model
/// 
/// # Errors
/// - Returns an error if the value of either `u` or `d` is negative.
/// - Returns an error if the value of `p_u` or `p_d` is not in the range \[0,1\].
/// - Returns an error if the sum of probabilities `p_u` and `p_d` exceeds `1`.
/// - Returns an error if the length of exercise_time_steps is not of length n_steps.
///
/// # Examples
///
/// 1. Pricing European Long Call option:
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::lattice_models::trinomial_model;
/// use digifi::financial_instruments::LongCall;
///
/// let long_call: LongCall = LongCall { k: 11.0, cost: 0.0 };
/// let fair_value: f64 = trinomial_model(&long_call, 10.0, 1.2, 0.9, 0.25, 0.25, 2, Some(vec![false, false])).unwrap();
/// // Sideways movement factor 
/// let s: f64 = 1.0392304845;
///
/// let analytic_solution: f64 = 0.25*(0.25*3.4 + 0.5*(12.0*s - 11.0) + 0.25*0.0) + 0.5*(0.25*(12.0*s - 11.0) + 0.5*0.0 + 0.25*0.0) + 0.25*(0.25*0.0 + 0.5*0.0 + 0.25*0.0);
/// assert!((fair_value - analytic_solution).abs() < TEST_ACCURACY);
/// ```
///
/// 2. Pricing European Long Straddle option:
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::lattice_models::trinomial_model;
/// use digifi::financial_instruments::Straddle;
///
/// let straddle: Straddle = Straddle { k: 11.0, cost: 0.0 };
/// let fair_value: f64 = trinomial_model(&straddle, 10.0, 1.2, 0.9, 0.25, 0.25, 2, Some(vec![false, false])).unwrap();
/// // Sideways movement factor 
/// let s: f64 = 1.0392304845;
///
/// let analytic_solution: f64 = 0.25*(0.25*3.4 + 0.5*(12.0*s - 11.0) + 0.25*(11.0 - 10.0*s*s)) + 0.5*(0.25*(12.0*s - 11.0) + 0.5*(11.0 - 10.0*s*s) + 0.25*(11.0 - 9.0*s)) + 0.25*(0.25*(11.0 - 10.0*s*s) + 0.5*(11.0 - 9.0*s) + 0.25*2.9);
/// assert!((fair_value - analytic_solution).abs() < TEST_ACCURACY);
/// ```
pub fn trinomial_model(payoff_object: &dyn Payoff, s_0: f64, u: f64, d: f64, p_u: f64, p_d: f64, n_steps: usize, exercise_time_steps: Option<Vec<bool>>) -> Result<f64, DigiFiError> {
    let error_title: String = String::from("Trinomial Model");
    // Data validation
    payoff_object.validate_payoff(5)?;
    if (u <= 0.0) || (d <= 0.0) {
        return Err(DigiFiError::ParameterConstraint { title: error_title.clone(), constraint: "The arguments `u` and `d` must be positive multiplicative factors of the trinomial model.".to_owned(), });
    }
    if (p_u <= 0.0) || (1.0 <= p_u) || (p_d <= 0.0) || (1.0 <= p_d) {
        return Err(DigiFiError::ParameterConstraint { title: error_title.clone(), constraint: "The arguments `p_u` and `p_d` must be a defined over a range `[0,1]`.".to_owned(), });
    }
    if 1.0 < (p_u + p_d) {
        return Err(DigiFiError::ParameterConstraint { title: error_title.clone(), constraint: "The probabilities `p_u`, `p_d` and `1 - p_u - p_d` must add up to `1`.".to_owned(), });
    }
    let p_s: f64 = 1.0 - p_u - p_d;
    let exercise_time_steps_: Vec<bool>;
    match exercise_time_steps {
        Some(exercise_time_steps_vec) => {
            if exercise_time_steps_vec.len() != n_steps {
                return Err(DigiFiError::ParameterConstraint { title: error_title.clone(), constraint: "The argument `exercise_time_steps` should be of length `n_steps`.".to_owned(), });
            }
            exercise_time_steps_ = exercise_time_steps_vec
        },
        None => { exercise_time_steps_ = vec![true; n_steps]; },
    }
    // Trinomial model
    let mut trinomial_tree: Vec<Array1<f64>> = trinomial_tree_nodes(s_0, u, d, n_steps)?;
    let final_payoff_step: usize = trinomial_tree.len() - 1;
    trinomial_tree[final_payoff_step] = payoff_object.payoff(&trinomial_tree[final_payoff_step]);
    for i in (0..(trinomial_tree.len()-1)).rev() {
        let mut layer: Vec<f64> = Vec::<f64>::new();
        for j in 1..(2 * (i+1)) {
            let value: f64 = p_d * trinomial_tree[i+1][j-1] + p_s * trinomial_tree[i+1][j] + p_u * trinomial_tree[i+1][j+1];
            if exercise_time_steps_[i] {
                let exercise: f64 = payoff_object.payoff(&Array1::from_vec(vec![trinomial_tree[i][layer.len()-1]]))[0];
                layer.push(value.max(exercise));
            } else {
                layer.push(value);
            }
        }
        trinomial_tree[i] = Array1::from_vec(layer);
    }
    Ok(trinomial_tree[0][0])
}


/// # Description
/// Trinomial models that are scaled to emulate Brownian motion.
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::lattice_models::{LatticeModel, BrownianMotionTrinomialModel};
/// use digifi::financial_instruments::LongCall;
///
/// let long_call: LongCall = LongCall { k: 11.0, cost: 0.0 };
/// let bmtm: BrownianMotionTrinomialModel = BrownianMotionTrinomialModel::new(Box::new(long_call), 10.0, 1.0, 0.02, 0.2, 0.0, 1_000).unwrap();
/// let predicted_value = bmtm.european().unwrap();
///
/// // Test accuracy depends on the conversion between Brownian-scaled binomial model and Black-Scholes analytic solution
/// assert!((predicted_value - 0.49438669572304805).abs() < 10_000.0*TEST_ACCURACY);
/// ```
pub struct BrownianMotionTrinomialModel {
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
    _q: f64,
    /// Number of steps in the binomial model
    n_steps: usize,
    /// Time difference between two consequtive time steps
    _dt: f64,
    /// Magnitude of step up
    u: f64,
    /// Magnitude of step down
    d: f64,
    /// Probability of a move up
    p_u: f64,
    /// Probability of a move down
    p_d: f64,
}

impl BrownianMotionTrinomialModel {
    /// # Description
    /// Creates a new `BrownianMotionTrinomialModel` instance.
    /// 
    /// # Input
    /// - `payoff_object`: Payoff function
    /// - `s_0`: Initial underlying asset price
    /// - `time_to_maturity`: Time to maturity
    /// - `r`: Risk-free interest rate
    /// - `sigma`: Volatility of the underlying asset
    /// - `q`: Dividend yield
    /// - `n_steps`: Number of steps in the trinomial model
    ///
    /// # Errors
    /// - Returns an error if the condition \\Delta t<\\frac{{\\sigma^{{2}}}}{{(r-q)^{{2}}}} is not satisfied
    pub fn new(payoff_object: Box<dyn Payoff>, s_0: f64, time_to_maturity: f64, r: f64, sigma: f64, q: f64, n_steps: usize) -> Result<Self, DigiFiError> {
        payoff_object.validate_payoff(5)?;
        let dt: f64 = time_to_maturity / (n_steps as f64);
        if (2.0 * sigma.powi(2) / (r-q).powi(2)) <= dt {
            return Err(DigiFiError::ParameterConstraint { title: "Brownian Motion Trinomial Model".to_owned(),
                constraint: "With the given arguments, the condition \\Delta t<\\frac{{\\sigma^{{2}}}}{{(r-q)^{{2}}}} is not satisfied.".to_owned(), });
        }
        let u: f64 = (sigma * (2.0*dt).sqrt()).exp();
        let d: f64 = (-sigma * (2.0*dt).sqrt()).exp();
        let p_u: f64 = (((dt*(r-q)/2.0).exp() - (-sigma*(dt/2.0).sqrt()).exp()) / ((sigma*(dt/2.0).sqrt()).exp() - (-sigma*(dt/2.0).sqrt()).exp())).powi(2);
        let p_d: f64 = (((sigma*(dt/2.0).sqrt()).exp() - ((dt*(r-q)/2.0).exp())) / ((sigma*(dt/2.0).sqrt()).exp() - (-sigma*(dt/2.0).sqrt()).exp())).powi(2);
        Ok(BrownianMotionTrinomialModel { payoff_object, s_0, time_to_maturity, r, _sigma: sigma, _q: q, n_steps, _dt: dt, u, d, p_u, p_d })
    }
}

impl LatticeModel for BrownianMotionTrinomialModel {
    /// # Description
    /// Trinomial model that computes the payoffs for each node in the trinomial tree to determine the initial payoff value.
    /// 
    /// # Output
    /// - The present value of an instrument with the European exercise style
    fn european(&self) -> Result<f64, DigiFiError> {
        let exercise_time_steps: Option<Vec<bool>> = Some(vec![false; self.n_steps]);
        Ok((-self.r*self.time_to_maturity).exp() * trinomial_model(self.payoff_object.as_ref(), self.s_0, self.u, self.d, self.p_u, self.p_d, self.n_steps, exercise_time_steps)?)
    }

    /// # Description
    /// Trinomial model that computes the payoffs for each node in the trinomial tree to determine the initial payoff value.
    /// 
    /// # Output
    /// - The present value of an instrument with the American exercise style
    fn american(&self) -> Result<f64, DigiFiError> {
        let exercise_time_steps: Option<Vec<bool>> = Some(vec![true; self.n_steps]);
        Ok((-self.r*self.time_to_maturity).exp() * trinomial_model(self.payoff_object.as_ref(), self.s_0, self.u, self.d, self.p_u, self.p_d, self.n_steps, exercise_time_steps)?)
    }

    /// # Description
    /// Trinomial model that computes the payoffs for each node in the trinomial tree to determine the initial payoff value.
    /// 
    /// # Input
    /// - `exercise_time_steps`: Indicators for exercise opportunity at each timestep
    /// 
    /// # Output
    /// - The present value of an instrument with the Bermudan exercise style
    fn bermudan(&self, exercise_time_steps: &Vec<bool>) -> Result<f64, DigiFiError> {
        Ok((-self.r*self.time_to_maturity).exp() * trinomial_model(self.payoff_object.as_ref(), self.s_0, self.u, self.d, self.p_u, self.p_d, self.n_steps, Some(exercise_time_steps.clone()))?)
    }
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_trinomial_tree_nodes() -> () {
        use crate::lattice_models::trinomial_models::trinomial_tree_nodes;
        let tree: Vec<Array1<f64>> = trinomial_tree_nodes(10.0, 1.2, 0.9, 2).unwrap();
        // Sideways movement factor
        let s: f64 = 1.0392304845;
        assert!((&tree[0] - Array1::from_vec(vec![10.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((&tree[1] - Array1::from_vec(vec![9.0, 10.0*s, 12.0])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
        assert!((&tree[2] - Array1::from_vec(vec![8.1, 9.0*s, 10.0*s*s, 12.0*s, 14.4])).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_trinomial_model_1() -> () {
        use crate::lattice_models::trinomial_models::trinomial_model;
        use crate::financial_instruments::LongCall;
        let long_call: LongCall = LongCall { k: 11.0, cost: 0.0 };
        let predicted_value: f64 = trinomial_model(&long_call, 10.0, 1.2, 0.9, 0.25, 0.25, 2, Some(vec![false, false])).unwrap();
        // Sideways movement factor 
        let s: f64 = 1.0392304845;
        let analytic_solution: f64 = 0.25*(0.25*3.4 + 0.5*(12.0*s - 11.0) + 0.25*0.0) + 0.5*(0.25*(12.0*s - 11.0) + 0.5*0.0 + 0.25*0.0) + 0.25*(0.25*0.0 + 0.5*0.0 + 0.25*0.0);
        assert!((predicted_value - analytic_solution).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_trinomial_model_2() -> () {
        use crate::lattice_models::trinomial_models::trinomial_model;
        use crate::financial_instruments::Straddle;
        let straddle: Straddle = Straddle { k: 11.0, cost: 0.0 };
        let predicted_value: f64 = trinomial_model(&straddle, 10.0, 1.2, 0.9, 0.25, 0.25, 2, Some(vec![false, false])).unwrap();
        // Sideways movement factor 
        let s: f64 = 1.0392304845;
        let analytic_solution: f64 = 0.25*(0.25*3.4 + 0.5*(12.0*s - 11.0) + 0.25*(11.0 - 10.0*s*s)) + 0.5*(0.25*(12.0*s - 11.0) + 0.5*(11.0 - 10.0*s*s) + 0.25*(11.0 - 9.0*s)) + 0.25*(0.25*(11.0 - 10.0*s*s) + 0.5*(11.0 - 9.0*s) + 0.25*2.9);
        assert!((predicted_value - analytic_solution).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_brownian_motion_trinomial_model() -> () {
        use crate::lattice_models::trinomial_models::BrownianMotionTrinomialModel;
        use crate::lattice_models::LatticeModel;
        use crate::financial_instruments::LongCall;
        let long_call: LongCall = LongCall { k: 11.0, cost: 0.0 };
        let bmtm: BrownianMotionTrinomialModel = BrownianMotionTrinomialModel::new(Box::new(long_call), 10.0, 1.0, 0.02, 0.2, 0.0, 1_000).unwrap();
        let predicted_value: f64 = bmtm.european().unwrap();
        // Test accuracy depends on the conversion between Brownian-scaled binomial model and Black-Scholes analytic solution
        assert!((predicted_value - 0.49438669572304805).abs() < 10_000.0*TEST_ACCURACY);
    }
}