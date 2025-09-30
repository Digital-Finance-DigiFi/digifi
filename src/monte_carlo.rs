//! # Monte Carlo Methods
//! 
//! Contains functionality for running Monte Carlo simulations for pricing instruments with payoff functions.


use ndarray::Array1;
use crate::error::DigiFiError;
use crate::financial_instruments::Payoff;
use crate::stochastic_processes::StochasticProcess;


/// Simulates the paths of the stochastic process and uses these simulations to price the payoff.
/// 
/// Note: To correctly price the payoff, the paths produced by the stochastic process must be in terms of future values of the asset
/// (i.e., they should be scaled by the discount rate or risk-free rate). If the paths are not defined as the future value, the value of `r` should be set to `0`.
/// 
/// # Input
/// - `stochastic_process`: Stochastic process that produces the paths to which the payoff applies
/// - `payoff_object`: Custom payoff object with the payoff function
/// - `r`: Discount rate
/// - `exercise_time_steps`: Time steps in the siulated paths when the payoff applies (Note that payoff will be always applied at the last time step)
/// 
/// # Output
/// - Expected discounted payoff or premium (e.g., option premium)
/// 
/// # Errors
/// - Returns an error if the length of `exercise_time_steps` is not of length `n_steps`.
/// 
/// # Examples
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::financial_instruments::{BlackScholesType, black_scholes_formula, LongCall};
/// use digifi::monte_carlo::monte_carlo_simulation;
/// use digifi::stochastic_processes::standard_stochastic_models::GeometricBrownianMotion;
/// 
/// // Parameter definition
/// let n_paths: usize = 1_000;
/// let n_steps: usize = 200;
/// let rf: f64 = 0.02;
/// 
/// // GBM definition (`mu` is set to the risk-free rate to account for the future value of assets's price)
/// let gbm: GeometricBrownianMotion = GeometricBrownianMotion::new(rf, 0.2, n_paths, n_steps, 1.0, 10.0);
/// let long_call: LongCall = LongCall { k: 11.0, cost: 0.0 };
/// 
/// // Predicted value (Monte-Carlo simulation)
/// let predicted_value: f64 = monte_carlo_simulation(&gbm, &long_call, rf, &Some(vec![false; n_steps])).unwrap();
/// 
/// // Theoretical value (Black-Scholes formula)
/// let theoretical_value: f64 = black_scholes_formula(10.0, 11.0, 0.2, 1.0, 0.02, 0.0, &BlackScholesType::Call).unwrap();
///
/// assert!((predicted_value - theoretical_value).abs() < 10_000_000.0 * TEST_ACCURACY);
/// ```
pub fn monte_carlo_simulation(stochastic_process: &dyn StochasticProcess, payoff_object: &dyn Payoff, r: f64, exercise_time_steps: &Option<Vec<bool>>) -> Result<f64, DigiFiError> {
    // Data validation
    let n_steps: usize = stochastic_process.get_n_steps();
    let dt: f64 = stochastic_process.get_t_f() / (n_steps as f64);
    let exercise_time_steps: &Vec<bool> = match exercise_time_steps {
        Some(exercise_time_steps_vec) => {
            if exercise_time_steps_vec.len() != n_steps {
                return Err(DigiFiError::WrongLength { title: "Monte-Carlo Simulation".to_owned(), arg: "exercise time steps".to_owned(), len: n_steps, });
            }
            exercise_time_steps_vec
        },
        None => &vec![true; n_steps],
    };
    // Stochastic model
    let stochastic_paths: Vec<Array1<f64>> = stochastic_process.get_paths()?;
    let sum_of_payoffs: f64 = stochastic_paths.iter().fold(0.0, |total, path| {
        let value: f64 = (0..(path.len() - 1)).rev().fold(payoff_object.payoff(path[path.len() - 1]), |value, i| {
            let mut value: f64 = value * (-dt * r).exp();
            if exercise_time_steps[i] {
                let exercise: f64 = payoff_object.payoff(path[i]);
                value = value.max(exercise);
            }
            value
        } );
        total + value
    } );
    Ok(sum_of_payoffs / stochastic_paths.len() as f64)
}


#[cfg(test)]
mod tests {
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_stochastic_model() -> () {
        use crate::financial_instruments::derivatives::{BlackScholesType, black_scholes_formula};
        use crate::monte_carlo::monte_carlo_simulation;
        use crate::financial_instruments::LongCall;
        use crate::stochastic_processes::standard_stochastic_models::GeometricBrownianMotion;
        // Parameter definition
        let n_paths: usize = 1_000;
        let n_steps: usize = 200;
        let rf: f64 = 0.02;
        // GBM definition (`mu` is set to the risk-free rate to account for the future value of assets's price)
        let gbm: GeometricBrownianMotion = GeometricBrownianMotion::new(rf, 0.2, n_paths, n_steps, 1.0, 10.0);
        let long_call: LongCall = LongCall { k: 11.0, cost: 0.0 };
        // Predicted value (Monte-Carlo simulation)
        let predicted_value: f64 = monte_carlo_simulation(
            &gbm, &long_call, rf, &Some(vec![false; n_steps])
        ).unwrap();
        // Theoretical value (Black-Scholes formula)
        let theoretical_value: f64 = black_scholes_formula(10.0, 11.0, 0.2, 1.0, 0.02, 0.0, &BlackScholesType::Call).unwrap();
        assert!((predicted_value - theoretical_value).abs() < 10_000_000.0 * TEST_ACCURACY);
    }
}