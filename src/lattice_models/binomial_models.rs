use ndarray::Array1;
// TODO: Test binomial models


/// # Description
/// Binomial tree with the defined parameters presented as an array of layers.
/// 
/// # Input
/// - start_point: Starting value
/// - u: Upward movement factor, must be positive
/// - d: Downward movement factor, must be positive
/// - n_steps: Number of steps in the tree
/// 
/// # Output
/// - List of layers with node values at each step
/// 
/// # panics:
/// - Panics if the value of u or d is non-positive
pub fn binomial_tree_nodes(start_point: f64, u: f64, d: f64, n_steps: usize) -> Vec<Array1<f64>> {
    if (u <= 0.0) || (d <= 0.0) {
        panic!("The arguments u and d must be positive multiplicative factors of the binomial model.");
    }
    let mut binomial_tree: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
    for layer in 0..(n_steps as i32 + 1) {
        let mut current_layer: Vec<f64> = Vec::<f64>::new();
        for i in 0..(layer + 1) {
            let node: f64 = start_point * u.powi(i) * d.powi(layer-i);
            current_layer.push(node);
        }
        binomial_tree.push(Array1::from_vec(current_layer));
    }
    binomial_tree
}


/// # Description
/// General binomial model with custom payoff. It constructs a binomial tree with given parameters; calculates the value at each node considering the custom payoff and probability; 
/// and aggregates these values to determine the final fair value of the option.\n
/// Note: The function assumes that there is a payoff at the final time step.\n
/// Note: This function does not discount future cashflows.
/// 
/// # Input
/// - payoff: Custom payoff object defining the payoff at each node
/// - start_point: Starting value
/// - u: Upward movement factor, must be positive
/// - d: Downward movement factor, must be positive
/// - p_u: Probability of an upward movement, must be in [0,1]
/// - n_steps: Number of steps in the tree
/// - payoff_timesteps: A list indicating whether there's a payoff at each timestep. Defaults to payoff at every step if None
/// 
/// # Output
/// - The fair value calculated by the binomial model
/// 
/// # Panics
/// - Panics if the value of p_u is not in the range [0,1]
/// - Panics if the length of payoff_timesteps does not match the length of n_steps
pub fn binomial_model(u: f64, d: f64, p_u: f64, n_steps: usize, payoff_timesteps: Option<Vec<bool>>) -> f64 {

    0.0
}