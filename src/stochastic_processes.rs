pub mod standard_stochastic_models;


use ndarray::Array1;


pub trait StochasticProcess {
    /// # Description
    /// Updates the number of paths that the stochastic process will generate.
    /// 
    /// # Input
    /// - n_paths: Number of paths to generate
    fn update_n_paths(&mut self, n_paths: usize) -> ();

    /// # Description
    /// Expected path, E[S], of the stochastic process.
    fn get_expectations(&self) -> Array1<f64>;

    /// # Description
    /// Paths, S, of the stochastic process.
    fn get_paths(&self) -> Vec<Array1<f64>>;
}