// Re-Exports
pub use self::standard_stochastic_models::{
    ArithmeticBrownianMotion, GeometricBrownianMotion, OrnsteinUhlenbeckProcess, BrownianBridge, FellerSquareRootProcess, FSRSimulationMethod,
};
pub use self::stochastic_volatility_models::{ConstantElasticityOfVariance, HestonStochasticVolatility, VarianceGammaProcess};
pub use self::jump_diffusion_models::{MertonJumpDiffusionProcess, KouJumpDiffusionProcess};


pub mod standard_stochastic_models;
pub mod stochastic_volatility_models;
pub mod jump_diffusion_models;
pub mod stochastic_process_generator;


use std::io::Error;
use ndarray::Array1;


pub trait StochasticProcess {

    /// # Description
    /// Updates the number of paths that the stochastic process will generate.
    /// 
    /// # Input
    /// - `n_paths`: Number of paths to generate
    fn update_n_paths(&mut self, n_paths: usize) -> ();

    /// # Description
    /// Expected path, E\[S\], of the stochastic process.
    fn get_expectations(&self) -> Array1<f64>;

    /// # Description
    /// Paths, S, of the stochastic process.
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, Error>;
}