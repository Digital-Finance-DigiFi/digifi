//! # Lattice Models
//! 
//! Contains differrent types of binomial and trinomial models for pricing instruments with payoff functions.


// Re-Exports
pub use self::binomial_models::{binomial_tree_nodes, binomial_model, BrownianMotionBinomialModel};
pub use self::trinomial_models::{trinomial_tree_nodes, trinomial_model, BrownianMotionTrinomialModel};


pub mod binomial_models;
pub mod trinomial_models;


use crate::error::DigiFiError;


pub trait LatticeModel {
    /// Fair value of an instrument with European exercise style.
    fn european(&self) -> Result<f64, DigiFiError>;

    /// Fair value of an instrument with American exercise style.
    fn american(&self) -> Result<f64, DigiFiError>;

    /// Fair value of an instrument with Bermudan exercise style.
    fn bermudan(&self, exercise_time_steps: &Vec<bool>) -> Result<f64, DigiFiError>;
}