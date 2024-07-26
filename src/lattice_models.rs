pub mod binomial_models;
pub mod trinomial_models;


use std::io::Error;


pub trait LatticeModel {
    /// # Description
    /// Fair value of an instrument with European exercise style.
    fn european(&self) -> Result<f64, Error>;

    /// # Description
    /// Fair value of an instrument with American exercise style.
    fn american(&self) -> Result<f64, Error>;

    /// # Description
    /// Fair value of an instrument with Bermudan exercise style.
    fn bermudan(&self, exercise_time_steps: Vec<bool>) -> Result<f64, Error>;
}