pub mod binomial_models;
pub mod trinomial_models;


pub trait LatticeModel {
    /// # Description
    /// Fair value of an instrument with European exercise style.
    fn european(&self) -> f64;

    /// # Description
    /// Fair value of an instrument with American exercise style.
    fn american(&self) -> f64;

    /// # Description
    /// Fair value of an instrument with Bermudan exercise style.
    fn bermudan(&self, exercise_time_steps: Vec<bool>) -> f64;
}