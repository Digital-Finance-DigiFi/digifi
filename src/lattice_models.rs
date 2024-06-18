pub mod binomial_models;
pub mod trinomial_models;


pub trait LatticeModel {
    /// # Description
    /// Fair value of European option.
    fn european_option(&self) -> f64;

    /// # Description
    /// Fair value of American Option.
    fn american_option(&self) -> f64;

    /// # Description
    /// Fair value of Bermudan option.
    fn bermudan_option(&self) -> f64;
}