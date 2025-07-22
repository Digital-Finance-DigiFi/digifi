//! # DigiFi
//!
//! `digifi` is a general purpose financial library and framework for financial modelling, portfolio optimization, and asset pricing. The purpose of
//! this library is to provide basic functions and algorithms used in finance, but with enhanced memory safety, minimum number of dependencies and
//! low computational requirements. `digifi` is composed of the following modules:
//!
//! - `utilities`: Generic utility functions including time-value calculators, mathematical functions, loss functions and numerical solvers.
//! - `statistics`: Contains probability distributions (e.g., Normal, Binomial, Poisson) along with methods for computing covariance, skewness, kurtosis, and
//! gamma and beta functions.
//! - `random_generators`: Algorithms for generating pseudo-random numbers from uniform and normal distributions, along with the algorithms that connect
//! with probability distributions through a polymorphism to allow generation of pseudo-random numbers from any defined distribution.
//! - `stochastic_processes`: Stochastic processes for simulation of price action, and a builder of custom stochastic processes.
//! - `financial_instruments`: Pricing of different financial instruments (e.g., bonds, stocks, options) and their utility functions.
//! - `portfolio_applications`: Portfolio performance metrics, risk metrics, algorithm for optimizing portfolio of financial instruments for a given
//! performance metric.
//! - `lattice_models`: Binomial and trinomial models for pricing options (European, American and Bermudan) with any payoff (e.g., Call, Straddle).
//! - `monte_carlo`: Function for pricing custom payoffs over any stochastic processes using Monte-Carlo simulations.
//! - `corporate_finance`: Functions for valuation of investment projects, and CAPM.
//! - `technical_indicators`: Trading indicators such as RSI, MACD, Bollinger Bands, etc.
//! - `market_making`: Automated Market Making (AMM) algorithm for simulation of transactions in liquidity pools.
//!
//! # Disclaimer
//!
//! Note that the developers of this package do not accept any responsibility or liability for the accuracy or completeness of the code or the information provided.
//!
//! # Features
//!
//! The following optional features are provided by `digifi` (Note: none of these features are the enabled by default):
//!
//! - `sample_data`: Provides sample data to test CAPM and portfolio optimization algorithms.
//! - `serde`: Provides serialization/deserialization for certain structs.
//! - `plotly`: Provides functions for plotting results using `plotly` library.
//!
//! # Errors
//!
//! In general, the functions and methods inside `digifi` propagate errors so that the users can decide on the error handling techniques applicable to
//! rtheir use case. However, for some numerical solutions closures are used which can panic at runtime.
//!
//! # Citations
//!
//! Some of the code and algorithms used are from external sources (aside from the dependencies of the library). Below is the list of the sources where
//! the code extracts/algorithms were taken from, along with the purpose they serve in the library. We would like to express gratitude to the developers
//! and academics who contributed to these projects.
//!
//! 1. Nelder-Mead Numerical Solver (<https://github.com/to266/optimize>): Only part of the `optimize` crate was used in the development of this library.
//! Unfortunately, the crate itself is outdated, but it contains a really good implementation of the Nelder-Mead numerical solver. `optimize` is not a
//! dependency to this library, but we have used an extract from its source code and slightly changed it to optimize it for our needs.
//! Nonetheless, `optimize` is a great package and we hope there will be a renewed support for it.
//! 2. Inverse CDF of Poisson Distribution (<https://people.maths.ox.ac.uk/gilesm/codes/poissinv/paper.pdf>): This paper covers the implementation of the
//! method for approximating the inverse CDF of the Poisson distribution.
//! 3. Algorithms for ln Gamma, ln Beta and Regularized Incomplete Beta functions (<https://github.com/statrs-dev/statrs>): The implementation of these functions were taken from `statrs`.
//!
//! # General information
//! If you would like to add a commit or an issue, please do so using the GitHub link to the project:
//! - <https://github.com/Digital-Finance-DigiFi/digifi>


pub mod consts;
pub mod corporate_finance;
pub mod error;
pub mod financial_instruments;
pub mod lattice_models;
pub mod market_making;
pub mod monte_carlo;
pub mod portfolio_applications;
pub mod random_generators;
pub mod statistics;
pub mod stochastic_processes;
pub mod technical_indicators;
pub mod utilities;

// Plotly feature
#[cfg(feature = "plotly")]
pub mod plots;