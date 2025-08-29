//! # Market Making
//! 
//! Contains tools for analysing order book data and running Automated Market Maker (AMM) simulations.


// Re-Exports
pub use self::order_book::volume_imbalance;
pub use self::amm::{LiquidityCurve, AMMToken, AMMLiquidityPool, AMMTransactionData, AMMTransactionResult, SimpleAMM};


pub mod amm;
pub mod order_book;