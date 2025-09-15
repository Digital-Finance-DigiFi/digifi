//! # Portfolio Applications
//! 
//! Provides functionality for portfolio selection and optimization. The module contains functionality for performing portfolio selection,
//! measuring portfolio performance and risk.


// Re-Exports
pub use self::portfolio_performance::{PortfolioPerformanceMetric, SharpeRatio, InformationRatio, TreynorRatio, JensensAlpha, SortinoRatio};
pub use self::portfolio_taxonomy::PortfolioTaxonomy;
pub use self::risk_measures::{value_at_risk, expected_shortfall};
pub use self::utility_functions::{cara, crra};
pub use self::portfolio_composition::{AssetReturnsType, PortfolioReturnsType, PortfolioOptimizationResult, EfficientFrontier, Asset, generate_portfolio, Portfolio};


pub mod portfolio_performance;
pub mod portfolio_taxonomy;
pub mod risk_measures;
pub mod utility_functions;
pub mod portfolio_composition;


use ndarray::{Array1, s};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::{compare_len, Time};


#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Struct with data to be used inside the InstumentsPortfolio.
pub struct AssetHistData {
    /// Historical price series of the instrument
    price_array: Array1<f64>,
    /// An array of preditable income readings (e.g., dividends for stocks, copouns for bonds, overnight fees, etc.)
    predictable_income: Array1<f64>,
    /// An array of time accompanying the price series
    pub time: Time,
}

impl AssetHistData {
    /// Creates a new `AssetHistData` instance.
    /// 
    /// # Input
    /// - `price_array`: Historical price series of the instrument
    /// - `predictable_income`: An array of preditable income readings (e.g., dividends for stocks, copouns for bonds, overnight fees, etc.)
    /// - `time_array`: An array of time accompanying the price series
    /// 
    /// # Errors
    /// - Returns an error if the length of `price_array`, `predictable_income` and/or `time_array` do not match.
    pub fn build(price_array: Array1<f64>, predictable_income: Array1<f64>, time: Time) -> Result<Self, DigiFiError> {
        compare_len(&price_array.iter(), &predictable_income.iter(), "price_array", "predictable_income")?;
        compare_len(&price_array.iter(), &time.time_array().iter(), "price_array", "time_array")?;
        Ok(AssetHistData { price_array, predictable_income, time })
    }

    /// Validation method for an index.
    /// 
    /// # Input
    /// - `index`: Time index beyond which no data will be returned
    /// - `index_lable`: Label of the index that will be included in the error message
    /// 
    /// # Errors
    /// - Returns an error if the index provided is out of bounds for the price array.
    fn validate_index(&self, index: usize, index_label: &str) -> Result<(), DigiFiError> {
        if self.price_array.len() < index {
            return Err(DigiFiError::IndexOutOfRange { title: Self::error_title(), index: index_label.to_owned(), array: "price array".to_owned(), });
        }
        Ok(())
    }

    /// Validates the pair of indices.
    fn validate_index_pair(&self, end_index: usize, start_index: Option<usize>) -> Result<(usize, usize), DigiFiError> {
        self.validate_index(end_index, "end_index")?;
        let start_index: usize = match start_index {
            Some(index) => {
                self.validate_index(index, "start_index")?;
                if end_index <= index {
                    return Err(DigiFiError::ParameterConstraint {
                        title: Self::error_title(),
                        constraint: "The argument `start_index` must be smaller than the `end_index`.".to_owned(),
                    });
                }
                index
            },
            None => 0,
        };
        Ok((start_index, end_index))
    }

    /// Returns the number of datapoints in the price time series.
    ///
    /// Note: Predictable income and time arrays will have the same length as the price array.
    pub fn len(&self) -> usize {
        self.price_array.len()
    }

    /// Safe method for working with historical price data.
    /// 
    /// This method prevents the user from using the future prices based on the indices value provided.
    /// 
    /// # Input
    /// - `end_index`: Time index beyond which no data will be returned
    /// - `start_index`: Time index below which no data will be returned
    /// 
    /// # Output
    /// - Historical and/or current prices(s)
    /// 
    /// # Errors
    /// - Returns an error if the index provided is out of bounds for the price array.
    pub fn price_slice(&self, end_index: usize, start_index: Option<usize>) -> Result<Array1<f64>, DigiFiError> {
        let (start_index, end_index) = self.validate_index_pair(end_index, start_index)?;
        Ok(self.price_array.slice(s![start_index..end_index]).to_owned())
    }

    /// Returns the clone of the entire price time series.
    pub fn price_clone(&self) -> Array1<f64> {
        self.price_array.clone()
    }

    /// Safe method for working with historical predictable income data.
    /// 
    /// This method prevents the user from using the future predictable incomes based on the indices value provided.
    /// 
    /// # Input
    /// - `end_index`: Time index beyond which no data will be returned
    /// - `start_index`: Time index below which no data will be returned
    /// 
    /// # Output
    /// - Historical and/or current predictable income(s)
    pub fn predictable_income_slice(&self, end_index: usize, start_index: Option<usize>) -> Result<Array1<f64>, DigiFiError> {
        let (start_index, end_index) = self.validate_index_pair(end_index, start_index)?;
        Ok(self.predictable_income.slice(s![start_index..end_index]).to_owned())
    }

    /// Returns the clone of the entire predictable income time series.
    pub fn predictable_income_clone(&self) -> Array1<f64> {
        self.predictable_income.clone()
    }
}

impl ErrorTitle for AssetHistData {
    fn error_title() -> String {
        String::from("Asset Historical Data (AssetHistData)")
    }
}


/// Provides access to the historical data of the financial instrument.
pub trait PortfolioInstrument {

    /// Returns asset name/label.
    fn asset_name(&self) -> String;

    /// Returns the historical data about the financial instrument.
    fn historical_data(&self) -> &AssetHistData;
}


/// Type of returns calculation.
pub enum ReturnsMethod {
    /// Computes returns of the mean returns per interval of the time series (e.g., average daily returns) and then extrapolates it over a specified period
    ImpliedAverageReturn,
    /// Computes compounded return of the time series and then reduces it to the specified period
    EstimatedFromTotalReturn,
}


/// Calculate the average return of returrns series.
/// 
/// # Input
/// - `returns`: Price time series
/// - `method`: Method for computing the returns
/// - `n_periods`: Number of periods used to estimate the average over (e.g., for daily prices n_periods=252 produces annualized average)
/// 
/// # Output
/// - Average return over a certain period
pub fn returns_average(returns: &Array1<f64>, method: &ReturnsMethod, n_periods: usize) -> Result<f64, DigiFiError> {
    match method {
        ReturnsMethod::ImpliedAverageReturn => {
            let mean: f64 =  returns.mean().ok_or(DigiFiError::MeanCalculation { title: "Returns Average".to_owned(), series: "returns".to_owned(), })?;
            Ok((1.0 + mean).powi(n_periods as i32) - 1.0)
        },
        ReturnsMethod::EstimatedFromTotalReturn => {
            let returns_len: f64 = returns.len() as f64;
            Ok((1.0 + returns).product().powf((n_periods as f64)/returns_len) - 1.0)
        },
    }
}