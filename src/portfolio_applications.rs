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
use crate::error::DigiFiError;
use crate::utilities::compare_array_len;
use crate::statistics::{covariance, percent_change, log_return_transformation};


/// # Description
/// Type of returns calculation.
pub enum ReturnsMethod {
    /// Computes returns of the mean returns per interval of the time series (e.g., average daily returns) and then extrapolates it over a specified period
    ImpliedAverageReturn,
    /// Computes compounded return of the time series and then reduces it to the specified period
    EstimatedFromTotalReturn,
}


/// # Description
/// Transformation applied when computing returns from price series.
pub enum ReturnsTransformation {
    /// No transformation is applied and returns are just the percent return from previous period.
    Arithmetic,
    /// Log transformation is applied to produce log-returns.
    LogReturn,
}


#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// # Description
/// Struct with data to be used inside the InstumentsPortfolio.
pub struct AssetHistData {
    /// Historical price series of the instrument
    price_array: Array1<f64>,
    /// An array of preditable income readings (e.g., dividends for stocks, copouns for bonds, overnight fees, etc.)
    predictable_income: Array1<f64>,
    /// An array of time accompanying the price series
    pub time_array: Array1<f64>,
}

impl AssetHistData {
    /// # Description
    /// Creates a new `AssetHistData` instance.
    /// 
    /// # Input
    /// - `price_array`: Historical price series of the instrument
    /// - `predictable_income`: An array of preditable income readings (e.g., dividends for stocks, copouns for bonds, overnight fees, etc.)
    /// - `time_array`: An array of time accompanying the price series
    /// 
    /// # Errors
    /// - Returns an error if the length of `price_array`, `predictable_income` and/or `time_array` do not match.
    pub fn new(price_array: Array1<f64>, predictable_income: Array1<f64>, time_array: Array1<f64>) -> Result<Self, DigiFiError> {
        compare_array_len(&price_array, &predictable_income, "price_array", "predictable_income")?;
        compare_array_len(&price_array, &time_array, "price_array", "time_array")?;
        Ok(AssetHistData { price_array, predictable_income, time_array })
    }

    /// # Description
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
            return Err(DigiFiError::IndexOutOfRange { title: "AssetHistData".to_owned(), index: index_label.to_owned(), array: "price array".to_owned(), });
        }
        Ok(())
    }

    /// # Description
    /// Returns the number of datapoints in the price time series.
    ///
    /// Note: Predictable income and time arrays will have the same length as the price array.
    pub fn get_n_datapoints(&self) -> usize {
        self.price_array.len()
    }

    /// # Description
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
    pub fn get_price(&self, end_index: usize, start_index: Option<usize>) -> Result<Array1<f64>, DigiFiError> {
        self.validate_index(end_index, "end_index")?;
        let mut start_index_: usize = 0;
        match start_index {
            Some(index) => {
                self.validate_index(index, "start_index")?;
                if end_index <= index {
                    return Err(DigiFiError::ParameterConstraint { title: "AssetHistData".to_owned(), constraint: "The argument `start_index` must be smaller than the `end_index`.".to_owned(), });
                }
                start_index_ = index;
            },
            None => (),
        }
        Ok(self.price_array.slice(s![start_index_..end_index]).to_owned())
    }

    /// # Description
    /// Returns the entire price time series.
    pub fn get_price_all(&self) -> Array1<f64> {
        self.price_array.clone()
    }

    /// # Description
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
    pub fn get_predictable_income(&self, end_index: usize, start_index: Option<usize>) -> Result<Array1<f64>, DigiFiError> {
        self.validate_index(end_index, "end_index")?;
        let mut start_index_: usize = 0;
        match start_index {
            Some(index) => {
                self.validate_index(index, "start_index")?;
                if end_index <= index {
                    return Err(DigiFiError::ParameterConstraint { title: "AssetHistData".to_owned(), constraint: "The argument `start_index` must be smaller than the `end_index`.".to_owned(), });
                }
                start_index_ = index;
            },
            None => (),
        }
        Ok(self.predictable_income.slice(s![start_index_..end_index]).to_owned())
    }

    /// # Description
    /// Returns the entire predictable income time series.
    pub fn get_predictable_income_all(&self) -> Array1<f64> {
        self.predictable_income.clone()
    }
}


/// # Description
/// Provides access to the historical data of the financial instrument.
pub trait PortfolioInstrument {

    /// # Description
    /// Returns asset name/label.
    fn asset_name(&self) -> String;

    /// # Description
    /// Returns the historical data about the financial instrument.
    fn historical_data(&self) -> &AssetHistData;
}


/// # Description
/// Convert an array of prices to an array of returns.
/// 
/// # Input
/// - `price_array`: Price time series
/// 
/// # Output
/// - Time series of returns
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::portfolio_applications::{prices_to_returns, ReturnsTransformation};
///
/// let price_array: Array1<f64> = Array1::from_vec(vec![100.0, 101.0, 102.0, 101.0, 103.0, 102.0]);
/// let result: Array1<f64> = prices_to_returns(&price_array, &ReturnsTransformation::Arithmetic);
///
/// assert!((result - Array1::from_vec(vec![0.01, 1.0/101.0, -1.0/102.0, 2.0/101.0, -1.0/103.0])).sum().abs() < TEST_ACCURACY);
pub fn prices_to_returns(price_array: &Array1<f64>, returns_transformation: &ReturnsTransformation) -> Array1<f64> {
    match returns_transformation {
        ReturnsTransformation::Arithmetic => percent_change(price_array),
        ReturnsTransformation::LogReturn => log_return_transformation(price_array),
    }
}


/// # Description
/// Calculate the average return of a price array.
/// 
/// # Input
/// - `price_array`: Price time series
/// - `method`: Method for computing the returns
/// - `n_periods`: Number of periods used to estimate the average over (e.g., for daily prices n_periods=252 produces annualized average)
/// 
/// # Output
/// - Average return over a certain period
pub fn returns_average(price_array: &Array1<f64>, method: &ReturnsMethod, returns_transformation: &ReturnsTransformation, n_periods: usize) -> Result<f64, DigiFiError> {
    let returns: Array1<f64> = prices_to_returns(price_array, returns_transformation);
    match method {
        ReturnsMethod::ImpliedAverageReturn => {
            Ok((1.0 + returns.mean().ok_or(DigiFiError::MeanCalculation { title: "Returns Average".to_owned(), series: "returns".to_owned(), })?).powi(n_periods as i32) - 1.0)
        },
        ReturnsMethod::EstimatedFromTotalReturn => {
            let returns_len: f64 = returns.len() as f64;
            Ok((1.0 + returns).product().powf((n_periods as f64)/returns_len) - 1.0)
        },
    }
}


/// # Description
/// Calculate the standard deviation of the returns of a price array.
/// 
/// # Input
/// - `price_array`: Price time series
/// - `n_periods`: Number of periods used to estimate the standard deviation over (e.g., for daily prices n_periods=252 produces annualized standard deviation)
/// 
/// # Output
/// - Standard deviation of returns over a certain period
pub fn returns_std(price_array: &Array1<f64>, returns_transformation: &ReturnsTransformation, n_periods: usize) -> Result<f64, DigiFiError> {
    let returns: Array1<f64> = prices_to_returns(price_array, returns_transformation);
    let returns_std: f64 = covariance(&returns, &returns, 0)?.sqrt();
    Ok(returns_std * (n_periods as f64).sqrt())
}


/// # Description
/// Calculate the variance of the returns of a price array.
/// 
/// # Input
/// - `price_array`: Price time series
/// - `n_periods`: Number of periods used to estimate the variance over (e.g., for daily prices n_periods=252 produces annualized variance)
/// 
/// # Output
/// - Variance of returns over a certain period
pub fn returns_variance(price_array: &Array1<f64>, returns_transformation: &ReturnsTransformation, n_periods: usize) -> Result<f64, DigiFiError> {
    let returns: Array1<f64> = prices_to_returns(price_array, returns_transformation);
    Ok(covariance(&returns, &returns, 0)? * (n_periods as f64))
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_prices_to_returns() -> () {
        use crate::portfolio_applications::{prices_to_returns, ReturnsTransformation};
        let price_array: Array1<f64> = Array1::from_vec(vec![100.0, 101.0, 102.0, 101.0, 103.0, 102.0]);
        let result: Array1<f64> = prices_to_returns(&price_array, &ReturnsTransformation::Arithmetic);
        assert!((result - Array1::from_vec(vec![0.01, 1.0/101.0, -1.0/102.0, 2.0/101.0, -1.0/103.0])).sum().abs() < TEST_ACCURACY);
    }
}