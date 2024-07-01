pub mod portfolio_performance;
pub mod portfolio_taxonomy;
pub mod risk_measures;
pub mod utility_functions;


use ndarray::{Array1, s, concatenate, Axis};
use crate::utilities::compare_array_len;
use crate::statistics::covariance;


pub enum ReturnsMethod {
    ImpliedAverageReturn,
    EstimatedFromTotalReturn,
}


pub enum ArrayReturnsFormat {
    ReturnsOfAssets,
    WeightedReturnsOfAssets,
    PortfolioReturns,
    CumulativePortfolioReturns,
}


pub enum PortfolioOptimizationResultFormat {
    Value,
    Weights,
}


#[derive(Clone)]
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
    /// Creates a new AssetHistData instance.
    /// 
    /// # Input
    /// - price_array: Historical price series of the instrument
    /// - predictable_income: An array of preditable income readings (e.g., dividends for stocks, copouns for bonds, overnight fees, etc.)
    /// - time_array: An array of time accompanying the price series
    /// 
    /// # Panics
    /// - Panics if the length of price_array, predictable_income and/or time_array do not match
    pub fn new(price_array: Array1<f64>, predictable_income: Array1<f64>, time_array: Array1<f64>) -> Self {
        compare_array_len(&price_array, &predictable_income, "price_array", "predictable_income");
        compare_array_len(&price_array, &time_array, "price_array", "time_array");
        AssetHistData { price_array, predictable_income, time_array }
    }

    /// # Description
    /// Validation method for an index.
    /// 
    /// # Input
    /// - index: Time index beyond which no data will be returned
    /// - index_lable: Label of the index that will be included in the error message
    /// 
    /// # Panics
    /// - Panics if the index provided is out of bounds for the price array
    fn validate_index(&self, index: usize, index_label: &str) -> () {
        if self.price_array.len() < index {
            panic!("The argument {} is out of range for price array of length {}.", index_label, self.price_array.len());
        }
    }

    /// # Description
    /// Safe method for working with historical price data.
    /// 
    /// This method prevents the user from using the future prices based on the indices value provided.
    /// 
    /// # Input
    /// - end_index: Time index beyond which no data will be returned
    /// - start_index: Time index below which no data will be returned
    /// 
    /// # Output
    /// - Historical and/or current prices(s)
    /// 
    /// # Panics
    /// - Panics if the index provided is out of bounds for the price array
    pub fn get_price(&self, end_index: usize, start_index: Option<usize>) -> Array1<f64> {
        self.validate_index(end_index, "end_index");
        match start_index {
            Some(index) => {
                self.validate_index(index, "start_index");
                if end_index <= index {
                    panic!("The argument start_index must be smaller than the end_index");
                }
            },
            None => (),
        }
        self.price_array.slice(s![start_index.expect("Could not unwrap start_index.")..end_index]).to_owned()
    }

    /// # Description
    /// Safe method for working with historical predictable income data.
    /// 
    /// This method prevents the user from using the future predictable incomes based on the indices value provided.
    /// 
    /// # Input
    /// - end_index: Time index beyond which no data will be returned
    /// - start_index: Time index below which no data will be returned
    /// 
    /// # Output
    /// - Historical and/or current predictable income(s)
    pub fn get_predictable_income(&self, end_index: usize, start_index: Option<usize>) -> Array1<f64> {
        self.validate_index(end_index, "end_index");
        match start_index {
            Some(index) => {
                self.validate_index(index, "start_index");
                if end_index <= index {
                    panic!("The argument start_index must be smaller than the end_index");
                }
            },
            None => (),
        }
        self.predictable_income.slice(s![start_index.expect("Could not unwrap start_index.")..end_index]).to_owned()
    }
}


/// # Description
/// Calculate the volatility of a price array from historical data.
/// 
/// Note: There must be fixed time intervals between prices, and the distribution of prices is considered to be log-normal.
/// 
/// # Input
/// - price_array: Price time series
/// - n_periods: Number of periods used to estimate the volatility over (e.g., for daily prices n_periods=252 produces annualized volatility)
/// 
/// # Output
/// - Volatility of price over a certain period
pub fn price_volatility(price_array: &Array1<f64>, n_periods: usize) -> f64 {
    let log_price_diff: Array1<f64> = price_array.slice(s![1..]).map(|x| x.ln()) - price_array.slice(s![0..(price_array.len()-1)]).map(|x| x.ln());
    let std: f64 = covariance(&log_price_diff, &log_price_diff, 0).sqrt();
    std * (n_periods as f64).sqrt()
}


/// # Description
/// Convert an array of prices to an array of returns.
/// 
/// # Input
/// - price_array: Price time series
/// 
/// # Output
/// - Time series of returns
pub fn prices_to_returns(price_array: &Array1<f64>) -> Array1<f64> {
    let price_diff: Array1<f64> = &price_array.slice(s![1..(price_array.len())]) - &price_array.slice(s![0..(price_array.len()-1)]);
    let returns: Array1<f64> = price_diff / price_array.slice(s![0..(price_array.len()-1)]);
    concatenate![Axis(0), Array1::from_vec(vec![0.0]), returns]
}


/// # Description
/// Calculate the average return of a price array.
/// 
/// # Input
/// - price_array: Price time series
/// - method: Method for computing the returns
/// - n_periods: Number of periods used to estimate the average over (e.g., for daily prices n_periods=252 produces annualized average)
/// 
/// # Output
/// - Average return over a certain period
pub fn returns_average(price_array: &Array1<f64>, method: ReturnsMethod, n_periods: usize) -> f64 {
    let returns: Array1<f64> = prices_to_returns(price_array);
    match method {
        ReturnsMethod::ImpliedAverageReturn => {
            (1.0 + returns.mean().expect("Could not compute the mean of the returns.")).powi(n_periods as i32) - 1.0
        },
        ReturnsMethod::EstimatedFromTotalReturn => {
            let returns_len: f64 = returns.len() as f64;
            (1.0 + returns).product().powf((n_periods as f64)/returns_len) - 1.0
        },
    }
}


/// # Description
/// Calculate the standard deviation of the returns of a price array.
/// 
/// # Input
/// - price_array: Price time series
/// - n_periods: Number of periods used to estimate the standard deviation over (e.g., for daily prices n_periods=252 produces annualized standard deviation)
/// 
/// # Output
/// - Standard deviation of returns over a certain period
pub fn returns_std(price_array: &Array1<f64>, n_periods: usize) -> f64 {
    let returns: Array1<f64> = prices_to_returns(price_array);
    let returns_std: f64 = covariance(&returns, &returns, 0).sqrt();
    returns_std * (n_periods as f64).sqrt()
}


/// # Description
/// Calculate the variance of the returns of a price array.
/// 
/// # Input
/// - price_array: Price time series
/// - n_periods: Number of periods used to estimate the variance over (e.g., for daily prices n_periods=252 produces annualized variance)
/// 
/// # Output
/// - Variance of returns over a certain period
pub fn returns_variance(price_array: &Array1<f64>, n_periods: usize) -> f64 {
    let returns: Array1<f64> = prices_to_returns(price_array);
    covariance(&returns, &returns, 0) * (n_periods as f64)
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_price_volatility() -> () {
        use crate::portfolio_applications::price_volatility;
        let price_array: Array1<f64> = Array1::from_vec(vec![100.0, 101.0, 102.0, 101.0, 103.0, 102.0]);
        let vol: f64 = price_volatility(&price_array, 252);
        // The result was found using alternative Python code
        assert!((vol - 0.18707565202263976).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_prices_to_returns() -> () {
        use crate::portfolio_applications::prices_to_returns;
        let price_array: Array1<f64> = Array1::from_vec(vec![100.0, 101.0, 102.0, 101.0, 103.0, 102.0]);
        let result: Array1<f64> = prices_to_returns(&price_array);
        assert!((result - Array1::from_vec(vec![0.0, 0.01, 1.0/101.0, -1.0/102.0, 2.0/101.0, -1.0/103.0])).sum().abs() < TEST_ACCURACY);
    }
}