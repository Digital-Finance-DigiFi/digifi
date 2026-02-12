use std::collections::HashMap;
use ndarray::{Array1, Array2, Axis, arr1, concatenate};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
#[cfg(feature = "plotly")]
use plotly::{Plot, Trace, Scatter, Layout, layout::Axis as PlotAxis, common::{Mode, Marker, MarkerSymbol, HoverInfo, color::NamedColor}};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::{
    data_transformations::percent_change,
    loss_functions::{LossFunction, StraddleLoss},
    numerical_engines::nelder_mead
};
use crate::statistics::covariance;
use crate::portfolio_applications::{ReturnsMethod, returns_average, AssetHistData, PortfolioInstrument};
use crate::portfolio_applications::portfolio_performance::PortfolioPerformanceMetric;


/// Type of asset returns calculation.
pub enum AssetReturnsType {
    /// Time series of the returns per asset
    ReturnsOfAssets,
    /// Weighted time series of the returns per asset
    WeightedReturnsOfAssets,
}


/// Type of portfolio returns calculation.
pub enum PortfolioReturnsType {
    /// Time series of the returns of the pertfolio 
    PortfolioReturns,
    /// Time series of the cumulative returns of the portfolio
    CumulativePortfolioReturns,
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Output produced by the portfolio optimization.
pub struct PortfolioOptimizationResult {
    /// The optimized output of the performance metric
    pub performance_score: f64,
    /// Weights of the portfolio that have optimized the performance metric
    pub weights: Vec<f64>,
    /// Names/labels of the assets in the portfolio (The order matches to the order of weights)
    pub assets_names: Vec<String>,
    /// Portfolio return
    pub expected_return: f64,
    /// Portfolio standard deviation
    pub std: f64,
}

impl PortfolioOptimizationResult {

    /// Returns a list of asset names as a `String`.
    pub fn assets_names_string(&self) -> String {
        let mut asset_names: String = String::from("[");
        let n_assets: usize = self.assets_names.len();
        for i in 0..n_assets {
            if i == n_assets {
                asset_names += &self.assets_names[i];
            } else {
                asset_names += &(self.assets_names[i].clone() + ", ");
            }
        }
        asset_names + "]"
    }

    /// Returns a list of weights as a `String`.
    pub fn weights_string(&self) -> String {
        let mut weights: String = String::from("[");
        let n_assets: usize = self.weights.len();
        for i in 0..n_assets {
            if i == n_assets {
                weights += &format!("{:.2}", self.weights[i]);
            } else {
                weights += &(format!("{:.2}", self.weights[i]) + ", ");
            }
        }
        weights + "]"
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Efficient frontier of the market for a given performance metric.
pub struct EfficientFrontier {
    /// Portfolio with maximum performance score based on the performance metric
    pub max_performance: PortfolioOptimizationResult,
    /// Efficient frontier
    pub frontier: Vec<PortfolioOptimizationResult>,
    /// Portfolio with minimum standard deviation of returns
    pub min_std: PortfolioOptimizationResult,
}

impl EfficientFrontier {

    /// Returns a tuple of standard deviations and expected returns of the portfolios that sit on the efficient frontier.
    ///
    /// # Output
    /// - Tuple of standard deviations and expected returns (i.e., (`standard_deviations`, `expected_returns`))
    pub fn frontier_line(&self) -> (Array1<f64>, Array1<f64>) {
        let mut stds: Vec<f64> = Vec::<f64>::new();
        let mut expected_returns: Vec<f64> = Vec::<f64>::new();
        for point in &self.frontier {
            stds.push(point.std);
            expected_returns.push(point.expected_return);
        }
        (Array1::from(stds), Array1::from(expected_returns))
    }
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Describes an asset inside a portfolio.
pub struct Asset {
    /// Historical time series of the asset
    pub hist_data: AssetHistData,
    /// Weight of the asset in the portfolio
    pub weight: f64,
}

impl Asset {
    
    /// Validates the asset data.
    ///
    /// # Errors
    /// - Returns an error if the weight is infinite or NAN.
    pub fn validate(&self) -> Result<(), DigiFiError> {
        if self.weight.is_nan() || self.weight.is_infinite() {
            return Err(DigiFiError::ValidationError {
                title: Self::error_title(),
                details: "The `weight` of the portfolio asset cannot be infinite or `NAN`.".to_owned(),
            });
        }
        Ok(())
    }
}

impl ErrorTitle for Asset {
    fn error_title() -> String {
        String::from("Asset")
    }
}


/// Generates `Portfolio` from the financial instruments provided.
///
/// # Input
/// - `financial_instruments`: List of financial instruments that will form the market for the portfolio
/// - `n_periods`: Number of periods used to estimate the average returns over (e.g., for daily prices n_periods=252 produces annualized average)
/// - `returns_method`: Method for computing the returns
/// - `performance_metric`: Performance metric used to optimized the weights of the protfolio
///
/// # Output
/// - `Portfolio` struct
pub fn generate_portfolio(
    financial_instruments: Vec<impl PortfolioInstrument>, n_periods: Option<usize>, returns_method: Option<ReturnsMethod>,
    performance_metric: Box<dyn PortfolioPerformanceMetric>
) -> Result<Portfolio, DigiFiError> {
    let mut assets: HashMap<String, Asset>  = HashMap::<String, Asset>::new();
    let weight: f64 = 1.0 / financial_instruments.len() as f64;
    for fi in financial_instruments {
        assets.insert(fi.asset_name(), Asset { hist_data: fi.historical_data().clone(), weight });
    }
    Portfolio::build(assets, None, n_periods, returns_method, performance_metric)
}


/// Portfolio of assets and its methods.
///
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Modern_portfolio_theory#Markowitz_bullet>
/// - Original Source: <https://doi.org/10.2307%2F2975974>
///
/// # Examples
///
/// 1. Maximizing the performance of the portfolio (Using Sharpe ratio):
///
/// ```rust
/// use std::collections::HashMap;
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, Time};
/// use digifi::portfolio_applications::{AssetHistData, SharpeRatio, Asset, EfficientFrontier, Portfolio, PortfolioOptimizationResult};
///
/// #[cfg(feature = "sample_data")]
/// fn test_max_performance() -> () {
///     use digifi::utilities::SampleData;
///
///     // Portfolio parameters
///     let sample_data: SampleData = SampleData::Portfolio;
///     let (time, data) = sample_data.load_sample_data();
///     let weight: f64 = 1.0 / data.len() as f64;
///     let dummy_array: Array1<f64> = Array1::from_vec(vec![0.0; time.len()]);
///     let mut assets: HashMap<String, Asset> = HashMap::<String, Asset>::new();
///     for (k, v) in data.into_iter() {
///         let hist_data: AssetHistData = AssetHistData::build(v, dummy_array.clone(), Time::new(dummy_array.clone())).unwrap();
///         assets.insert(k, Asset { hist_data, weight, });
///     }
///     let performance_metric: Box<SharpeRatio> = Box::new(SharpeRatio { rf: 0.02 });
///
///     // Portfolio definition and optimization
///     let mut portfolio: Portfolio = Portfolio::build(assets, None, None, None, performance_metric).unwrap();
///     let max_sr: PortfolioOptimizationResult = portfolio.maximize_performance(Some(1_000), Some(10_000)).unwrap();
///
///     assert!((max_sr.performance_score - 2.010620260010254).abs() < TEST_ACCURACY);
///     assert!((max_sr.expected_return - 0.7005606636574792).abs() < TEST_ACCURACY);
///     assert!((max_sr.std - 0.3384829433947951).abs() < TEST_ACCURACY);
/// }
/// ```
///
/// 2. Minimizing the standard deviation of the portfolio:
///
/// ```rust
/// use std::collections::HashMap;
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, Time};
/// use digifi::portfolio_applications::{AssetHistData, SharpeRatio, Asset, EfficientFrontier, Portfolio, PortfolioOptimizationResult};
///
/// #[cfg(feature = "sample_data")]
/// fn test_min_std() -> () {
///     use digifi::utilities::SampleData;
///
///     // Portfolio parameters
///     let sample_data: SampleData = SampleData::Portfolio;
///     let (time, data) = sample_data.load_sample_data();
///     let weight: f64 = 1.0 / data.len() as f64;
///     let dummy_array: Array1<f64> = Array1::from_vec(vec![0.0; time.len()]);
///     let mut assets: HashMap<String, Asset> = HashMap::<String, Asset>::new();
///     for (k, v) in data.into_iter() {
///         let hist_data: AssetHistData = AssetHistData::build(v, dummy_array.clone(), Time::new(dummy_array.clone())).unwrap();
///         assets.insert(k, Asset { hist_data, weight, });
///     }
///     let performance_metric: Box<SharpeRatio> = Box::new(SharpeRatio { rf: 0.02 });
///
///     // Portfolio definition and optimization
///     let mut portfolio: Portfolio = Portfolio::build(assets, None, None, None, performance_metric).unwrap();
///     let min_std: PortfolioOptimizationResult = portfolio.minimize_std(Some(1_000), Some(10_000)).unwrap();
///
///     assert!((min_std.performance_score - 1.1159801330821704).abs() < TEST_ACCURACY);
///     assert!((min_std.expected_return - 0.22963237821918314).abs() < TEST_ACCURACY);
///     assert!((min_std.std - 0.18784597682775037).abs() < TEST_ACCURACY);
/// }
/// ```
///
/// 3. Building efficient frontier (Using Sharpe ratio):
///
/// ```rust
/// use std::collections::HashMap;
/// use ndarray::Array1;
/// use digifi::utilities::{TEST_ACCURACY, Time};
/// use digifi::portfolio_applications::{AssetHistData, SharpeRatio, Asset, EfficientFrontier, Portfolio, PortfolioOptimizationResult};
///
/// #[cfg(feature = "sample_data")]
/// fn test_efficient_frontier() -> () {
///     use digifi::utilities::SampleData;
///
///     // Portfolio parameters
///     let sample_data: SampleData = SampleData::Portfolio;
///     let (time, data) = sample_data.load_sample_data();
///     let weight: f64 = 1.0 / data.len() as f64;
///     let dummy_array: Array1<f64> = Array1::from_vec(vec![0.0; time.len()]);
///     let mut assets: HashMap<String, Asset> = HashMap::<String, Asset>::new();
///     for (k, v) in data.into_iter() {
///         let hist_data: AssetHistData = AssetHistData::build(v, dummy_array.clone(), Time::new(dummy_array.clone())).unwrap();
///         assets.insert(k, Asset { hist_data, weight, });
///     }
///     let performance_metric: Box<SharpeRatio> = Box::new(SharpeRatio { rf: 0.02 });
///
///     // Portfolio definition and optimization
///     let mut portfolio: Portfolio = Portfolio::build(assets, None, None, None, performance_metric).unwrap();
///     let frontier: EfficientFrontier = portfolio.efficient_frontier(30, Some(1_000), Some(10_000)).unwrap();
///
///     for point in &frontier.frontier {
///         assert!(point.performance_score <= frontier.max_performance.performance_score);
///         assert!(frontier.min_std.std <= point.std)
///     }
/// }
/// ```
pub struct Portfolio {
    /// Names/labels of assets in the portfolio
    assets_names: Vec<String>,
    /// Historical time series of every asset
    assets: Vec<AssetHistData>,
    /// Weights of every asset
    weights: Vec<f64>,
    /// Tolerance for the difference between the cumulative sum of the weights and 1
    rounding_error_tol: f64,
    /// Number of periods used to estimate the average returns over (e.g., for daily prices `n_periods=252` produces annualized average)
    n_periods: usize,
    /// Method for computing the returns
    returns_method: ReturnsMethod,
    /// Performance metric used to optimized the weights of the protfolio
    performance_metric: Box<dyn PortfolioPerformanceMetric>,
}

impl Portfolio {
    /// Creates a new `Portfolio` instance.
    ///
    /// # Input
    /// - `assets`: Hashmap of assets and their historical data
    /// - `rounding_error_tol`: Tolerance for the difference between the cumulative sum of the weights and `1`
    /// - `n_periods`: Number of periods used to estimate the average returns over (e.g., for daily prices `n_periods=252` produces annualized average)
    /// - `returns_method`: Method for computing the returns
    /// - `performance_metric`: Performance metric used to optimized the weights of the protfolio
    ///
    /// # Errors
    /// - Returns an error if the weight is infinite or `NAN`.
    /// - Returns an error if the assets provided do not have time series of the same length.
    /// - Returns an error if the sum of portfolio weights is not equal to `1` (Subject to `rounding_error_tol`).
    pub fn build(
        assets: HashMap<String, Asset>, rounding_error_tol: Option<f64>, n_periods: Option<usize>, returns_method: Option<ReturnsMethod>,
        performance_metric: Box<dyn PortfolioPerformanceMetric>
    ) -> Result<Self, DigiFiError> {
        let rounding_error_tol: f64 = match rounding_error_tol { Some(v) => v, None => 0.001 };
        let mut assets_: Vec<AssetHistData> = Vec::<AssetHistData>::new();
        let mut assets_names: Vec<String> = Vec::<String>::new();
        let mut weights: Vec<f64> = Vec::<f64>::new();
        let mut time_series_len: Option<usize> = None;
        for (k, v) in assets {
            // Validation of the asset (independent of other assets)
            v.validate()?;
            // Validation of the asset (dependent on other assets)
            match time_series_len {
                Some(l) => {
                    if v.hist_data.len() != l {
                        return Err(DigiFiError::ValidationError {
                            title: Self::error_title(),
                            details: "The assets provided do not have time series of the same length.".to_owned(),
                        });
                    }
                },
                None => { time_series_len = Some(v.hist_data.len()); },
            }
            // Generation of data for portfolio
            weights.push(v.weight);
            assets_names.push(k);
            assets_.push(v.hist_data);
        }
        Self::validate_and_clean_weights(&mut weights, rounding_error_tol)?;
        let n_periods: usize = match n_periods { Some(v) => v, None => 252, };
        let returns_method: ReturnsMethod = match returns_method { Some(v) => v, None => ReturnsMethod::EstimatedFromTotalReturn };
        Ok(Self { assets_names, assets: assets_, weights, rounding_error_tol, n_periods, returns_method, performance_metric, })
    }

    /// Returns the names of the assets in portfolio.
    pub fn assets_names(&self) -> &Vec<String> {
        &self.assets_names
    }

    /// Returns the historical asset data for every asset in the portfolio.
    pub fn assets(&self) -> &Vec<AssetHistData> {
        &self.assets
    }

    /// Validates the weights of the portfolio, and cleans them (i.e., makes the sum of weights equal to `1`) if the dicrepancy
    /// between the weights and 1 is less than the `rounding_error_tol`.
    ///
    /// # Input
    /// - `weights`: Weights of the portfolio
    /// - `rounding_error_tol`: Tolerance for the difference between the cumulative sum of the weights and `1`
    ///
    /// # Errors
    /// - Returns an error if the sum of portfolio weights is not equal to 1 (Subject to `rounding_error_tol`).
    fn validate_and_clean_weights(weights: &mut Vec<f64>, rounding_error_tol: f64) -> Result<(), DigiFiError> {
        let mut cumsum: f64 = 0.0;
        for w in weights.iter() {
            cumsum += *w;
        }
        let diff: f64 = cumsum - 1.0;
        if rounding_error_tol < diff.abs() {
            return Err(DigiFiError::ValidationError { title: Self::error_title(), details: "The sum of protfolio weights is not equal to `1`.".to_owned(), });
        }
        weights[0] = weights[0] + diff;
        Ok(())
    }

    /// Generates uniform weights across all assets in the market.
    fn uniform_weights(&self) -> Vec<f64> {
        let n_assets: usize = self.assets.len();
        let mut weights: Vec<f64> = vec![1.0 / (n_assets as f64); n_assets];
        let diff: f64 = weights.iter().sum::<f64>() - 1.0;
        weights[0] += diff;
        weights
    }

    /// Update weights of the portfolio of assets.
    ///
    /// # Input
    /// - `weights`: New set of weights for the portfolio
    ///
    /// # Errors
    /// - Returns an error if the number of weights does not match the number of assets.
    /// - Rerutns an error if the sum of portfolio weights is not equal to 1 (Subject to `rounding_error_tol`).
    pub fn change_weights(&mut self, mut new_weights: Vec<f64>) -> Result<(), DigiFiError> {
        if self.assets.len() != new_weights.len() {
            return Err(DigiFiError::ValidationError { title: Self::error_title(), details: "The number of weights does not match the number of assets.".to_owned() });
        }
        Self::validate_and_clean_weights(&mut new_weights, self.rounding_error_tol)?;
        self.weights = new_weights;
        Ok(())
    }

    /// Adds assets to the market of the portfolio.
    ///
    /// Note: This action will reset the weights in the portfolio to be uniform.
    ///
    /// # Input
    /// - `new_assets`: New assets to add to the market of the portfolio
    ///
    /// # Errors
    /// - Returns an error if the weight is infinite or `NAN`.
    /// - Returns an error if the assets provided do not have time series of the same length.
    pub fn add_assets(&mut self, new_assets: HashMap<String, Asset>) -> Result<(), DigiFiError> {
        let mut new_assets_names: Vec<String> = Vec::<String>::new();
        let mut new_hist_data: Vec<AssetHistData> = Vec::<AssetHistData>::new();
        let time_series_len: usize = self.assets[0].len();
        for (k, v) in new_assets {
            // Validation of the asset (independent of other assets)
            v.validate()?;
            // Validation of the asset (dependend of other assets)
            if v.hist_data.len() != time_series_len {
                return Err(DigiFiError::ValidationError {
                    title: Self::error_title(),
                    details: "The assets provided do not have time series of the same length.".to_owned(),
                });
            }
            // Generation of data for portfolio
            new_assets_names.push(k);
            new_hist_data.push(v.hist_data);
        }
        // Generate uniform weights
        let weights: Vec<f64> = self.uniform_weights();
        // Update portfolio state
        self.change_weights(weights)?;
        self.assets_names.extend(new_assets_names);
        self.assets.extend(new_hist_data);
        Ok(())
    }

    /// Removes assets from the market of the portfolio.
    ///
    /// Note: This action will reset the weights in the portfolio to be uniform.
    ///
    /// # Input
    /// - `assets_names`: Names/labels of the assets to remove
    pub fn remove_assets(&mut self, assets_names: Vec<String>) -> Result<(), DigiFiError> {
        for label in assets_names {
            let mut index: Option<usize> = None;
            for i in 0..self.assets_names.len() {
                // Delete historical data for one asset
                if label == self.assets_names[i] {
                    self.assets.remove(i);
                    index = Some(i);
                    break;
                }
            }
            // Delete asset name
            if let Some(i) = index { self.assets_names.remove(i); }
        }
        let weights: Vec<f64> = self.uniform_weights();
        self.change_weights(weights)?;
        Ok(())
    }

    fn predictable_income_to_return(price: &Array1<f64>, predictable_income: &Array1<f64>) -> Array1<f64> {
        predictable_income / price
    }

    /// Calculates the returns of the assets for the provided asset returns type.
    ///
    /// # Input
    /// - `asset_returns_type`: Type of asset returns calculation to perform
    ///
    /// # Output
    /// - Time series of asset returns (Regular or weighted)
    pub fn asset_returns(&self, asset_returns_type: &AssetReturnsType) -> Vec<Array1<f64>> {
        let mut returns: Vec<Array1<f64>> = Vec::<Array1<f64>>::new();
        for a in &self.assets {
            let extra_returns: Array1<f64> = Self::predictable_income_to_return(&a.price_array, &a.predictable_income);
            let returns_: Array1<f64> = percent_change(&a.price_array);
            // Append 0.0 to the front of the arithmetic returns array
            let returns_: Array1<f64> = concatenate![Axis(0), Array1::from_vec(vec![0.0]), returns_];
            returns.push(returns_ + extra_returns);
        }
        match asset_returns_type {
            AssetReturnsType::ReturnsOfAssets => (),
            AssetReturnsType::WeightedReturnsOfAssets => {
                for i in 0..self.weights.len() {
                    returns[i] *= self.weights[i];
                }
            },
        }
        returns
    }

    /// Calculates the returns of the portfolio.
    ///
    /// # Input
    /// - `portfolio_returns_type`: Type of portfolio returns calculation to perform
    ///
    /// # Output
    /// - Time series of portfolio returns (Regular or cumulative)
    pub fn portfolio_returns(&self, portfolio_returns_type: &PortfolioReturnsType) -> Array1<f64> {
        let asset_returns: Vec<Array1<f64>> = self.asset_returns(&AssetReturnsType::WeightedReturnsOfAssets);
        let mut portfolio_returns: Array1<f64> = Array1::from_vec(vec![0.0; self.assets[0].len()]);
        for a in asset_returns {
            portfolio_returns = portfolio_returns + a;
        }
        match portfolio_returns_type {
            PortfolioReturnsType::PortfolioReturns => { portfolio_returns },
            PortfolioReturnsType::CumulativePortfolioReturns => {
                portfolio_returns.accumulate_axis_inplace(Axis(0), |&prev, curr| { *curr *= prev });
                portfolio_returns
            },
        }
    }

    /// Calculate the mean return of the portfolio.
    ///
    /// # Output
    /// - Mean of the portfolio returns
    pub fn mean_return(&self) -> Result<f64, DigiFiError> {
        let mut mean: f64 = 0.0;
        for i in 0..self.assets.len() {
            let returns: Array1<f64> = percent_change(&self.assets[i].price_array);
            mean += returns_average(&returns, &self.returns_method, self.n_periods)? * self.weights[i];
        }
        Ok(mean)
    }

    /// Calculate the covariance of asset returns.
    ///
    /// # Output
    /// - Covariance of asset returns
    pub fn covariance(&self) -> Result<Array2<f64>, DigiFiError> {
        let n_assets: usize = self.assets.len();
        let n_periods: f64 = self.n_periods as f64;
        let asset_returns: Vec<Array1<f64>> = self.asset_returns(&AssetReturnsType::ReturnsOfAssets);
        let base: Vec<f64> = vec![0.0; n_assets];
        let mut cov_matrix: Vec<Vec<f64>> = vec![base; n_assets];
        for i in 0..n_assets {
            for j in 0..n_assets {
                if i <= j {
                    let cov: f64 = covariance(&asset_returns[i], &asset_returns[j], 0)? * n_periods;
                    cov_matrix[i][j] = cov;
                    cov_matrix[j][i] = cov;
                }
            }
        }
        let flat_arr: Vec<f64> = cov_matrix.into_iter().flatten().collect();
        Ok(Array2::from_shape_vec((n_assets, n_assets), flat_arr)?)
    }

    /// Calculate the standard deviation of the portfolio.
    ///
    /// # Output
    /// - Standard deviavion of the portfolio returns
    pub fn standard_deviation(&self) -> Result<f64, DigiFiError> {
        let weights: Array1<f64> = Array1::from_vec(self.weights.clone());
        let cov_matrix: Array2<f64> = self.covariance()?;
        // Standard deviation
        let proxy: Array1<f64> = cov_matrix.dot(&weights);
        Ok((&weights.t().dot(&proxy)).sqrt())
    }

    /// Computes the performance of the portfolio for a given performance metric.
    ///
    /// # Output
    /// - Portfolio performance score
    pub fn performance(&self) -> Result<f64, DigiFiError> {
        let mean_returns: f64 = self.mean_return()?;
        let std_returns: f64 = self.standard_deviation()?;
        Ok(self.performance_metric.performance(mean_returns, std_returns))
    }

    /// Finds the portfolio that optimizes the performance metric.
    ///
    /// # Input
    /// - `max_iterations`: Maximum number of iterations the algorithm is allowed to perform
    /// - `max_fun_calls`: Maximum number of function calls the algorithm is allowed to perform
    ///
    /// # Output
    /// - Portfolio optimization result (i.e., optimized performance score and weights of the protfolio that produce that performance score)
    pub fn maximize_performance(&mut self, max_iterations: Option<u64>, max_fun_calls: Option<u64>) -> Result<PortfolioOptimizationResult, DigiFiError> {
        let initial_guess: Vec<f64> = self.uniform_weights();
        let f = |v: &[f64]| {
            // Weights constraint
            let weights: Array1<f64> = arr1(v);
            let weights: Vec<f64> = (&weights / weights.sum()).to_vec();
            self.change_weights(weights).unwrap();
            // Portfolio optimization
            let mean_returns: f64 = self.mean_return().unwrap();
            let std_returns: f64 = self.standard_deviation().unwrap();
            self.performance_metric.objective_function(mean_returns, std_returns)
        };
        let weights: Vec<f64> = nelder_mead(f, initial_guess, max_iterations, max_fun_calls, Some(true), None, None)?.to_vec();
        self.change_weights(weights)?;
        let performance: f64 = self.performance()?;
        let expected_return: f64 = self.mean_return()?;
        let std: f64 = self.standard_deviation()?;
        Ok(PortfolioOptimizationResult { performance_score: performance, weights: self.weights.clone(), assets_names: self.assets_names.clone(), expected_return, std })
    }

    /// Find portfolio with lowest standard deviation.
    ///
    /// # Input
    /// - `max_iterations`: Maximum number of iterations the algorithm is allowed to perform
    /// - `max_fun_calls`: Maximum number of function calls the algorithm is allowed to perform
    ///
    /// # Output
    /// - Portfolio optimization result (i.e., optimized performance score and weights of the protfolio that produce minimum standard deviation)
    pub fn minimize_std(&mut self, max_iterations: Option<u64>, max_fun_calls: Option<u64>) -> Result<PortfolioOptimizationResult, DigiFiError> {
        let initial_guess: Vec<f64> = self.uniform_weights();
        let f = |v: &[f64]| {
            // Weights constraint
            let weights: Array1<f64> = arr1(v);
            let weights: Vec<f64> = (&weights / weights.sum()).to_vec();
            self.change_weights(weights).unwrap();
            // Portfolio optimization
            self.standard_deviation().unwrap()
        };
        let weights: Vec<f64> = nelder_mead(f, initial_guess, max_iterations, max_fun_calls, Some(true), None, None)?.to_vec();
        self.change_weights(weights)?;
        let performance: f64 = self.performance()?;
        let expected_return: f64 = self.mean_return()?;
        let std: f64 = self.standard_deviation()?;
        Ok(PortfolioOptimizationResult { performance_score: performance, weights: self.weights.clone(), assets_names: self.assets_names.clone(), expected_return, std })
    }

    /// Find risk level on the efficient frontier for a given target return.
    ///
    /// # Input
    /// - `target_return`: Expected return to optimize volatility for
    /// - `max_iterations`: Maximum number of iterations the algorithm is allowed to perform
    /// - `max_fun_calls`: Maximum number of function calls the algorithm is allowed to perform
    ///
    /// # Output
    /// - Portfolio optimization result (i.e., optimized performance score and weights of the protfolio that produce the target return)
    pub fn efficient_optimization(&mut self, target_return: f64, max_iterations: Option<u64>, max_fun_calls: Option<u64>) -> Result<PortfolioOptimizationResult, DigiFiError> {
        let initial_guess: Vec<f64> = self.uniform_weights();
        let f = |v: &[f64]| {
            // Weights constraint
            let weights: Array1<f64> = arr1(v);
            let weights: Vec<f64> = (&weights / weights.sum()).to_vec();
            self.change_weights(weights).unwrap();
            // Portfolio optimization
            let mean_returns: f64 = self.mean_return().unwrap();
            let std_returns: f64 = self.standard_deviation().unwrap();
            let score: f64 = self.performance_metric.objective_function(mean_returns, std_returns);
            // Combined loss of the objective function and target return constraint (Optimized via straddle payoff function)
            score + StraddleLoss.loss(mean_returns, target_return)
        };
        let weights: Vec<f64> = nelder_mead(f, initial_guess, max_iterations, max_fun_calls, Some(false), None, None)?.to_vec();
        self.change_weights(weights)?;
        let performance: f64 = self.performance()?;
        let expected_return: f64 = self.mean_return()?;
        let std: f64 = self.standard_deviation()?;
        Ok(PortfolioOptimizationResult {
            performance_score: performance, weights: self.weights.clone(), assets_names: self.assets_names.clone(), expected_return, std
        })
    }

    /// Calculate efficient frontier.
    ///
    /// # Input
    /// - `n_points`: Number of points to generate in the efficient frontier
    /// - `max_iterations`: Maximum number of iterations the algorithm is allowed to perform
    /// - `max_fun_calls`: Maximum number of function calls the algorithm is allowed to perform
    ///
    /// # Output
    /// - Efficient frontier over the market for a given performance metric
    pub fn efficient_frontier(&mut self, n_points: usize, max_iterations: Option<u64>, max_fun_calls: Option<u64>) -> Result<EfficientFrontier, DigiFiError> {
        let min_std: PortfolioOptimizationResult = self.minimize_std(max_iterations, max_fun_calls)?;
        let max_performance: PortfolioOptimizationResult = self.maximize_performance(max_iterations, max_fun_calls)?;
        let mut frontier: Vec<PortfolioOptimizationResult> = Vec::<PortfolioOptimizationResult>::new();
        let target_returns: Array1<f64> = Array1::linspace(min_std.expected_return, max_performance.expected_return + min_std.expected_return, n_points);
        for target_return in target_returns {
            frontier.push(self.efficient_optimization(target_return, max_iterations, max_fun_calls)?);
        } 
        Ok(EfficientFrontier { max_performance, frontier, min_std })
    }
}

impl ErrorTitle for Portfolio {
    fn error_title() -> String {
        String::from("Portfolio")
    }
}


#[cfg(feature = "plotly")]
/// Plots the efficient frontier.
///
/// # Input
/// - `frontier`: Efficient frontier data to plot
///
/// # Output
/// - Efficient frontier plot
///
/// # Examples
///
/// ```rust,ignore
/// use std::collections::HashMap;
/// use ndarray::Array1;
/// use digifi::utilities::Time;
/// use digifi::portfolio_applications::{AssetHistData, SharpeRatio, Asset, EfficientFrontier, Portfolio};
///
/// #[cfg(all(feature = "plotly", feature = "sample_data"))]
/// fn test_candlestick_chart() -> () {
///     use plotly::Plot;
///     use digifi::utilities::sample_data::SampleData;
///     use digifi::plots::plot_efficient_frontier;
///
///     // Portfolio parameters
///     let sample_data: SampleData = SampleData::Portfolio;
///     let (time, data) = sample_data.load_sample_data();
///     let weight: f64 = 1.0 / data.len() as f64;
///     let dummy_array: Array1<f64> = Array1::from_vec(vec![0.0; time.len()]);
///     let mut assets: HashMap<String, Asset> = HashMap::<String, Asset>::new();
///     for (k, v) in data.into_iter() {
///         let hist_data: AssetHistData = AssetHistData::build(v, dummy_array.clone(), Time::new(dummy_array.clone())).unwrap();
///         assets.insert(k, Asset { hist_data, weight, });
///     }
///     let performance_metric: Box<SharpeRatio> = Box::new(SharpeRatio { rf: 0.02 });
///
///     // Portfolio definition and optimization
///     let mut portfolio: Portfolio = Portfolio::build(assets, None, None, None, performance_metric).unwrap();
///     let frontier: EfficientFrontier = portfolio.efficient_frontier(30, Some(1_000), Some(10_000)).unwrap();
///
///     // Efficient frontier plot
///     let plot: Plot = plot_efficient_frontier(frontier);
///     plot.show();
/// }
/// ```
pub fn plot_efficient_frontier(frontier: EfficientFrontier) -> Plot {
    let mut plot: Plot = Plot::new();
    // Minimum volatility portfolio
    let min_vol_marker: Marker = Marker::new().symbol(MarkerSymbol::Diamond).color(NamedColor::Red).size(14);
    let min_vol_hover: String = format!(
        "Standard Deviation: {:.2}<br>Expected Return: {:.2}<br>Performance Score: {:.2}<br>Asset Names: {}<br>Weights: {}",
        frontier.min_std.std, frontier.min_std.expected_return, frontier.min_std.performance_score, frontier.min_std.assets_names_string(),
        frontier.min_std.weights_string()
    );
    let min_vol: Box<Scatter<f64, f64>> = Scatter::new(vec![100.0 * frontier.min_std.std], vec![100.0 * frontier.min_std.expected_return])
        .name("Minimum Volatility Portfolio")
        .mode(Mode::Markers)
        .marker(min_vol_marker)
        .hover_info(HoverInfo::Text)
        .hover_text(min_vol_hover);
    // Maximum performance portfolio
    let max_per_marker: Marker = Marker::new().symbol(MarkerSymbol::Star).color(NamedColor::Green).size(14);
    let max_per_hover: String = format!(
        "Standard Deviation: {:.2}<br>Expected Return: {:.2}<br>Performance Score: {:.2}<br>Asset Names: {}<br>Weights: {}",
        frontier.max_performance.std, frontier.max_performance.expected_return, frontier.max_performance.performance_score,
        frontier.max_performance.assets_names_string(), frontier.max_performance.weights_string()
    );
    let max_per: Box<Scatter<f64, f64>> = Scatter::new(vec![100.0 * frontier.max_performance.std], vec![100.0 * frontier.max_performance.expected_return])
        .name("Maximum Performance Portfolio")
        .mode(Mode::Markers)
        .marker(max_per_marker)
        .hover_info(HoverInfo::Text)
        .hover_text(max_per_hover);
    // Efficient frontier
    let eff_marker: Marker = Marker::new().symbol(MarkerSymbol::Circle).color(NamedColor::Black).size(7);
    let mut eff_traces: Vec<Box<dyn Trace>> = Vec::<Box<dyn Trace>>::new();
    for eff in frontier.frontier {
        let hover_text: String = format!(
            "Standard Deviation: {:.2}<br>Expected Return: {:.2}<br>Performance Score: {:.2}<br>Asset Names: {}<br>Weights: {}",
            eff.std, eff.expected_return, eff.performance_score, eff.assets_names_string(), eff.weights_string()
        );
        eff_traces.push(
            Scatter::new(vec![100.0 * eff.std], vec![100.0 * eff.expected_return])
                .mode(Mode::Markers)
                .marker(eff_marker.clone())
                .hover_info(HoverInfo::Text).hover_text(hover_text)
        );
    }
    // Select trace order
    plot.add_traces(eff_traces);
    plot.add_trace(min_vol);
    plot.add_trace(max_per);
    // Layout
    let x_axis: PlotAxis = PlotAxis::new().title("Standard Deviation");
    let y_axis: PlotAxis = PlotAxis::new().title("Expected Return (%)");
    let layout: Layout = Layout::new().title("<b>Efficient Frontier</b>").x_axis(x_axis).y_axis(y_axis);
    plot.set_layout(layout);
    plot
}


#[cfg(all(test, feature = "sample_data"))]
mod tests {
    use std::collections::HashMap;
    use ndarray::Array1;
    use crate::portfolio_applications::{AssetHistData, portfolio_performance::SharpeRatio};
    use crate::portfolio_applications::portfolio_composition::{Asset, EfficientFrontier, Portfolio, PortfolioOptimizationResult};
    use crate::utilities::{TEST_ACCURACY, Time, sample_data::SampleData};

    #[test]
    fn unit_test_portfolio_maximize_performance() -> () {
        // Portfolio parameters
        let sample_data: SampleData = SampleData::Portfolio;
        let (time, data) = sample_data.load_sample_data();
        let weight: f64 = 1.0 / data.len() as f64;
        let dummy_array: Array1<f64> = Array1::from_vec(vec![0.0; time.len()]);
        let mut assets: HashMap<String, Asset> = HashMap::<String, Asset>::new();
        for (k, v) in data.into_iter() {
            let hist_data: AssetHistData = AssetHistData::build(v, dummy_array.clone(), Time::new(dummy_array.clone())).unwrap();
            assets.insert(k, Asset { hist_data, weight, });
        }
        let performance_metric: Box<SharpeRatio> = Box::new(SharpeRatio { rf: 0.02 });
        // Portfolio definition and optimization
        let mut portfolio: Portfolio = Portfolio::build(assets, None, None, None, performance_metric).unwrap();
        let max_sr: PortfolioOptimizationResult = portfolio.maximize_performance(Some(1_000), Some(10_000)).unwrap();
        assert!((max_sr.performance_score - 2.010620260010254).abs() < TEST_ACCURACY);
        assert!((max_sr.expected_return - 0.7005606636574792).abs() < TEST_ACCURACY);
        assert!((max_sr.std - 0.3384829433947951).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_portfolio_minimize_std() -> () {
        // Portfolio parameters
        let sample_data: SampleData = SampleData::Portfolio;
        let (time, data) = sample_data.load_sample_data();
        let weight: f64 = 1.0 / data.len() as f64;
        let dummy_array: Array1<f64> = Array1::from_vec(vec![0.0; time.len()]);
        let mut assets: HashMap<String, Asset> = HashMap::<String, Asset>::new();
        for (k, v) in data.into_iter() {
            let hist_data: AssetHistData = AssetHistData::build(v, dummy_array.clone(), Time::new(dummy_array.clone())).unwrap();
            assets.insert(k, Asset { hist_data, weight, });
        }
        let performance_metric: Box<SharpeRatio> = Box::new(SharpeRatio { rf: 0.02 });
        // Portfolio definition and optimization
        let mut portfolio: Portfolio = Portfolio::build(assets, None, None, None, performance_metric).unwrap();
        let min_std: PortfolioOptimizationResult = portfolio.minimize_std(Some(1_000), Some(10_000)).unwrap();
        assert!((min_std.performance_score - 1.1159801330821704).abs() < TEST_ACCURACY);
        assert!((min_std.expected_return - 0.22963237821918314).abs() < TEST_ACCURACY);
        assert!((min_std.std - 0.18784597682775037).abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_portfolio_efficient_frontier() -> () {
        // Portfolio parameters
        let sample_data: SampleData = SampleData::Portfolio;
        let (time, data) = sample_data.load_sample_data();
        let weight: f64 = 1.0 / data.len() as f64;
        let dummy_array: Array1<f64> = Array1::from_vec(vec![0.0; time.len()]);
        let mut assets: HashMap<String, Asset> = HashMap::<String, Asset>::new();
        for (k, v) in data.into_iter() {
            let hist_data: AssetHistData = AssetHistData::build(v, dummy_array.clone(), Time::new(dummy_array.clone())).unwrap();
            assets.insert(k, Asset { hist_data, weight, });
        }
        let performance_metric: Box<SharpeRatio> = Box::new(SharpeRatio { rf: 0.02 });
        // Portfolio definition and optimization
        let mut portfolio: Portfolio = Portfolio::build(assets, None, None, None, performance_metric).unwrap();
        let frontier: EfficientFrontier = portfolio.efficient_frontier(30, Some(1_000), Some(10_000)).unwrap();
        for point in &frontier.frontier {
            assert!(point.performance_score <= frontier.max_performance.performance_score);
            assert!(frontier.min_std.std <= point.std)
        }
    }

    #[cfg(feature = "plotly")]
    #[test]
    #[ignore]
    fn unit_test_plot_efficient_frontier() -> () {
        use plotly::Plot;
        use crate::portfolio_applications::portfolio_composition::plot_efficient_frontier;
        // Portfolio parameters
        let sample_data: SampleData = SampleData::Portfolio;
        let (time, data) = sample_data.load_sample_data();
        let weight: f64 = 1.0 / data.len() as f64;
        let dummy_array: Array1<f64> = Array1::from_vec(vec![0.0; time.len()]);
        let mut assets: HashMap<String, Asset> = HashMap::<String, Asset>::new();
        for (k, v) in data.into_iter() {
            let hist_data: AssetHistData = AssetHistData::build(v, dummy_array.clone(), Time::new(dummy_array.clone())).unwrap();
            assets.insert(k, Asset { hist_data, weight, });
        }
        let performance_metric: Box<SharpeRatio> = Box::new(SharpeRatio { rf: 0.02 });
        // Portfolio definition and optimization
        let mut portfolio: Portfolio = Portfolio::build(assets, None, None, None, performance_metric).unwrap();
        let frontier: EfficientFrontier = portfolio.efficient_frontier(30, Some(1_000), Some(10_000)).unwrap();
        // Efficient frontier plot
        let plot: Plot = plot_efficient_frontier(frontier);
        plot.show();
    }
}