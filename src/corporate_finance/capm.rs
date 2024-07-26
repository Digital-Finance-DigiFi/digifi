use std::io::Error;
use std::collections::HashMap;
use ndarray::{Array1, Array2};
use crate::utilities::{compare_array_len, input_error, data_error, some_or_error, shape_error_to_error};
use crate::statistics::{covariance, linear_regression};


#[derive(Clone)]
pub enum CAPMType {
    Standard,
    ThreeFactorFamaFrench,
    FiveFactorFamaFrench,
}


#[derive(Clone)]
pub enum CAPMSolutionType {
    LinearRegression,
    Covariance,
}


/// # Description
/// Parameters for the CAPM class.
pub struct CAPMData {
    /// Type of CAPM (i.e., STANDARD, THREE_FACTOR_FAMA_FRENCH, ot FIVE_FACTOR_FAMA_FRENCH)
    capm_type: CAPMType,
    /// Type of solution to use (i.e., COVARIANCE - works only for STANDARD CAPM, or LINEAR_REGRESSION)
    solution_type: CAPMSolutionType,
    /// Time series of market returns
    market_returns: Array1<f64>,
    /// Time series of risk-free rate of return
    rf: Array1<f64>,
    /// Difference in returns between a portfolio of "big" stocks and a portfolio of "small" stocks.
    smb: Option<Array1<f64>>,
    /// Difference in returns between a portfolio with "high" Book-to-Market ratio and a portfolio with "low" Book-to_market ratio.
    hml: Option<Array1<f64>>,
    /// Difference in returns between a portfolio with "strong" profitability and a portfolio with "weak" profitability.
    rmw: Option<Array1<f64>>,
    /// Difference in returns between a portfolio with "high" inner investment and a portfolio with "low" inner investment.
    cma: Option<Array1<f64>>,
}

impl CAPMData {
    /// # Description
    /// Creates a new CAPMParameters instance.
    /// 
    /// # Input
    /// - capm_type (CAPMType): Type of CAPM (i.e., STANDARD, THREE_FACTOR_FAMA_FRENCH, ot FIVE_FACTOR_FAMA_FRENCH)
    /// - solution_type (CAPMSolutionType): Type of solution to use (i.e., COVARIANCE - works only for STANDARD CAPM, or LINEAR_REGRESSION)
    /// - market_returns (np.ndarray): Time series of market returns
    /// - rf (np.ndarray): Time series of risk-free rate of return
    /// - SMB (Union[np.ndarray, None]): Difference in returns between a portfolio of "big" stocks and a portfolio of "small" stocks.
    /// - HML (Union[np.ndarray, None]): Difference in returns between a portfolio with "high" Book-to-Market ratio and a portfolio with "low" Book-to_market ratio.
    /// - RMW (Union[np.ndarray, None]): Difference in returns between a portfolio with "strong" profitability and a portfolio with "weak" profitability.
    /// - CMA (Union[np.ndarray, None]): Difference in returns between a portfolio with "high" inner investment and a portfolio with "low" inner investment.
    /// 
    /// # Panics
    /// - Panics if covariance solution type used with non-standard CAPM
    /// - Panics if the length of the market_returns array does not match any other array (i.e., rf, smb, hml, rmw, cma)
    pub fn new(capm_type: CAPMType, solution_type: CAPMSolutionType, market_returns: Array1<f64>, rf: Array1<f64>,
               smb: Option<Array1<f64>>, hml: Option<Array1<f64>>, rmw: Option<Array1<f64>>, cma: Option<Array1<f64>>) -> Result<Self, Error> {
        // Check solution type compatibility with the choice of the model.
        match capm_type {
            CAPMType::Standard => (),
            _ => {
                match solution_type {
                    CAPMSolutionType::Covariance => {
                        return Err(input_error("The covariance solution method is only available for the standard CAPM."));
                    },
                    CAPMSolutionType::LinearRegression => ()
                }
            }
        }
        // Validate array lengths
        compare_array_len(&market_returns, &rf, "market_returns", "rf")?;
        match capm_type {
            CAPMType::Standard => (),
            CAPMType::ThreeFactorFamaFrench => {
                CAPMData::validate_option_array(&market_returns, &smb, "smb")?;
                CAPMData::validate_option_array(&market_returns, &hml, "hml")?;
            },
            CAPMType::FiveFactorFamaFrench => {
                CAPMData::validate_option_array(&market_returns, &smb, "smb")?;
                CAPMData::validate_option_array(&market_returns, &hml, "hml")?;
                CAPMData::validate_option_array(&market_returns, &rmw, "rmw")?;
                CAPMData::validate_option_array(&market_returns, &cma, "cma")?;
            },
        }
        Ok(CAPMData { capm_type, solution_type, market_returns, rf, smb, hml, rmw, cma })
    }

    /// # Description
    /// Validation of length of arrays inside the Option enum.
    fn validate_option_array(market_returns: &Array1<f64>, option_array: &Option<Array1<f64>>, option_array_name: &str) -> Result<(), Error> {
        match option_array {
            Some(array) => {
                compare_array_len(&market_returns, array, "market_returns", option_array_name)?;
            },
            None => {
                return Err(data_error(format!("The argument {} must contain an array for the selected CAPM type.", option_array_name)));
            },
        }
        Ok(())
    }

    pub fn capm_type(&self) -> CAPMType {
        self.capm_type.clone()
    }

    pub fn solution_type(&self) -> CAPMSolutionType {
        self.solution_type.clone()
    }

    pub fn market_returns(&self) -> Array1<f64> {
        self.market_returns.clone()
    }

    pub fn rf(&self) -> Array1<f64> {
        self.rf.clone()
    }

    pub fn smb(&self) -> Option<Array1<f64>> {
        self.smb.clone()
    }

    pub fn hml(&self) -> Option<Array1<f64>> {
        self.hml.clone()
    }

    pub fn rmw(&self) -> Option<Array1<f64>> {
        self.rmw.clone()
    }

    pub fn cma(&self) -> Option<Array1<f64>> {
        self.cma.clone()
    }
}


/// # Description
/// CAPM, three-factor and five-factor Famma-French models.
/// 
/// Contains methods for finding asset beta and predicting expected asset returns with the given beta.
/// 
/// # Links
/// - Wikipedia: https://en.wikipedia.org/wiki/Capital_asset_pricing_model
/// - Original Source: N/A
pub struct CAPM {
    // Parameters for defining a CAPM instance
    capm_data: CAPMData,
}

impl CAPM {
    /// # Description
    /// Creates a new CAPM instance.
    /// 
    /// # Input
    /// - capm_data: Parameters for defining a CAPM instance
    pub fn new(capm_data: CAPMData) -> Self {
        CAPM { capm_data }
    }

    /// # Description
    /// Computes the expected return of an asset/project given the risk-free rate, expected market return, SMB, HML, RMW, CMA and their betas.
    /// 
    /// # Input
    /// - alpha: y-axis intersection of the CAPM
    /// - beta: Sensitivity of the asset with respect to premium market returns
    /// - beta_s: Sensitivity of the asset with respect to SMB returns
    /// - beta_h: Sensitivity of the asset with respect to HML returns
    /// - beta_r: Sensitivity of the asset with respect to RMW returns
    /// - beta_c: Sensitivity of the asset with respect to CMA returns
    /// 
    /// # Output
    /// - Array of asset returns
    /// 
    /// # LaTeX Formula
    /// - E[R_{A}] = R_{rf} + \\alpha + \\beta_{M}(E[R_{M}] - R_{rf}) + \\beta_{S}SMB + \\beta_{H}HML + \\beta_{R}RMW + \\beta_{C}CMA
    pub fn predict_asset_return(&self, alpha: f64, beta: f64, beta_s: f64, beta_h: f64, beta_r: f64, beta_c: f64) -> Result<Array1<f64>, Error> {
        let mut lin_reg: Array1<f64> = alpha + beta*(&self.capm_data.market_returns - self.capm_data.rf());
        match self.capm_data.capm_type {
            CAPMType::Standard => (),
            CAPMType::ThreeFactorFamaFrench => {
                lin_reg = lin_reg + beta_s*some_or_error(self.capm_data.smb(), "The argument smb is required for Three Factor Fama-French CAPM.")?
                                  + beta_h*some_or_error(self.capm_data.hml(), "The argument hml is required for Three Factor Fama-French CAPM.")?;
            },
            CAPMType::FiveFactorFamaFrench => {
                lin_reg = lin_reg + beta_s*some_or_error(self.capm_data.smb(), "The argument smb is required for Five Factor Fama-French CAPM.")?
                                  + beta_h*some_or_error(self.capm_data.hml(), "The argument hml is required for Five Factor Fama-French CAPM.")?;
                lin_reg = lin_reg + beta_r*some_or_error(self.capm_data.rmw(), "The argument rmw is required for Five Factor Fama-French CAPM.")?
                                  + beta_c*some_or_error(self.capm_data.cma(), "The argument cma is required for Five Factor Fama-French CAPM.")?;
            },
        }
        Ok(lin_reg)
    }

    fn train(&self, asset_returns: &Array1<f64>) -> Result<HashMap<String, f64>, Error> {
        let reg_params: HashMap<String, f64>;
        let mut data_vec: Vec<Vec<f64>> = vec![(self.capm_data.market_returns() - self.capm_data.rf()).to_vec()];
        data_vec.push(vec![1.0; data_vec[0].len()]);
        match self.capm_data.capm_type {
            CAPMType::Standard => {
                let data_matrix: Array2<f64> = shape_error_to_error(Array2::from_shape_vec((2, asset_returns.len()), data_vec.into_iter().flatten().collect()))?;
                let params: Array1<f64> = linear_regression(&data_matrix.t().to_owned(), &asset_returns)?;
                reg_params = HashMap::from([(String::from("beta"), params[0]), (String::from("alpha"), params[1])]);
            },
            CAPMType::ThreeFactorFamaFrench => {
                data_vec.push(some_or_error(self.capm_data.smb(), "The argument smb is required for Three Factor Fama-French CAPM.")?.to_vec());
                data_vec.push(some_or_error(self.capm_data.hml(), "The argument hml is required for Three Factor Fama-French CAPM.")?.to_vec());
                let data_matrix: Array2<f64> = shape_error_to_error(Array2::from_shape_vec((4, asset_returns.len()), data_vec.into_iter().flatten().collect()))?;
                let params: Array1<f64> = linear_regression(&data_matrix.t().to_owned(), &asset_returns)?;
                reg_params = HashMap::from([(String::from("beta"), params[0]), (String::from("alpha"), params[1]),
                                            (String::from("beta_s"), params[2]), (String::from("beta_h"), params[3])]);
            },
            CAPMType::FiveFactorFamaFrench => {
                data_vec.push(some_or_error(self.capm_data.smb(), "The argument smb is required for Five Factor Fama-French CAPM.")?.to_vec());
                data_vec.push(some_or_error(self.capm_data.hml(), "The argument hml is required for Five Factor Fama-French CAPM.")?.to_vec());
                data_vec.push(some_or_error(self.capm_data.rmw(), "The argument rmw is required for Five Factor Fama-French CAPM.")?.to_vec());
                data_vec.push(some_or_error(self.capm_data.cma(), "The argument cma is required for Five Factor Fama-French CAPM.")?.to_vec());
                let data_matrix: Array2<f64> = shape_error_to_error(Array2::from_shape_vec((6, asset_returns.len()), data_vec.into_iter().flatten().collect()))?;
                let params: Array1<f64> = linear_regression(&data_matrix.t().to_owned(), &asset_returns)?;
                reg_params = HashMap::from([(String::from("beta"), params[0]), (String::from("alpha"), params[1]),
                                            (String::from("beta_s"), params[2]), (String::from("beta_h"), params[3]),
                                            (String::from("beta_r"), params[4]), (String::from("beta_c"), params[5])]);
            },
        }
        Ok(reg_params)
    }

    /// # Description
    /// Finds the values of parameters alpha and betas (if COVARIANCE solution type is used, only beta is returned).
    /// 
    /// # Input
    /// - asset_returns: Array of asset returns
    /// 
    /// # Output
    /// - alpha: y-axis intersection of the CAPM model
    /// - beta: Sensitivity of the asset with respect to premium market returns
    /// - beta_s: Sensitivity of the asset with respect to SMB returns
    /// - beta_h: Sensitivity of the asset with respect to HML returns
    /// - beta_r: Sensitivity of the asset with respect to RMW returns
    /// - beta_c: Sensitivity of the asset with respect to CMA returns
    pub fn get_parameters(&self, asset_returns: Array1<f64>) -> Result<HashMap<String, f64>, Error> {
        compare_array_len(&asset_returns, &self.capm_data.market_returns, "asset_returns", "market_returns")?;
        if let CAPMSolutionType::Covariance = self.capm_data.solution_type {
            let numerator: f64 = covariance(&asset_returns, &self.capm_data.market_returns, 0)?;
            let denominator: f64 = covariance(&self.capm_data.market_returns, &self.capm_data.market_returns, 0)?;
            Ok(HashMap::from([(String::from("beta"), numerator/denominator)]))
        } else {
            self.train(&asset_returns)
        }
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_capm_get_parameters() -> () {
        use crate::utilities::SampleData;
        use crate::corporate_finance::capm::{CAPMData, CAPMType, CAPMSolutionType, CAPM};
        let sample: SampleData = SampleData::CAPM; 
        let (_, sample_data) = sample.load_sample_data();
        let capm_data: CAPMData = CAPMData::new(CAPMType::FiveFactorFamaFrench, CAPMSolutionType::LinearRegression,
                                                sample_data.get("market").unwrap().clone(), sample_data.get("rf").unwrap().clone(),
                                                Some(sample_data.get("smb").unwrap().clone()), Some(sample_data.get("hml").unwrap().clone()),
                                                Some(sample_data.get("rmw").unwrap().clone()), Some(sample_data.get("cma").unwrap().clone())).unwrap();
        let capm: CAPM = CAPM::new(capm_data);
        let params: HashMap<String, f64> = capm.get_parameters(sample_data.get("aapl").unwrap().clone()).unwrap();
        // The results were found using LinearRegression from sklearn
        assert!((params.get("alpha").unwrap() - 0.013530149403422963).abs() < TEST_ACCURACY);
        assert!((params.get("beta").unwrap() - 1.37731033).abs() < TEST_ACCURACY);
        assert!((params.get("beta_s").unwrap() - -0.38490771).abs() < TEST_ACCURACY);
        assert!((params.get("beta_h").unwrap() - -0.58771487).abs() < TEST_ACCURACY);
        assert!((params.get("beta_r").unwrap() - 0.11692186).abs() < TEST_ACCURACY);
        assert!((params.get("beta_c").unwrap() - 0.4192746).abs() < TEST_ACCURACY);
    }
}