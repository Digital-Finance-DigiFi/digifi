use ndarray::{Array1, Array2};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::error::{DigiFiError, ErrorTitle};
use crate::utilities::compare_len;
use crate::statistics::{covariance, linear_regression};


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Parameters of CAPM model.
pub struct CAPMParams {
    /// y-axis intersection of the CAPM model
    pub alpha: Option<f64>,
    /// Sensitivity of the asset with respect to premium market returns
    pub beta: f64,
    /// Sensitivity of the asset with respect to SMB returns
    pub beta_s: Option<f64>,
    /// Sensitivity of the asset with respect to HML returns
    pub beta_h: Option<f64>,
    /// Sensitivity of the asset with respect to RMW returns
    pub beta_r: Option<f64>,
    /// Sensitivity of the asset with respect to CMA returns
    pub beta_c: Option<f64>,
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CAPMType {
    Standard,
    ThreeFactorFamaFrench {
        /// Difference in returns between a portfolio of "big" stocks and a portfolio of "small" stocks.
        smb: Array1<f64>,
        /// Difference in returns between a portfolio with "high" Book-to-Market ratio and a portfolio with "low" Book-to_market ratio.
        hml: Array1<f64>,
    },
    FiveFactorFamaFrench {
        /// Difference in returns between a portfolio of "big" stocks and a portfolio of "small" stocks.
        smb: Array1<f64>,
        /// Difference in returns between a portfolio with "high" Book-to-Market ratio and a portfolio with "low" Book-to_market ratio.
        hml: Array1<f64>,
        /// Difference in returns between a portfolio with "strong" profitability and a portfolio with "weak" profitability.
        rmw: Array1<f64>,
        /// Difference in returns between a portfolio with "high" inner investment and a portfolio with "low" inner investment.
        cma: Array1<f64>,
    },
}

impl CAPMType {

    /// Validates data inside each of the variants.
    ///
    /// # Errors
    /// - Returns an error if the length of any of the arrays do not match in length (i.e., `smb`, `hml`, `rmw`, `cma`)
    pub fn self_validate(&self) -> Result<(), DigiFiError> {
        match self {
            CAPMType::Standard => Ok(()),
            CAPMType::ThreeFactorFamaFrench { smb, hml } => {
                compare_len(&smb.iter(), &hml.iter(), "smb", "hml")
            },
            CAPMType::FiveFactorFamaFrench { smb, hml, rmw, cma } => {
                compare_len(&smb.iter(), &hml.iter(), "smb", "hml")?;
                compare_len(&smb.iter(), &rmw.iter(), "smb", "rmw")?;
                compare_len(&smb.iter(), &cma.iter(), "smb", "cma")
            },
        }
    }
}


#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Type of solution used for computing the parameters of the CAPM.
///
/// Note: `Covariance` solution type is only available for the `Standard` CAPM.
pub enum CAPMSolutionType {
    LinearRegression,
    Covariance,
}


#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// CAPM, three-factor and five-factor Famma-French models.
/// 
/// Contains methods for finding asset beta and predicting expected asset returns with the given beta.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Capital_asset_pricing_model>
/// - Original Source: N/A
///
/// # Examples
///
/// ```rust
/// use digifi::utilities::TEST_ACCURACY;
/// use digifi::corporate_finance::{CAPMParams, CAPMSolutionType, CAPMType, CAPM};
///
/// #[cfg(feature = "sample_data")]
/// fn test_capm() -> () {
///     use digifi::utilities::SampleData;
///
///     let sample: SampleData = SampleData::CAPM; 
///     let (_, mut sample_data) = sample.load_sample_data();
///     let capm_type: CAPMType = CAPMType::FiveFactorFamaFrench {
///         smb: sample_data.remove("SMB").unwrap(), hml: sample_data.remove("HML").unwrap(),
///         rmw: sample_data.remove("RMW").unwrap(), cma: sample_data.remove("CMA").unwrap(),
///     };
///     let solution_type: CAPMSolutionType = CAPMSolutionType::LinearRegression;
///
///     let capm: CAPM = CAPM::build(sample_data.remove("Market").unwrap(), sample_data.remove("RF").unwrap(), capm_type, solution_type).unwrap();
///     let params: CAPMParams = capm.get_parameters(&sample_data.remove("Stock Returns").unwrap()).unwrap();
///
///     // The results were found using LinearRegression from sklearn
///     assert!((params.alpha.unwrap() - 0.01353015).abs() < TEST_ACCURACY);
///     assert!((params.beta - 1.37731033).abs() < TEST_ACCURACY);
///     assert!((params.beta_s.unwrap() - -0.38490771).abs() < TEST_ACCURACY);
///     assert!((params.beta_h.unwrap() - -0.58771487).abs() < TEST_ACCURACY);
///     assert!((params.beta_r.unwrap() - 0.11692186).abs() < TEST_ACCURACY);
///     assert!((params.beta_c.unwrap() - 0.4192746).abs() < TEST_ACCURACY);
/// }
/// ```
pub struct CAPM {
    /// Time series of market returns
    market_returns: Array1<f64>,
    /// Time series of risk-free rate of return
    rf: Array1<f64>,
    /// Type of CAPM (i.e., `Standard`, `ThreeFactorFamaFrench`, ot `FiveFactorFamaFrench`)
    capm_type: CAPMType,
    /// Type of solution to use (i.e., `Covariance` - works only for the `Standard` CAPM, or `LinearRegression`)
    solution_type: CAPMSolutionType,
}

impl CAPM {
    /// Creates a new `CAPM` instance.
    /// 
    /// # Input
    /// - `market_returns`: Time series of market returns
    /// - `rf`: Time series of risk-free rate of return
    /// - `capm_type`: Type of CAPM (i.e., `Standard`, `ThreeFactorFamaFrench`, ot `FiveFactorFamaFrench`)
    /// - `solution_type`: Type of solution to use (i.e., `Covariance` - works only for the `Standard` CAPM, or `LinearRegression`)
    ///
    /// # Errors
    /// - Returns an error if the length of the market_returns array does not match any other array (i.e., `rf`, `smb`, `hml`, `rmw`, `cma`).
    /// - Returns an error if covariance solution type used with non-standard CAPM.
    pub fn build(market_returns: Array1<f64>, rf: Array1<f64>, capm_type: CAPMType, solution_type: CAPMSolutionType) -> Result<Self, DigiFiError> {
        // Cross-validate the lengths of all arrays
        compare_len(&market_returns.iter(), &rf.iter(), "market_returns", "rf")?;
        capm_type.self_validate()?;
        match &capm_type {
            CAPMType::Standard => (),
            CAPMType::ThreeFactorFamaFrench { smb, .. } => {
                compare_len(&market_returns.iter(), &smb.iter(), "market_returns", "smb")?;
            },
            CAPMType::FiveFactorFamaFrench { smb, ..} => {
                compare_len(&market_returns.iter(), &smb.iter(), "market_returns", "smb")?;
            },
        };
        // Check solution type compatibility with the choice of the model.
        match capm_type {
            CAPMType::Standard => (),
            _ => {
                match solution_type {
                    CAPMSolutionType::Covariance => {
                        return Err(DigiFiError::ValidationError { title: "CAPM".to_owned(), details: "The `covariance` solution method is only available for the standard CAPM.".to_owned(), });
                    },
                    CAPMSolutionType::LinearRegression => (),
                }
            }
        }
        Ok(CAPM { market_returns, rf, capm_type, solution_type })
    }

    /// Computes the expected return premium of an asset/project given the risk-free rate, expected market return, SMB, HML, RMW, CMA and their betas.
    /// 
    /// # Input
    /// - `alpha`: y-axis intersection of the CAPM
    /// - `beta`: Sensitivity of the asset with respect to premium market returns
    /// - `beta_s`: Sensitivity of the asset with respect to SMB returns
    /// - `beta_h`: Sensitivity of the asset with respect to HML returns
    /// - `beta_r`: Sensitivity of the asset with respect to RMW returns
    /// - `beta_c`: Sensitivity of the asset with respect to CMA returns
    /// 
    /// # Output
    /// - Array of risk premiums (i.e., asset return premiums)
    /// 
    /// # LaTeX Formula
    /// - E[R_{A}] - R_{rf} = \\alpha + \\beta_{M}(E[R_{M}] - R_{rf}) + \\beta_{S}SMB + \\beta_{H}HML + \\beta_{R}RMW + \\beta_{C}CMA
    pub fn predict_asset_return(&self, alpha: f64, beta: f64, beta_s: f64, beta_h: f64, beta_r: f64, beta_c: f64) -> Result<Array1<f64>, DigiFiError> {
        let mut lin_reg: Array1<f64> = alpha + beta * (&self.market_returns - &self.rf);
        match &self.capm_type {
            CAPMType::Standard => (),
            CAPMType::ThreeFactorFamaFrench { smb, hml } => {
                lin_reg = lin_reg + (beta_s * smb) + (beta_h * hml);
            },
            CAPMType::FiveFactorFamaFrench { smb, hml, rmw, cma } => {
                lin_reg = lin_reg + (beta_s * smb) + (beta_h * hml) + (beta_r * rmw) + (beta_c * cma);
            },
        }
        Ok(lin_reg)
    }

    fn train(&self, risk_premium: &Array1<f64>) -> Result<CAPMParams, DigiFiError> {
        let mut data_vec: Vec<Vec<f64>> = vec![(&self.market_returns - &self.rf).to_vec()];
        data_vec.push(vec![1.0; data_vec[0].len()]);
        match &self.capm_type {
            CAPMType::Standard => {
                let data_matrix: Array2<f64> = Array2::from_shape_vec((2, risk_premium.len()), data_vec.into_iter().flatten().collect())?;
                let params: Array1<f64> = linear_regression(&data_matrix.t().to_owned(), &risk_premium)?;
                self.unstack_parameters(&mut params.into_iter())

            },
            CAPMType::ThreeFactorFamaFrench { smb, hml } => {
                data_vec.push(smb.to_vec());
                data_vec.push(hml.to_vec());
                let data_matrix: Array2<f64> = Array2::from_shape_vec((4, risk_premium.len()), data_vec.into_iter().flatten().collect())?;
                let params: Array1<f64> = linear_regression(&data_matrix.t().to_owned(), &risk_premium)?;
                self.unstack_parameters(&mut params.into_iter())
            },
            CAPMType::FiveFactorFamaFrench { smb, hml, rmw, cma } => {
                data_vec.push(smb.to_vec());
                data_vec.push(hml.to_vec());
                data_vec.push(rmw.to_vec());
                data_vec.push(cma.to_vec());
                let data_matrix: Array2<f64> = Array2::from_shape_vec((6, risk_premium.len()), data_vec.into_iter().flatten().collect())?;
                let params: Array1<f64> = linear_regression(&data_matrix.t().to_owned(), &risk_premium)?;
                self.unstack_parameters(&mut params.into_iter())
            },
        }
    }

    /// Converts an iterator over parameters of linear regression model to `CAPMParams` instance.
    fn unstack_parameters(&self, params: &mut impl Iterator<Item = f64>) -> Result<CAPMParams, DigiFiError> {
        let mut reg_params: CAPMParams = CAPMParams { alpha: None, beta: 0.0, beta_s: None, beta_h: None, beta_r: None, beta_c: None };
        reg_params.beta = match params.next() {
            Some(v) => v,
            None => return Err(DigiFiError::Other { title: Self::error_title(), details: "No `beta` was found.".to_owned(), }),
        };
        reg_params.alpha = params.next();
        if let CAPMType::ThreeFactorFamaFrench { .. } | CAPMType::FiveFactorFamaFrench { .. } = &self.capm_type {
            reg_params.beta_s = params.next();
            reg_params.beta_h = params.next();
        }
        if let CAPMType::FiveFactorFamaFrench { .. } = &self.capm_type {
            reg_params.beta_r = params.next();
            reg_params.beta_c = params.next();
        }
        Ok(reg_params)
    }

    /// Finds the values of parameters alpha and betas (if `Covariance` solution type is used, only beta is returned).
    /// 
    /// # Input
    /// - `risk_premium`: Array of risk premiums (i.e., asset return premiums)
    /// 
    /// # Output
    /// - Parameters of the chosen CAPM model
    pub fn get_parameters(&self, risk_premium: &Array1<f64>) -> Result<CAPMParams, DigiFiError> {
        compare_len(&risk_premium.iter(), &self.market_returns.iter(), "risk_premium", "market_returns")?;
        match self.solution_type {
            CAPMSolutionType::Covariance => {
                let numerator: f64 = covariance(risk_premium, &self.market_returns, 0)?;
                let denominator: f64 = covariance(&self.market_returns, &self.market_returns, 0)?;
                Ok(CAPMParams { alpha: None, beta: numerator/denominator, beta_s: None, beta_h: None, beta_r: None, beta_c: None })
            },
            CAPMSolutionType::LinearRegression => {
                self.train(risk_premium)
            },
        }
    }
}

impl ErrorTitle for CAPM {
    fn error_title() -> String {
        String::from("CAPM")
    }
}


#[cfg(all(test, feature = "sample_data"))]
mod tests {
    use crate::utilities::{TEST_ACCURACY, sample_data::SampleData};
    use crate::corporate_finance::capm::{CAPMParams, CAPMSolutionType, CAPMType, CAPM};

    #[test]
    fn unit_test_capm_get_parameters() -> () {
        let sample: SampleData = SampleData::CAPM; 
        let (_, mut sample_data) = sample.load_sample_data();
        let capm_type: CAPMType = CAPMType::FiveFactorFamaFrench {
            smb: sample_data.remove("SMB").unwrap(), hml: sample_data.remove("HML").unwrap(),
            rmw: sample_data.remove("RMW").unwrap(), cma: sample_data.remove("CMA").unwrap(),
        };
        let solution_type: CAPMSolutionType = CAPMSolutionType::LinearRegression;
        let capm: CAPM = CAPM::build(sample_data.remove("Market").unwrap(), sample_data.remove("RF").unwrap(), capm_type, solution_type).unwrap();
        let params: CAPMParams = capm.get_parameters(&sample_data.remove("Stock Returns").unwrap()).unwrap();
        // The results were found using LinearRegression from sklearn
        assert!((params.alpha.unwrap() - 0.01353015).abs() < TEST_ACCURACY);
        assert!((params.beta - 1.37731033).abs() < TEST_ACCURACY);
        assert!((params.beta_s.unwrap() - -0.38490771).abs() < TEST_ACCURACY);
        assert!((params.beta_h.unwrap() - -0.58771487).abs() < TEST_ACCURACY);
        assert!((params.beta_r.unwrap() - 0.11692186).abs() < TEST_ACCURACY);
        assert!((params.beta_c.unwrap() - 0.4192746).abs() < TEST_ACCURACY);
    }
}