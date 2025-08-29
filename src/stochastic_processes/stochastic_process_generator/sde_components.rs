use ndarray::Array1;
use crate::error::DigiFiError;
use crate::statistics::{continuous_distributions::ContinuousUniformDistribution, discrete_distributions::PoissonDistribution};
use crate::random_generators::{RandomGenerator, generator_algorithms::inverse_transform, standard_normal_generators::StandardNormalInverseTransform};


pub trait CustomSDEComponent {

    /// Custom SDE component function.
    ///
    /// # Input
    /// - `n_paths`: Number of simulation paths
    /// - `stochastic_values`: Stochastic process values
    /// - `t`: Current time step
    /// - `dt`: Time step increment
    fn comp_func(&self, n_paths: usize, stochastic_values: &Array1<f64>, t: f64, dt: f64) -> Result<Array1<f64>, DigiFiError>;

    /// Validates a custom SDE component function ensuring it meets computational requirements.
    ///
    /// # Input
    /// - `n_paths`: Number of simulation paths
    ///
    /// # Errors
    /// - Returns an error if the custom function does not return an array of the same length as there are number of paths.
    fn validate(&self, n_paths: usize) -> Result<(), DigiFiError> {
        let cont_uni_dist: ContinuousUniformDistribution = ContinuousUniformDistribution::build(0.0, 1.0)?;
        let sample: Array1<f64> = inverse_transform(&cont_uni_dist, 2)?;
        if self.comp_func(n_paths, &Array1::from_vec(vec![1.0; n_paths]), sample[0], sample[1])?.len() != n_paths {
            return Err(DigiFiError::ValidationError {
                title: "Custom SDE Component".to_owned(),
                details: "Custom function does not produce an array of the same size as the defined number of paths.".to_owned(),
            });
        }
        Ok(())
    }
}


/// An SDE component that is used in either drift or diffusion terms of the SDE.
pub enum SDEComponent {
    /// Generates a constant array representing a linear SDE component.
    /// 
    /// # LaTeX Formula
    /// - f(t) = a \\quad a\\in\\mathbb(R)
    Linear {
        /// Constant linear coefficient (i.e., the gradient of the linear function)
        a: f64,
    },
    /// Generates an array representing a quadratic time-dependent SDE component.
    ///
    /// # LaTeX Formula
    /// - f(t) = 2at + b
    QuadraticTime {
        /// Coefficient for the time-dependent term in quadratic gradient
        a: f64,
        /// Constant term in quadratic gradient
        b: f64,
    },
    /// Generates an array representing a power-law stochastic SDE component.
    ///
    /// # LaTeX Formula
    /// - f(X_{t}) = aX^{power}_{t}
    PowerStochastic {
        /// Scaling coefficient
        a: f64,
        /// Power-law exponent
        power: f64,
    },
    /// Generate a term that converges to a specific final value (e.g., Brownian Bridge Drift -\\frac{b - a}{T - t}).
    ///
    /// # LaTeX Formula
    /// - f(t) = \\frac{b - a}{T - t}
    ConvergenceToValue {
        /// Final time step
        final_time: f64,
        /// Initial process value
        a: f64,
        /// Final process value
        b: f64,
    },
    /// Generates a term that regresses to a trend value (e.g., Ornstein-Uhlenbeck Process Drift - \\alpha(\\mu - S_{t})).
    ///
    /// # LaTeX Formula
    /// - f(X_{t}) = \\textit{scale} * (\\textit{trend} - X_{t})
    RegressionToTrend {
        /// Scaling coefficient
        scale: f64,
        /// Average trend value
        trend: f64,
    },
    Custom { f: Box<dyn CustomSDEComponent> },
}

impl SDEComponent {

    /// Validates the parameters of the SDE component type.
    ///
    /// # Input
    /// - `n_paths`: Number of simulation paths
    ///
    /// # Errors
    /// - Returns an error if in convergence-to-value component the argument `b` is smaller or equal to the argument `a`.
    /// - Returns an error if in convergence-to-value component the argument `final_time` is non-positive.
    /// - Returns an error if the custom function does not return an array of the same length as there are number of paths.
    pub fn validate(&self, n_paths: usize) -> Result<(), DigiFiError> {
        let error_title: String = String::from("SDE Component Function Type");
        match &self {
            SDEComponent::ConvergenceToValue { final_time, a, b } => {
                if b <= a {
                    return Err(DigiFiError::ParameterConstraint {
                        title: error_title,
                        constraint: "The argument `b` must be larger that the argument `a`.".to_owned(),
                    });
                }
                if final_time <= &0.0 {
                    return Err(DigiFiError::ParameterConstraint {
                        title: error_title,
                        constraint: "The argument `final_time` must be positive.".to_owned(),
                    });
                }
            },
            SDEComponent::Custom { f } => { f.validate(n_paths)?; },
            _ => (),
        }
        Ok(())
    }

    /// Computes SDE component values for a given component type using the specified parameters and stochastic values.
    ///
    /// # Input
    /// - `n_paths`: Number of simulation paths
    /// - `stochastic_values`: Stochastic process values
    /// - `t`: Current time step
    /// - `dt`: Time step increment
    ///
    /// # Output
    /// - Array of computed SDE component values
    ///
    /// # Errors
    /// - Returns an error if the argument `stochastic_values` has length that is not same as the value of `n_paths`.
    pub fn get_component_values(&self, n_paths: usize, stochastic_values: &Array1<f64>, t: f64, dt: f64) -> Result<Array1<f64>, DigiFiError> {
        if stochastic_values.len() != n_paths {
            return Err(DigiFiError::ParameterConstraint {
                title: "SDE Component".to_owned(),
                constraint: "The argument `stochastic_values` must have the length of `n_paths`.".to_owned(),
            });
        }
        match &self {
            SDEComponent::Linear { a } => {
                Ok(Array1::from_vec(vec![*a; n_paths]))
            },
            SDEComponent::QuadraticTime { a, b } => {
                Ok(Array1::from_vec(vec![2.0*a + b; n_paths]))
            },
            SDEComponent::PowerStochastic { a, power } => {
                Ok(*a * stochastic_values.map(|v| v.powf(*power) ))
            },
            SDEComponent::ConvergenceToValue { final_time, a, b } => {
                Ok(Array1::from_vec(vec![(b - a) / (final_time - t); n_paths]))
            },
            SDEComponent::RegressionToTrend { scale, trend } => {
                Ok(*scale * (*trend - stochastic_values))
            },
            SDEComponent::Custom { f } => {
                f.comp_func(n_paths, stochastic_values, t, dt)
            },
        }
    }
}


pub trait CustomNoise {

    /// Custom noise function.
    ///
    /// # Input
    /// - `n_paths`: Number of simulation paths
    /// - `dt`: Time step increment
    fn noise_func(&self, n_paths: usize, dt: f64) -> Result<Array1<f64>, DigiFiError>;

    /// Validates a custom noise function ensuring it meets computational requirements.
    ///
    /// # Errors
    /// - Returns an error if the custom function does not return an array of the same length as there are number of paths.
    fn validate(&self, n_paths: usize) -> Result<(), DigiFiError> {
        let cont_uni_dist: ContinuousUniformDistribution = ContinuousUniformDistribution::build(0.0, 1.0)?;
        let dt: f64 = inverse_transform(&cont_uni_dist, 2)?[0];
        if self.noise_func(n_paths, dt)?.len() != n_paths {
            return Err(DigiFiError::ValidationError {
                title: "Custom Noise Function".to_owned(),
                details: "Custom function does not produce an array of the same size as the defined number of paths.".to_owned(),
            });
        }
        Ok(())
    }
}


/// Type of noise function to use in the diffusion term of the stochastic process.
pub enum Noise {
    /// Generates standard Gaussian white noise with mean 0 and variance 1.
    ///
    /// # LaTeX Formula
    /// - \\mathcal{N}(0, 1)
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/White_noise>
    /// - Original Source: N/A
    StandardWhiteNoise,
    /// Generates increments of the Wiener process, suitable for simulating Brownian motion.
    ///
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Wiener_process>
    /// - Original Source: N/A
    WeinerProcess,
    Custom { f: Box<dyn CustomNoise> },
}

impl Noise {

    /// Validates the parameters of the noise type.
    ///
    /// # Input
    /// - `n_paths`: Number of simulation paths
    ///
    /// # Errors
    /// - Returns an error if the custom function does not return an array of the same length as there are number of paths.
    pub fn validate(&self, n_paths: usize) -> Result<(), DigiFiError> {
        match &self {
            Noise::Custom { f } => { f.validate(n_paths)?; },
            _ => (),
        }
        Ok(())
    }

    /// Computes noise for the diffusion term.
    ///
    /// # Input
    /// - `n_paths`: Number of simulation paths
    /// - `dt`: Time step increment
    ///
    /// # Output
    /// - An array of noises for each path at a time step.
    pub fn get_noise(&self, n_paths: usize, dt: f64) -> Result<Array1<f64>, DigiFiError> {
        match &self {
            Noise::StandardWhiteNoise => {
                StandardNormalInverseTransform::new_shuffle(n_paths)?.generate()
            },
            Noise::WeinerProcess => {
                Ok(dt.sqrt() * StandardNormalInverseTransform::new_shuffle(n_paths)?.generate()?)
            },
            Noise::Custom { f } => { f.noise_func(n_paths, dt) }
        }
    }
}


pub trait CustomJump {

    /// Custom jump function.
    ///
    /// # Input
    /// - `dt`:Time step increment
    fn jump_func(&self, n_paths: usize, dt: f64) -> Result<Array1<f64>, DigiFiError>;

    /// Validates a custom jump function ensuring it meets computational requirements.
    ///
    /// # Errors
    /// - Returns an error if the custom function does not return an array of the same length as there are number of paths.
    fn validate(&self, n_paths: usize) -> Result<(), DigiFiError> {
        let cont_uni_dist: ContinuousUniformDistribution = ContinuousUniformDistribution::build(0.0, 1.0)?;
        let dt: f64 = inverse_transform(&cont_uni_dist, 2)?[0];
        if self.jump_func(n_paths, dt)?.len() != n_paths {
            return Err(DigiFiError::ValidationError {
                title: "Custom Jump Function".to_owned(),
                details: "Custom function does not produce an array of the same size as the defined number of paths.".to_owned(),
            });
        }
        Ok(())
    }
}


/// Represents the jump term in a discrete-time SDE. The jump term accounts for sudden, discontinuous changes in the value of the process.
pub enum Jump {
    NoJumps,
    /// Generates jumps according to a compound Poisson process with normally distributed jump sizes.
    ///
    /// # LaTeX Formula
    /// - \\textit{Jumps} = \\textit{Pois}(\\lambda \\cdot dt) \\cdot (\\mu_{j} + \\sigma_{j} \\cdot \\sqrt{\\text{Jumps}} \\cdot \\mathcal{N}(0,1))
    /// where \\lambda is the jump rate, \\mu_{j} is the mean jump size, and \\sigma_{j} is the jump size volatility.
    CompoundPoissonNormal {
        /// Jump rate (0 < lambda_j)
        lambda_j: f64,
        /// Average jump magnitude
        mu_j: f64,
        /// Standard deviation of jump magnitude
        sigma_j: f64
    },
    /// Generates jumps according to a compound Poisson process with bilateral exponential jump sizes.
    ///
    /// # LaTeX Formula
    /// - \\textit{Jumps Frequency} = Pois(\\lambda dt)
    /// - \\textit{Assymetric Double Exponential RV} = \\mathbb{1}_{p\\leq U(0,1)}*(-\\frac{1}{\\eta_{u}} * ln(\\frac{1-U(0,1)}{p})) + \\mathbb{1}_{U(0,1)<p}*(\\frac{1}{\\eta_{d}} * ln(\\frac{U(0,1)}{1-p}))
    /// - \\textit{Jumps Distribution} = (e^{\\textit{Assymetric Double Exponential RV}} - 1) * \\textit{Jumps Frequency}
    CompoundPoissonBilateral {
        /// Jump rate (0 < lambda_j)
        lambda_j: f64,
        /// Probability of a jump up (0<=p<=1)
        p: f64,
        /// Scaling of jump down
        eta_d: f64,
        /// Scaling of jump up
        eta_u: f64,
    },
    CustomJump {f: Box<dyn CustomJump>},
}

impl Jump {

    /// Validates the parameters of the jump function type.
    ///
    /// # Input
    /// - `n_paths`: Number of simulation paths
    ///
    /// # Errors
    /// - Returns an error if the argument `lambda_j` of the compound Poisson normal jump is negative.
    /// - Returns an error if the argument `lambda_j` of the compound Poisson bilateral jump be negative.
    /// - Returns an error if the rgument `p` of the compound Poisson bilateral jump is not in the range \[0, 1\].
    pub fn validate(&self, n_paths: usize) -> Result<(), DigiFiError> {
        let error_title: String = String::from("Jump Type");
        match &self {
            Jump::NoJumps => Ok(()),
            Jump::CompoundPoissonNormal { lambda_j, .. } => {
                if lambda_j <= &0.0 {
                    return Err(DigiFiError::ParameterConstraint {
                        title: error_title,
                        constraint: "The argument `lambda_j` of the compound Poisson normal jump must be positive.".to_owned(),
                    });
                }
                Ok(())
            },
            Jump::CompoundPoissonBilateral { lambda_j, p, .. } => {
                if lambda_j <= &0.0 {
                    return Err(DigiFiError::ParameterConstraint {
                        title: error_title,
                        constraint: "The argument `lambda_j` of the compound Poisson bilateral jump must be positive.".to_owned(),
                    });
                }
                if (p < &0.0) || (&1.0 < p) {
                    return Err(DigiFiError::ParameterConstraint {
                        title: error_title,
                        constraint: "The argument `p` of the compound Poisson bilateral jump must be in the range `[0, 1]`.".to_owned(),
                    });
                }
                Ok(())
            },
            Jump::CustomJump { f } => { f.validate(n_paths) },
        }
    }

    /// Computes the jump term for each path in the stochastic process at a given time.
    ///
    /// # Input
    /// - `n_paths`: Number of simulation paths
    /// - `dt`: Time step increment
    pub fn get_jumps(&self, n_paths: usize, dt: f64) -> Result<Array1<f64>, DigiFiError> {
        match &self {
            Jump::NoJumps => { Ok(Array1::from_vec(vec![0.0; n_paths])) },
            Jump::CompoundPoissonNormal { lambda_j, mu_j, sigma_j } => {
                let pois_dist: PoissonDistribution = PoissonDistribution::build(lambda_j*dt)?;
                let dp: Array1<f64> = inverse_transform(&pois_dist, n_paths)?;
                Ok(*mu_j * &dp + *sigma_j * dp.map(|v| v.sqrt() ) * StandardNormalInverseTransform::new_shuffle(n_paths)?.generate()?)
            },
            Jump::CompoundPoissonBilateral { lambda_j, p, eta_d, eta_u } => {
                let pois_dist: PoissonDistribution = PoissonDistribution::build(lambda_j*dt)?;
                let dp: Array1<f64> = inverse_transform(&pois_dist, n_paths)?;
                // Assymetric double exponential distribution
                let cont_uni_dist: ContinuousUniformDistribution = ContinuousUniformDistribution::build(0.0, 1.0)?;
                let u: Array1<f64> = inverse_transform(&cont_uni_dist, n_paths)?;
                let mut y: Array1<f64> = Array1::from_vec(vec![0.0; n_paths]);
                for i in 0..u.len() {
                    if p <= &u[i] {
                        y[i] = -1.0 / eta_u * ((1.0 - u[i]) / p).ln()
                    } else {
                        y[i] = 1.0 / eta_d * (u[i] / (1.0 - p)).ln()
                    }
                }
                // Distribution of jumps
                Ok(y.map(|v| v.exp() - 1.0 ) * dp)
            },
            Jump::CustomJump { f } => { f.jump_func(n_paths, dt) },
        }
    }
}