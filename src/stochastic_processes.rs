// Re-Exports
pub use self::standard_stochastic_models::{
    ArithmeticBrownianMotion, GeometricBrownianMotion, OrnsteinUhlenbeckProcess, BrownianBridge, FellerSquareRootProcess, FSRSimulationMethod,
};
pub use self::stochastic_volatility_models::{ConstantElasticityOfVariance, HestonStochasticVolatility, VarianceGammaProcess};
pub use self::jump_diffusion_models::{MertonJumpDiffusionProcess, KouJumpDiffusionProcess};


pub mod standard_stochastic_models;
pub mod stochastic_volatility_models;
pub mod jump_diffusion_models;
pub mod stochastic_process_generator;


use ndarray::Array1;
#[cfg(feature = "plotly")]
use plotly::{Plot, Trace, Layout, Scatter, layout::Axis};
use crate::error::DigiFiError;
#[cfg(feature = "plotly")]
use crate::utilities::compare_array_len;


pub trait StochasticProcess {

    /// # Description
    /// Updates the number of paths that the stochastic process will generate.
    /// 
    /// # Input
    /// - `n_paths`: Number of paths to generate
    fn update_n_paths(&mut self, n_paths: usize) -> ();

    /// # Description
    /// Returns the number of time steps in the stochastic process.
    fn get_n_steps(&self) -> usize;

    /// # Description
    /// Returns the final time step.
    fn get_t_f(&self) -> f64;

    /// # Description
    /// Paths, S, of the stochastic process.
    fn get_paths(&self) -> Result<Vec<Array1<f64>>, DigiFiError>;
}


#[cfg(feature = "plotly")]
/// # Description
/// Plots all generated paths of the stochastic process.
///
/// # Input
/// - `paths`: Vector of paths generated by a stochastic process
/// - `expected_path`: Expected path, E\[S\], of the stochastic process
///
/// # Output
/// - Plot of the paths of the stochastic process
///
/// # Errors
/// - Returns an error if the `paths` and `expected_paths` have arrays of different lengths.
///
/// # Examples
///
/// 1. Geometric Brownian motion paths simulation:
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::stochastic_processes::{StochasticProcess, GeometricBrownianMotion};
///
/// #[cfg(feature = "plotly")]
/// fn test_gbm_plot() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_stochastic_paths;
///
///     // Geometric brownian motion
///     let gbm: GeometricBrownianMotion = GeometricBrownianMotion::new(0.0, 0.2, 1_000, 200, 1.0, 100.0);
///     let paths: Vec<Array1<f64>> = gbm.get_paths().unwrap();
///     let expected_path: Array1<f64> = gbm.get_expectations();
///
///     // Paths plot
///     let plot: Plot = plot_stochastic_paths(&paths, Some(&expected_path)).unwrap();
///     plot.show()
/// }
/// ```
///
/// 2. Heston stochastic volatility paths simulation:
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::stochastic_processes::{StochasticProcess, HestonStochasticVolatility};
///
/// #[cfg(feature = "plotly")]
/// fn test_hsv_plot() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_stochastic_paths;
///
///     // Heston stochastic volatility
///     let hsv: HestonStochasticVolatility = HestonStochasticVolatility::new(0.1, 5.0, 0.07, 0.2, 0.0, 100, 200, 1.0, 100.0, 0.03);
///     let paths: Vec<Array1<f64>> = hsv.get_paths().unwrap();
///     let expected_path: Array1<f64> = hsv.get_expectations();
///
///     // Paths plot
///     let plot: Plot = plot_stochastic_paths(&paths, Some(&expected_path)).unwrap();
///     plot.show()
/// }
/// ```
///
/// 3. Kou jump diffusion paths simulation:
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::stochastic_processes::{StochasticProcess, KouJumpDiffusionProcess};
///
/// #[cfg(feature = "plotly")]
/// fn test_hsv_plot() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_stochastic_paths;
///
///     // Kou jump diffusion
///     let kjd: KouJumpDiffusionProcess = KouJumpDiffusionProcess::new(0.2, 0.3, 0.5, 9.0, 5.0, 0.5, 100, 200, 1.0, 100.0).unwrap();
///     let paths: Vec<Array1<f64>> = kjd.get_paths().unwrap();
///     let expected_path: Array1<f64> = kjd.get_expectations();
///
///     // Paths plot
///     let plot: Plot = plot_stochastic_paths(&paths, Some(&expected_path)).unwrap();
///     plot.show()
/// }
/// ```
///
/// 4. Custom SDE simulation plot (e.g., Arithmetic Brownian motion):
///
/// ```rust
/// use ndarray::Array1;
/// use digifi::stochastic_processes::{StochasticProcess, stochastic_process_generator::{SDE, SDEComponent, Noise, Jump}};
///
/// #[cfg(feature = "plotly")]
/// fn unit_test_sde_abm_plot() -> () {
///     use plotly::Plot;
///     use digifi::plots::plot_stochastic_paths;
///
///     // Parameter definition
///     let t_f: f64 = 1.0;
///     let s_0: f64 = 100.0;
///     let drift_component: SDEComponent = SDEComponent::Linear { a: 10.0 };
///     let diffusion_component: SDEComponent = SDEComponent::Linear { a: 0.4 };
///     let noise: Noise = Noise::WeinerProcess;
///     let jump: Jump = Jump::NoJumps;
///
///     // Arithmetic Brownian motion SDE definition
///     let sde: SDE = SDE::new(t_f, s_0, 200, 100, drift_component, diffusion_component, noise, jump).unwrap();
///     let paths: Vec<Array1<f64>> = sde.get_paths().unwrap();
///
///     // Paths plot
///     let plot: Plot = plot_stochastic_paths(&paths, None).unwrap();
///     plot.show()
/// }
/// ```
pub fn plot_stochastic_paths(paths: &Vec<Array1<f64>>, expected_path: Option<&Array1<f64>>) -> Result<Plot, DigiFiError> {
    let t: Array1<f64> = Array1::range(0.0, paths[0].len() as f64, 1.0);
    let mut traces: Vec<Box<dyn Trace>> = Vec::<Box<dyn Trace>>::new();
    match expected_path {
        Some(p) => {
            compare_array_len(&t, &p, "path", "expected_path")?;
            traces.push(Scatter::new(t.to_vec(), p.to_vec()).name("Expected Path"));
        },
        None => (),
    }
    for path in paths {
        traces.push(Scatter::new(t.to_vec(), path.to_vec()));
    }
    // Push expected path to be the last trace plotted
    match expected_path {
        Some(_) => traces.rotate_left(1),
        None => (),
    }
    let mut plot: Plot = Plot::new();
    plot.add_traces(traces);
    let x_axis: Axis = Axis::new().title("Time Step");
    let y_axis: Axis = Axis::new().title("Stochastic Process Value");
    let layout: Layout = Layout::new().title("<b>Stochastic Path Simulations</b>").x_axis(x_axis).y_axis(y_axis);
    plot.set_layout(layout);
    Ok(plot)
}


#[cfg(all(test, feature = "plotly"))]
mod tests {
    use ndarray::Array1;
    use plotly::Plot;
    use crate::stochastic_processes::{StochasticProcess, plot_stochastic_paths};

    #[test]
    fn unit_test_abm_plot() -> () {
        use crate::stochastic_processes::standard_stochastic_models::ArithmeticBrownianMotion;
        // Arithmetic Brownian motion
        let abm: ArithmeticBrownianMotion = ArithmeticBrownianMotion::new(1.0, 0.4, 100, 200, 1.0, 100.0);
        let paths: Vec<Array1<f64>> = abm.get_paths().unwrap();
        let expected_path: Array1<f64> = abm.get_expectations();
        // Paths plot
        let plot: Plot = plot_stochastic_paths(&paths, Some(&expected_path)).unwrap();
        plot.show()
    }

    #[test]
    fn unit_test_gbm_plot() -> () {
        use crate::stochastic_processes::standard_stochastic_models::GeometricBrownianMotion;
        // Geometric Brownian motion
        let gbm: GeometricBrownianMotion = GeometricBrownianMotion::new(0.0, 0.2, 1_000, 200, 1.0, 100.0);
        let paths: Vec<Array1<f64>> = gbm.get_paths().unwrap();
        let expected_path: Array1<f64> = gbm.get_expectations();
        // Paths plot
        let plot: Plot = plot_stochastic_paths(&paths, Some(&expected_path)).unwrap();
        plot.show()
    }

    #[test]
    fn unit_test_oup_plot() -> () {
        use crate::stochastic_processes::standard_stochastic_models::OrnsteinUhlenbeckProcess;
        // Ornstein-Uhlebeck process
        let oup: OrnsteinUhlenbeckProcess = OrnsteinUhlenbeckProcess::new(0.07, 0.1, 10.0, 100, 200, 1.0, 0.05, true);
        let paths: Vec<Array1<f64>> = oup.get_paths().unwrap();
        let expected_path: Array1<f64> = oup.get_expectations();
        // Paths plot
        let plot: Plot = plot_stochastic_paths(&paths, Some(&expected_path)).unwrap();
        plot.show()
    }

    #[test]
    fn unit_test_bb_plot() -> () {
        use crate::stochastic_processes::standard_stochastic_models::BrownianBridge;
        // Brownian bridge
        let bb: BrownianBridge = BrownianBridge::new(1.0, 2.0, 0.5, 100, 200, 1.0);
        let paths: Vec<Array1<f64>> = bb.get_paths().unwrap();
        let expected_path: Array1<f64> = bb.get_expectations();
        // Paths plot
        let plot: Plot = plot_stochastic_paths(&paths, Some(&expected_path)).unwrap();
        plot.show()
    }

    #[test]
    fn unit_test_fsrp_plot() -> () {
        use crate::stochastic_processes::standard_stochastic_models::{FellerSquareRootProcess, FSRSimulationMethod};
        // Feller square root process
        let fsrp: FellerSquareRootProcess = FellerSquareRootProcess::new(0.05, 0.265, 5.0, 100, 200,
                                                                         1.0, 0.03, FSRSimulationMethod::EulerMaruyama);
        let paths: Vec<Array1<f64>> = fsrp.get_paths().unwrap();
        let expected_path: Array1<f64> = fsrp.get_expectations();
        // Paths plot
        let plot: Plot = plot_stochastic_paths(&paths, Some(&expected_path)).unwrap();
        plot.show()
    }

    #[test]
    fn unit_test_cev_plot() -> () {
        use crate::stochastic_processes::stochastic_volatility_models::ConstantElasticityOfVariance;
        // Constant elasticity of variance
        let cev: ConstantElasticityOfVariance = ConstantElasticityOfVariance::new(1.0, 0.4, 0.5, 100, 200, 1.0, 100.0).unwrap();
        let paths: Vec<Array1<f64>> = cev.get_paths().unwrap();
        let expected_path: Array1<f64> = cev.get_expectations();
        // Paths plot
        let plot: Plot = plot_stochastic_paths(&paths, Some(&expected_path)).unwrap();
        plot.show()
    }

    #[test]
    fn unit_test_hsv_plot() -> () {
        use crate::stochastic_processes::stochastic_volatility_models::HestonStochasticVolatility;
        // Heston stochastic volatility
        let hsv: HestonStochasticVolatility = HestonStochasticVolatility::new(0.1, 5.0, 0.07, 0.2, 0.0, 100, 200, 1.0, 100.0, 0.03);
        let paths: Vec<Array1<f64>> = hsv.get_paths().unwrap();
        let expected_path: Array1<f64> = hsv.get_expectations();
        // Paths plot
        let plot: Plot = plot_stochastic_paths(&paths, Some(&expected_path)).unwrap();
        plot.show()
    }

    #[test]
    fn unit_test_mjd_plot() -> () {
        use crate::stochastic_processes::jump_diffusion_models::MertonJumpDiffusionProcess;
        // Merton jump diffusion
        let mjd: MertonJumpDiffusionProcess = MertonJumpDiffusionProcess::new(0.03, 0.2, -0.03, 0.1, 1.5, 100, 200, 1.0, 100.0);
        let paths: Vec<Array1<f64>> = mjd.get_paths().unwrap();
        let expected_path: Array1<f64> = mjd.get_expectations();
        // Paths plot
        let plot: Plot = plot_stochastic_paths(&paths, Some(&expected_path)).unwrap();
        plot.show()
    }

    #[test]
    fn unit_test_kjd_plot() -> () {
        use crate::stochastic_processes::jump_diffusion_models::KouJumpDiffusionProcess;
        // Kou jump diffusion
        let kjd: KouJumpDiffusionProcess = KouJumpDiffusionProcess::new(0.2, 0.3, 0.5, 9.0, 5.0, 0.5, 100, 200, 1.0, 100.0).unwrap();
        let paths: Vec<Array1<f64>> = kjd.get_paths().unwrap();
        let expected_path: Array1<f64> = kjd.get_expectations();
        // Paths plot
        let plot: Plot = plot_stochastic_paths(&paths, Some(&expected_path)).unwrap();
        plot.show()
    }

    #[test]
    fn unit_test_sde_abm_plot() -> () {
        use crate::stochastic_processes::stochastic_process_generator::{SDE, sde_components::{SDEComponent, Noise, Jump}};
        // Parameter definition
        let t_f: f64 = 1.0;
        let s_0: f64 = 100.0;
        let drift_component: SDEComponent = SDEComponent::Linear { a: 10.0 };
        let diffusion_component: SDEComponent = SDEComponent::Linear { a: 0.4 };
        let noise: Noise = Noise::WeinerProcess;
        let jump: Jump = Jump::NoJumps;
        // Arithmetic Brownian motion SDE definition
        let sde: SDE = SDE::new(t_f, s_0, 200, 100, drift_component, diffusion_component, noise, jump).unwrap();
        let paths: Vec<Array1<f64>> = sde.get_paths().unwrap();
        // Paths plot
        let plot: Plot = plot_stochastic_paths(&paths, None).unwrap();
        plot.show()
    }
}