use ndarray::Array2;
use crate::error::DigiFiError;
use crate::utilities::minimal_spanning_tree::{MSTDistance, MSTNode, MSTEdge, MST};
use crate::portfolio_applications::{AssetHistData, portfolio_composition::Portfolio};


pub trait PortfolioTaxonomy {
    fn minimal_spanning_tree(&self, distance_type: &MSTDistance) -> Result<(Vec<String>, Array2<Option<f64>>), DigiFiError>;
}


impl PortfolioTaxonomy for Portfolio {

    /// Computes the minimal-spanning tree for the portfolio of assets based on the distance (i.e., weights) metric provided.
    /// 
    /// # Input
    /// - `distance_type`: Type of metric used to compute the distance (i.e., weights) between the edges.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use std::collections::HashMap;
    /// use ndarray::Array1;
    /// use digifi::utilities::{Time, MSTDistance};
    /// use digifi::portfolio_applications::{AssetHistData, portfolio_performance::SharpeRatio};
    /// use digifi::portfolio_applications::portfolio_composition::{Asset, Portfolio};
    /// use digifi::portfolio_applications::portfolio_taxonomy::PortfolioTaxonomy;
    ///
    /// #[cfg(feature = "sample_data")]
    /// fn test_portfolio_taxonomy() -> () {
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
    ///
    ///     // Minimal-spanning tree
    ///     let (asset_names, distance_matrix) = portfolio.minimal_spanning_tree(&MSTDistance::MantegnaDistance).unwrap();
    ///     println!("Asset Names: {:?}", asset_names);
    ///     println!("{:?}", distance_matrix);
    /// }
    /// ```
    fn minimal_spanning_tree(&self, distance_type: &MSTDistance) -> Result<(Vec<String>, Array2<Option<f64>>), DigiFiError> {
        let mut node_index: usize = 0;
        let mut nodes: Vec<MSTNode> = Vec::<MSTNode>::new();
        // Create nodes
        let assets_names: &Vec<String> = self.assets_names();
        let assets: &Vec<AssetHistData> = self.assets();
        for i in 0..assets_names.len() {
            let node: MSTNode = MSTNode { name: assets_names[i].clone(), index: node_index, coordinate: &assets[i].price_array };
            nodes.push(node);
            node_index += 1;
        }
        // Create edges
        let mut edges: Vec<MSTEdge> = Vec::<MSTEdge>::new();
        let n_nodes: usize = nodes.len();
        for i in 0..n_nodes {
            for j in 0..i {
                if i == j { continue; }
                let edge: MSTEdge = MSTEdge { node_1: &nodes[i], node_2: &nodes[j], weight: 0.0, };
                edges.push(edge);
            }
        }
        let mut mst: MST = MST::new(n_nodes, edges);
        mst.compute_edge_weights(distance_type)?;
        mst.kruskal_mst();
        mst.distance_matrix(&nodes)
    }
}


#[cfg(all(test, feature = "sample_data"))]
mod tests {
    use std::collections::HashMap;
    use ndarray::Array1;
    use crate::utilities::Time;
    use crate::portfolio_applications::{AssetHistData, portfolio_performance::SharpeRatio};
    use crate::portfolio_applications::portfolio_composition::{Asset, Portfolio};
    use crate::portfolio_applications::portfolio_taxonomy::PortfolioTaxonomy;
    use crate::utilities::{minimal_spanning_tree::MSTDistance, sample_data::SampleData};

    #[test]
    fn unit_test_portfolio_taxonomy() -> () {
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
        // Portfolio definition
        let portfolio: Portfolio = Portfolio::build(assets, None, None, None, performance_metric).unwrap();
        // Minimal-spanning tree
        let (asset_names, distance_matrix) = portfolio.minimal_spanning_tree(&MSTDistance::MantegnaDistance).unwrap();
        println!("Asset Names: {:?}", asset_names);
        println!("{:?}", distance_matrix);
    }
}