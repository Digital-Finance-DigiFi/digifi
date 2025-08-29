use std::cmp::Ordering;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use ndarray::{Array1, Array2};
use crate::error::DigiFiError;
use crate::utilities::{maths_utils::euclidean_distance, data_transformations::log_return_transformation};
use crate::statistics::pearson_correlation;


#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Type of distance function to apply when computing the distance between the nodes in the minimal spanning tree.
pub enum MSTDistance {
    /// Euclidean distance measuring the distance between two points in an Euclidean space.
    ///
    /// LaTeX Fomula
    /// - d(i,j) = \\sum^{n}_{k=1}(S_{ik} - S_{jk})^{2}
    /// 
    /// # Links
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Euclidean_distance>
    /// - Original Source: N/A
    EuclideanDistance,
    /// Distance computed based on the Pearson correlation between the log returns of the two time series.
    /// 
    /// LaTeX Fomula
    /// - d(i,j) = 1 - \\rho^{2}_{ij}
    /// 
    /// # Links
    /// - Wikipedia: N/A
    /// - Original Source: <http://dx.doi.org/10.1007/s100510050929>
    MantegnaDistance,
}

impl MSTDistance {

    fn mantegna_distance(v_1: &Array1<f64>, v_2: &Array1<f64>) -> Result<f64, DigiFiError>  {
        let returns_1: Array1<f64> = log_return_transformation(v_1);
        let returns_2: Array1<f64> = log_return_transformation(v_2);
        pearson_correlation(&returns_1, &returns_2, 0)
    }

    /// Computes the distance between points (or vectors) to measure the distance between the nodes of the minimal-spanning tree (MST).
    /// 
    /// # Input
    /// - `v_1`: Point or vector that represents a node of the MST.
    /// - `v_2`: Point or vector that represents a node of the MST.
    /// 
    /// # Output
    /// - Distance between the provided nodes
    pub fn distance(&self, v_1: &Array1<f64>, v_2: &Array1<f64>) -> Result<f64, DigiFiError> {
        match self {
            MSTDistance::EuclideanDistance => Ok(euclidean_distance(v_1, v_2)),
            MSTDistance::MantegnaDistance => MSTDistance::mantegna_distance(v_1, v_2),
        }
    }
}


#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize))]
/// Node (vertex) of the minimal-spanning tree (MST).
pub struct MSTNode<'x> {
    /// Name of the node.
    pub name: String,
    /// Index of the node
    pub index: usize,
    /// Coordinate of the node.
    pub coordinate: &'x Array1<f64>,
}


#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize))]
/// Edge of the minimal-spanning tree (MST).
pub struct MSTEdge<'a, 'b, 'x> {
    /// Node of the graph.
    pub node_1: &'a MSTNode<'x>,
    /// Node of the graph
    pub node_2: &'b MSTNode<'x>,
    /// Weight or distance between `node_1` and `node_2`.
    pub weight: f64,
}


#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize))]
/// Minimal-spanning tree algorithm.
/// 
/// # Links
/// - Wikipedia: <https://en.wikipedia.org/wiki/Minimum_spanning_tree>
/// - Original Source: <https://www.geeksforgeeks.org/kruskals-algorithm-in-python/>
/// 
/// # Examples
///
/// ```rust
/// use ndarray::{arr1, arr2, Array2};
/// use digifi::utilities::minimal_spanning_tree::{MSTNode, MSTEdge, MST};
///
/// // Define nodes
/// let node_0: MSTNode = MSTNode { name: "First".to_owned(), index: 0, coordinate: &arr1(&[1.0, 2.0, 3.0]), };
/// let node_1: MSTNode = MSTNode { name: "Second".to_owned(), index: 1, coordinate: &arr1(&[1.0, 2.0, 3.0]), };
/// let node_2: MSTNode = MSTNode { name: "Third".to_owned(), index: 2, coordinate: &arr1(&[1.0, 2.0, 3.0]), };
/// let node_3: MSTNode = MSTNode { name: "Fourth".to_owned(), index: 3, coordinate: &arr1(&[1.0, 2.0, 3.0]), };
/// let nodes: Vec<MSTNode> = vec![node_0, node_1, node_2, node_3];
/// 
/// // Define graph edges
/// let edge_0: MSTEdge = MSTEdge { node_1: &nodes[0], node_2: &nodes[1], weight: 10.0 };
/// let edge_1: MSTEdge = MSTEdge { node_1: &nodes[0], node_2: &nodes[2], weight: 6.0 };
/// let edge_2: MSTEdge = MSTEdge { node_1: &nodes[0], node_2: &nodes[3], weight: 5.0 };
/// let edge_3: MSTEdge = MSTEdge { node_1: &nodes[1], node_2: &nodes[3], weight: 15.0 };
/// let edge_4: MSTEdge = MSTEdge { node_1: &nodes[2], node_2: &nodes[3], weight: 4.0 };
/// 
/// // MST
/// let mut mst: MST = MST::new(nodes.len(), vec![edge_0, edge_1, edge_2, edge_3, edge_4]);
/// mst.kruskal_mst();
/// println!("Edges in the constructed MST:");
/// for edge in &mst.result {
///     println!("{} -- {} == {}", edge.node_1.name, edge.node_2.name, edge.weight);
/// }
/// assert_eq!(mst.minimum_cost(), 19.0);
/// 
/// // Distance matrix
/// let (node_names, distances) = mst.distance_matrix(&nodes).unwrap();
/// assert_eq!(node_names, vec!["First".to_owned(), "Second".to_owned(), "Third".to_owned(), "Fourth".to_owned()]);
/// let distances_: Array2<Option<f64>> = arr2(&[
///     [None, Some(10.0), None, Some(5.0)],
///     [Some(10.0), None, None, None],
///     [None, None, None, Some(4.0)],
///     [Some(5.0), None, Some(4.0), None]
/// ]);
/// assert_eq!(distances, distances_);
/// ```
pub struct MST<'a, 'b, 'x> {
    /// Number of nodes in the graph.
    pub n_nodes: usize,
    /// Edges of the graph.
    pub graph: Vec<MSTEdge<'a, 'b, 'x>>,
    /// Edges that define the MST.
    pub result: Vec<MSTEdge<'a, 'b, 'x>>
}

impl<'a, 'b, 'x> MST<'a, 'b, 'x> {

    /// Creates a new instance of MST struct.
    /// 
    /// # Input
    /// - `n_nodes`: Number of nodes in the graph
    /// - `edges`: Original edges of the graph that will be used to comute the MST
    pub fn new(n_nodes: usize, edges: Vec<MSTEdge<'a, 'b, 'x>>) -> Self {
        MST { n_nodes, graph: edges, result: Vec::<MSTEdge>::new(), }
    }

    /// Adds an edge to the existing graph.
    pub fn add_edge(&mut self, edge: MSTEdge<'a, 'b, 'x>) -> () {
        self.graph.push(edge);
    }

    /// Adds multiple edges to the existing graph.
    pub fn add_edges(&mut self, edges: Vec<MSTEdge<'a, 'b, 'x>>) -> () {
        for edge in edges {
            self.add_edge(edge);
        }
    }

    /// Computes the weights of the existing edges in the graph.
    pub fn compute_edge_weights(&mut self, distance_type: &MSTDistance) -> Result<(), DigiFiError> {
        for edge in &mut self.graph {
            edge.weight = distance_type.distance(&edge.node_1.coordinate, &edge.node_2.coordinate)?;
        }
        Ok(())
    }

    /// A utility function to find the set of an element `i` (uses path compression technique).
    fn find(&self, parent: &mut Vec<usize>, i: usize) -> usize {
        if parent[i] != i {
            // Reassignment of node's parent to the root node as path compression requires
            parent[i] = self.find(parent, parent[i])
        }
        parent[i]
    }

    /// A function that does the union of two sets of `x` and `y` (uses union by rank).
    fn union(&self, parent: &mut Vec<usize>, rank: &mut Vec<usize>, x: usize, y: usize) -> () {
        // Attach smaller rank tree under the root of the high rank tree (union by rank)
        if rank[x] < rank[y] {
            parent[x] = y;
        } else if rank[x] > rank[y] {
            parent[y] = x;
        } else {
            // If ranks are the same, then make one as the root and increment its rank by one
            parent[y] = x;
            rank[x] += 1;
        }
    }

    /// Finds minimal-spanning tree using the Kruskal's algorithm.
    pub fn kruskal_mst(&mut self) -> () {
        // Result of the MST
        let mut result: Vec<MSTEdge> = Vec::<MSTEdge>::new();
        // Index variable for stored edges
        let mut i: usize = 0;
        // Index variable for result
        let mut e: usize = 0;
        // Sorts all edges in non-decreasing order of their weight
        self.graph.sort_by(|a, b| {
            if a.weight < b.weight {
                Ordering::Less
            } else if a.weight > b.weight {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        } );
        let mut parent: Vec<usize> = Vec::<usize>::new();
        let mut rank: Vec<usize> = Vec::<usize>::new();
        // Create `n_nodes`` subsets with single elements
        for node in 0..self.n_nodes {
            parent.push(node);
            rank.push(0);
        }
        // Number of edges to be taken is less than `n_nodes - 1``
        while e < (self.n_nodes - 1) {
            // Picks the smallest edge and inccrements the index for next iteration
            let edge: &MSTEdge<'a, 'b, 'x> = &self.graph[i];
            i += 1;
            let x: usize = self.find(&mut parent, edge.node_1.index);
            let y: usize = self.find(&mut parent, edge.node_2.index);
            // If including this edge doesn't cause a cycle, then include it in the result abd increment the index of result for the next edge
            if x != y {
                e += 1;
                result.push(edge.clone());
                self.union(&mut parent, &mut rank, x, y);
            }
            // Else discard the edge
        }
        self.result = result;
    }

    /// Calculates the minimum cost of the minimal-spanning tree.
    pub fn minimum_cost(&self) -> f64 {
        let mut minimum_cost: f64 = 0.0;
        for edge in &self.result {
            minimum_cost += edge.weight;
        }
        minimum_cost
    }

    /// Extracts the distance matrix from the resulting MST.
    pub fn distance_matrix(&self, nodes: &Vec<MSTNode>) -> Result<(Vec<String>, Array2<Option<f64>>), DigiFiError> {
        let error_title: String = String::from("Minimal-Spanning Tree Distance Matrix");
        // Get ordered node names
        let mut node_names: Vec<String> = Vec::<String>::new();
        for node in nodes {
            node_names.push(node.name.clone());
        }
        // Create distance matrix
        let distances: Vec<Option<f64>> = vec![None; self.n_nodes.pow(2)];
        let mut distances: Array2<Option<f64>> = Array2::from_shape_vec((self.n_nodes, self.n_nodes), distances)?;
        for edge in &self.result {
            let i: usize = node_names.iter().position(|v| v == &edge.node_1.name )
                .ok_or(DigiFiError::NotFound { title: error_title.clone(), data: "matching edge name".to_owned(), })?;
            let j: usize = node_names.iter().position(|v| v == &edge.node_2.name )
                .ok_or(DigiFiError::NotFound { title: error_title.clone(), data: "matching edge name".to_owned(), })?;
            distances[[i, j]] = Some(edge.weight);
            distances[[j, i]] = Some(edge.weight);
        }
        Ok((node_names, distances))
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, Array2};

    #[test]
    fn test_mst() -> () {
        use crate::utilities::minimal_spanning_tree::{MSTNode, MSTEdge, MST};
        // Define nodes
        let node_0: MSTNode = MSTNode { name: "First".to_owned(), index: 0, coordinate: &arr1(&[1.0, 2.0, 3.0]), };
        let node_1: MSTNode = MSTNode { name: "Second".to_owned(), index: 1, coordinate: &arr1(&[1.0, 2.0, 3.0]), };
        let node_2: MSTNode = MSTNode { name: "Third".to_owned(), index: 2, coordinate: &arr1(&[1.0, 2.0, 3.0]), };
        let node_3: MSTNode = MSTNode { name: "Fourth".to_owned(), index: 3, coordinate: &arr1(&[1.0, 2.0, 3.0]), };
        let nodes: Vec<MSTNode> = vec![node_0, node_1, node_2, node_3];
        // Define graph edges
        let edge_0: MSTEdge = MSTEdge { node_1: &nodes[0], node_2: &nodes[1], weight: 10.0 };
        let edge_1: MSTEdge = MSTEdge { node_1: &nodes[0], node_2: &nodes[2], weight: 6.0 };
        let edge_2: MSTEdge = MSTEdge { node_1: &nodes[0], node_2: &nodes[3], weight: 5.0 };
        let edge_3: MSTEdge = MSTEdge { node_1: &nodes[1], node_2: &nodes[3], weight: 15.0 };
        let edge_4: MSTEdge = MSTEdge { node_1: &nodes[2], node_2: &nodes[3], weight: 4.0 };
        // MST
        let mut mst: MST = MST::new(nodes.len(), vec![edge_0, edge_1, edge_2, edge_3, edge_4]);
        mst.kruskal_mst();
        println!("Edges in the constructed MST:");
        for edge in &mst.result {
            println!("{} -- {} == {}", edge.node_1.name, edge.node_2.name, edge.weight);
        }
        assert_eq!(mst.minimum_cost(), 19.0);
        // Distance matrix
        let (node_names, distances) = mst.distance_matrix(&nodes).unwrap();
        assert_eq!(node_names, vec!["First".to_owned(), "Second".to_owned(), "Third".to_owned(), "Fourth".to_owned()]);
        let distances_: Array2<Option<f64>> = arr2(&[
            [None, Some(10.0), None, Some(5.0)],
            [Some(10.0), None, None, None],
            [None, None, None, Some(4.0)],
            [Some(5.0), None, Some(4.0), None]
        ]);
        assert_eq!(distances, distances_);
    }
}