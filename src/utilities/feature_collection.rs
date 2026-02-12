use std::borrow::Borrow;
use ndarray::{Array1, Array2};
use crate::error::{DigiFiError, ErrorTitle};


#[derive(Debug, Default)]
/// Collection of features that are organized for modelling.
/// 
/// # Examples
/// 
/// 1. Working with feature collection
/// 
/// ```rust
/// use ndarray::{Array1, Array2, array};
/// use digifi::utilities::FeatureCollection;
/// 
/// // Features
/// let x_1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let x_2: Vec<f64> = vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
/// let x_3: Array1<f64> = Array1::from_vec(vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0]);
/// let x_4: Array1<f64> = Array1::from_vec(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
/// 
/// // Create feature collection with iterators from different types of data structures
/// let mut fc: FeatureCollection = FeatureCollection::new();
/// fc.add_feature(x_1.iter(), "x_1").unwrap();
/// assert_eq!(fc.len(), 1);
/// assert_eq!(fc.feature_size(), Some(6));
/// fc.add_feature(x_2.into_iter(), "x_2").unwrap();
/// assert_eq!(fc.len(), 2);
/// assert_eq!(fc.feature_size(), Some(6));
/// fc.add_feature(x_3.iter(), "x_3").unwrap();
/// assert_eq!(fc.len(), 3);
/// assert_eq!(fc.feature_size(), Some(6));
/// fc.add_feature(x_4.into_iter(), "x_4").unwrap();
/// assert_eq!(fc.len(), 4);
/// assert_eq!(fc.feature_size(), Some(6));
/// 
/// // Get features from the collection
/// assert_eq!(fc.get_feature_array("x_3").unwrap(), &x_3);
/// let matrix: Array2<f64> = array![
///     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
///     [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
///     [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
///     [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
/// ].reversed_axes();
/// assert_eq!(fc.get_matrix().unwrap(), matrix);
/// 
/// // Remove feature
/// fc.remove_feature("x_1").unwrap();
/// assert_eq!(fc.len(), 3);
/// assert_eq!(fc.feature_size(), Some(6));
/// ```
/// 
/// 2. Using feature collection for modelling
/// 
/// ```rust
/// use ndarray::{Array1, Array2, array};
/// use digifi::utilities::{TEST_ACCURACY, FeatureCollection};
/// use digifi::statistics::linear_regression;
/// 
/// let mut fc: FeatureCollection = FeatureCollection::new();
/// fc.add_constant = true;
/// fc.add_feature(vec![1.0, 4.0, 6.0].into_iter(), "x_1").unwrap();
/// fc.add_feature(vec![3.0, 4.0, 5.0].into_iter(), "x_2").unwrap();
/// let y: Array1<f64> = array![1.0, 2.0, 3.0];
/// let params: Array1<f64> = linear_regression(&fc.into_matrix().unwrap(), &y).unwrap();
/// 
/// // The results were found using LinearRegression from sklearn
/// let results: Array1<f64> = Array1::from(vec![-2.49556592e-16, 1.0, -2.0]);
/// assert!((&params - &results).map(|v| v.abs() ).sum() < TEST_ACCURACY);
/// ```
pub struct FeatureCollection {
    pub features: Vec<Array1<f64>>,
    pub feature_names: Vec<String>,
    feature_size: Option<usize>,
    pub add_constant: bool,
}

impl FeatureCollection {
    /// Creates a new instance  of `FeatureCollection`.
    pub fn new() -> Self {
        Self::default()
    }

    fn validate_feature<T, I>(&self, feature: &T, feature_name: &str) -> Result<(), DigiFiError>
    where
        T: Iterator<Item = I> + ExactSizeIterator,
        I: Borrow<f64>,
    {
        if self.feature_names.contains(&feature_name.to_owned()) {
            return Err(DigiFiError::Other { title: Self::error_title(), details: format!("Feature `{}` already exists in the collection.", feature_name), });
        }
        if let Some(feature_size) =  self.feature_size {
            if feature.len() != feature_size {
                return Err(DigiFiError::WrongLength { title: Self::error_title(), arg: feature_name.to_owned(), len: feature_size, });
            }
        }
        Ok(())
    }

    /// Checks that the feature exists in the collection.
    fn feature_exists(&self, feature_name: &str) -> Result<(), DigiFiError> {
        if !self.feature_names.contains(&feature_name.to_owned()) {
            return Err(DigiFiError::Other { title: Self::error_title(), details: format!("Feature `{}` doesn't exists in the collection.", feature_name), });
        }
        Ok(())
    }

    /// Returns the number of features in the collection.
    pub fn len(&self) -> usize {
        self.feature_names.len()
    }

    /// Returns the length of a feature in the collection.
    /// 
    /// Note: This length will be the same for every feature.
    pub fn feature_size(&self) -> Option<usize> {
        self.feature_size
    }

    /// Returns the index of the feature in the collection.
    pub fn get_feature_index(&self, feature_name: &str) -> Result<usize, DigiFiError> {
        self.feature_names.iter().position(|v| v == feature_name )
            .ok_or(DigiFiError::NotFound { title: Self::error_title(), data: feature_name.to_owned(), })
    }

    /// Adds the feature to the collection.
    pub fn add_feature<T, I>(&mut self, feature: T, feature_name: &str) -> Result<(), DigiFiError>
    where
        T: Iterator<Item = I> + ExactSizeIterator,
        I: Borrow<f64>,
    {
        let feature_name: String = String::from(feature_name);
        self.validate_feature(&feature, &feature_name)?;
        let feature_len: usize = feature.len();
        // Update feature information
        self.features.push(feature.map(|v| *v.borrow() ).collect());
        self.feature_names.push(feature_name);
        // Update other metadata
        if let None = self.feature_size {
            self.feature_size = Some(feature_len);
        }
        Ok(())
    }

    /// Removes the feature from the collection.
    pub fn remove_feature(&mut self, feature_name: &str) -> Result<(), DigiFiError> {
        let feature_name: String = String::from(feature_name);
        self.feature_exists(&feature_name)?;
        let index: usize = self.get_feature_index(&feature_name)?;
        // Update feature information
        self.features.remove(index);
        self.feature_names.remove(index);
        // Update other metadata
        if self.feature_names.is_empty() {
            self.feature_size = None;
        }
        Ok(())
    }

    /// Returns a feature as an `Array1`
    pub fn get_feature_array(&self, feature_name: &str) -> Result<&Array1<f64>, DigiFiError> {
        let feature_name: String = String::from(feature_name);
        self.feature_exists(&feature_name)?;
        let index: usize = self.get_feature_index(&feature_name)?;
        Ok(&self.features[index])
    }

    /// Returns a matrix that is composed of the features from the collection.
    pub fn get_matrix(&self) -> Result<Array2<f64>, DigiFiError> {
        let feature_size: usize = self.feature_size
            .ok_or(DigiFiError::Other { title: Self::error_title(), details: "No features are present in the collection.".to_owned(), })?;
        let (mut shape, mut x_matrix) = (
            (self.len(), feature_size),
            self.features.iter().fold(vec![], |mut prev, curr| { prev.append(&mut curr.to_vec()); prev } )
        );
        if self.add_constant {
            shape.0 += 1;
            x_matrix.append(&mut vec![1.0; feature_size]);
        }
        Ok(Array2::from_shape_vec(shape, x_matrix)?.reversed_axes())
    }

    pub fn into_matrix(self) -> Result<Array2<f64>, DigiFiError> {
        self.get_matrix()
    }

    /// Returns delta degrees of freedom.
    pub fn ddof(&self) -> usize {
        match self.add_constant {
            true => self.len() + 1,
            false => self.len(),
        }
    }
}

impl ErrorTitle for FeatureCollection {
    fn error_title() -> String {
        String::from("Feature Collection")
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, array};
    use crate::utilities::{TEST_ACCURACY, feature_collection::FeatureCollection};

    #[test]
    fn unit_test_feature_collection() -> () {
        // Features
        let x_1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_2: Vec<f64> = vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let x_3: Array1<f64> = Array1::from_vec(vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0]);
        let x_4: Array1<f64> = Array1::from_vec(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        // Create feature collection with iterators from different types of data structures
        let mut fc: FeatureCollection = FeatureCollection::new();
        fc.add_feature(x_1.iter(), "x_1").unwrap();
        assert_eq!(fc.len(), 1);
        assert_eq!(fc.feature_size(), Some(6));
        fc.add_feature(x_2.into_iter(), "x_2").unwrap();
        assert_eq!(fc.len(), 2);
        assert_eq!(fc.feature_size(), Some(6));
        fc.add_feature(x_3.iter(), "x_3").unwrap();
        assert_eq!(fc.len(), 3);
        assert_eq!(fc.feature_size(), Some(6));
        fc.add_feature(x_4.into_iter(), "x_4").unwrap();
        assert_eq!(fc.len(), 4);
        assert_eq!(fc.feature_size(), Some(6));
        // Get features from the collection
        assert_eq!(fc.get_feature_array("x_3").unwrap(), &x_3);
        let matrix: Array2<f64> = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ].reversed_axes();
        assert_eq!(fc.get_matrix().unwrap(), matrix);
        // Remove feature
        fc.remove_feature("x_1").unwrap();
        assert_eq!(fc.len(), 3);
        assert_eq!(fc.feature_size(), Some(6));
    }

    #[test]
    fn unit_test_feature_collection_for_linear_regression() -> () {
        use crate::statistics::linear_regression;
        let mut fc: FeatureCollection = FeatureCollection::new();
        fc.add_constant = true;
        fc.add_feature(vec![1.0, 4.0, 6.0].into_iter(), "x_1").unwrap();
        fc.add_feature(vec![3.0, 4.0, 5.0].into_iter(), "x_2").unwrap();
        let y: Array1<f64> = array![1.0, 2.0, 3.0];
        let params: Array1<f64> = linear_regression(&fc.into_matrix().unwrap(), &y).unwrap();
        // The results were found using LinearRegression from sklearn
        let results: Array1<f64> = Array1::from(vec![-2.49556592e-16, 1.0, -2.0]);
        assert!((&params - &results).map(|v| v.abs() ).sum() < TEST_ACCURACY);
    }
}