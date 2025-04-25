// Error types
use std::{error::Error, io::Error as IOError, time::SystemTimeError};
use ndarray::ShapeError;
// Dependencies
use std::{fmt, convert::From};


#[derive(Debug)]
pub enum DigiFiError {
    UnmatchingLength { array_1: String, array_2: String, },
    ParameterConstraint { title: String, constraint: String, },
    ValidationError { title: String, details: String, },
    MeanCalculation { title: String, series: String, },
    NotFound { title: String, data: String, },
    IndexOutOfRange { title: String, index: String, array: String, },
    Other { title: String, details: String, },
    // Std errors
    StdIOError(IOError),
    StdSystemTimeError(SystemTimeError),
    // Ndarray errors
    NdarrayShapeError(ShapeError),
}

impl fmt::Display for DigiFiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DigiFiError::UnmatchingLength { array_1, array_2 } => write!(f, "Unmatching Array Length: The lengths of arrays `{}` and `{}` do not match.", array_1, array_2),
            DigiFiError::ParameterConstraint { title, constraint } => write!(f, "Parameter Constraint (Source: {}): {}", title, constraint),
            DigiFiError::ValidationError { title, details } => write!(f, "Validation Error (Source: {}): {}", title, details),
            DigiFiError::MeanCalculation { title, series } => write!(f, "Mean Calculation (Source: {}): Mean of `{}` is not computed.", title, series),
            DigiFiError::NotFound { title, data } => write!(f, "Not Found (Source: {}): No {} is found.", title, data),
            DigiFiError::IndexOutOfRange { title, index, array } => write!(f, "Index Out of Range (Source: {}): The index `{}` is out of range for the array `{}`.", title, index, array),
            DigiFiError::Other { title, details } => write!(f, "Other Error (Source: {}): {}", title, details),
            // Std errors
            DigiFiError::StdIOError(e) => write!(f, "Std IO Error: {}", e.to_string()),
            DigiFiError::StdSystemTimeError(e) => write!(f, "Std System Time Error: {}", e.to_string()),
            // Ndarray errors
            DigiFiError::NdarrayShapeError(e) => write!(f, "Ndarray Shape Error: {}", e.to_string()),
        }
    }
}

impl Error for DigiFiError {}

impl From<IOError> for DigiFiError {
    fn from(value: IOError) -> Self {
        DigiFiError::StdIOError(value)
    }
}

impl From<SystemTimeError> for DigiFiError {
    fn from(value: SystemTimeError) -> Self {
        DigiFiError::StdSystemTimeError(value)
    }
}

impl From<ShapeError> for DigiFiError {
    fn from(value: ShapeError) -> Self {
        DigiFiError::NdarrayShapeError(value)
    }
}