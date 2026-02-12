// Error types
use std::{error::Error, io::Error as IOError, time::SystemTimeError, num::ParseIntError};
use ndarray::ShapeError;
// Dependencies
use std::{fmt, convert::From};


#[derive(Debug)]
pub enum DigiFiError {
    UnmatchingLength { array_1: String, array_2: String, },
    WrongLength { title: String, arg: String, len: usize, },
    ParameterConstraint { title: String, constraint: String, },
    ValidationError { title: String, details: String, },
    MeanCalculation { title: String, series: String, },
    NotFound { title: String, data: String, },
    IndexOutOfRange { title: String, index: String, array: String, },
    CustomFunctionLengthVal { title: String, },
    Other { title: String, details: String, },
    // Std errors
    StdIOError(IOError),
    StdSystemTimeError(SystemTimeError),
    ParseIntError(ParseIntError),
    // Ndarray errors
    NdarrayShapeError(ShapeError),
}

impl fmt::Display for DigiFiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnmatchingLength { array_1, array_2 } => write!(f, "Unmatching Length: The lengths of arrays `{}` and `{}` do not match.", array_1, array_2),
            Self::WrongLength { title, arg, len } => write!(f, "Wrong Length (Source: {}): The argument `{}` should be of length {}.", title, arg, len),
            Self::ParameterConstraint { title, constraint } => write!(f, "Parameter Constraint (Source: {}): {}", title, constraint),
            Self::ValidationError { title, details } => write!(f, "Validation Error (Source: {}): {}", title, details),
            Self::MeanCalculation { title, series } => write!(f, "Mean Calculation (Source: {}): Mean of `{}` is not computed.", title, series),
            Self::NotFound { title, data } => write!(f, "Not Found (Source: {}): No {} is found.", title, data),
            Self::IndexOutOfRange { title, index, array } => write!(f, "Index Out of Range (Source: {}): The index `{}` is out of range for the array `{}`.", title, index, array),
            Self::CustomFunctionLengthVal { title } => write!(f, "Custom Function Length Validation (Source: {}): Custom function does not produce the array of desired length.", title),
            Self::Other { title, details } => write!(f, "Other Error (Source: {}): {}", title, details),
            // Std errors
            Self::StdIOError(e) => write!(f, "Std IO Error: {}", e.to_string()),
            Self::StdSystemTimeError(e) => write!(f, "Std System Time Error: {}", e.to_string()),
            Self::ParseIntError(e) => write!(f, "Parse Int Error: {}", e.to_string()),
            // Ndarray errors
            Self::NdarrayShapeError(e) => write!(f, "Ndarray Shape Error: {}", e.to_string()),
        }
    }
}

impl Error for DigiFiError {}

impl From<IOError> for DigiFiError {
    fn from(value: IOError) -> Self {
        Self::StdIOError(value)
    }
}

impl From<SystemTimeError> for DigiFiError {
    fn from(value: SystemTimeError) -> Self {
        Self::StdSystemTimeError(value)
    }
}

impl From<ParseIntError> for DigiFiError {
    fn from(value: ParseIntError) -> Self {
        Self::ParseIntError(value)
    }
}

impl From<ShapeError> for DigiFiError {
    fn from(value: ShapeError) -> Self {
        Self::NdarrayShapeError(value)
    }
}


pub trait ErrorTitle {

    /// Returns the error title.
    fn error_title() -> String;
}