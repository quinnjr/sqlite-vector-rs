use std::fmt;

use half::f16;
use serde_json::Value;

use crate::types::{VectorType, VectorTypeError};

/// Errors from JSON conversion.
#[derive(Debug)]
pub enum JsonError {
    Parse(serde_json::Error),
    NotAnArray,
    NonNumericElement(usize),
    Type(VectorTypeError),
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid JSON: {e}"),
            Self::NotAnArray => write!(f, "expected a JSON array"),
            Self::NonNumericElement(i) => write!(f, "element {i} is not a number"),
            Self::Type(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for JsonError {}

/// Parse a JSON array string into a vector blob of the given type.
pub fn json_to_blob(json: &str, vtype: VectorType) -> Result<Vec<u8>, JsonError> {
    let value: Value = serde_json::from_str(json).map_err(JsonError::Parse)?;
    let arr = value.as_array().ok_or(JsonError::NotAnArray)?;

    match vtype {
        VectorType::Float2 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_f64().ok_or(JsonError::NonNumericElement(i))?;
                let h = f16::from_f64(n);
                if !h.is_finite() {
                    return Err(JsonError::Type(VectorTypeError::NonFiniteValue));
                }
                values.push(h);
            }
            Ok(vtype.slice_to_blob(&values))
        }
        VectorType::Float4 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_f64().ok_or(JsonError::NonNumericElement(i))? as f32;
                if !n.is_finite() {
                    return Err(JsonError::Type(VectorTypeError::NonFiniteValue));
                }
                values.push(n);
            }
            Ok(vtype.slice_to_blob(&values))
        }
        VectorType::Float8 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_f64().ok_or(JsonError::NonNumericElement(i))?;
                if !n.is_finite() {
                    return Err(JsonError::Type(VectorTypeError::NonFiniteValue));
                }
                values.push(n);
            }
            Ok(vtype.slice_to_blob(&values))
        }
        VectorType::Int1 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_i64().ok_or(JsonError::NonNumericElement(i))? as i8;
                values.push(n);
            }
            Ok(vtype.slice_to_blob(&values))
        }
        VectorType::Int2 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_i64().ok_or(JsonError::NonNumericElement(i))? as i16;
                values.push(n);
            }
            Ok(vtype.slice_to_blob(&values))
        }
        VectorType::Int4 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_i64().ok_or(JsonError::NonNumericElement(i))? as i32;
                values.push(n);
            }
            Ok(vtype.slice_to_blob(&values))
        }
    }
}

/// Convert a vector blob back to a JSON array string.
pub fn blob_to_json(blob: &[u8], vtype: VectorType) -> Result<String, JsonError> {
    let values: Vec<Value> = match vtype {
        VectorType::Float2 => {
            let s: &[f16] = vtype.blob_to_slice(blob);
            s.iter().map(|v| Value::from(v.to_f64())).collect()
        }
        VectorType::Float4 => {
            let s: &[f32] = vtype.blob_to_slice(blob);
            s.iter().map(|v| Value::from(*v)).collect()
        }
        VectorType::Float8 => {
            let s: &[f64] = vtype.blob_to_slice(blob);
            s.iter().map(|v| Value::from(*v)).collect()
        }
        VectorType::Int1 => {
            let s: &[i8] = vtype.blob_to_slice(blob);
            s.iter().map(|v| Value::from(*v as i64)).collect()
        }
        VectorType::Int2 => {
            let s: &[i16] = vtype.blob_to_slice(blob);
            s.iter().map(|v| Value::from(*v as i64)).collect()
        }
        VectorType::Int4 => {
            let s: &[i32] = vtype.blob_to_slice(blob);
            s.iter().map(|v| Value::from(*v as i64)).collect()
        }
    };
    serde_json::to_string(&values).map_err(JsonError::Parse)
}
