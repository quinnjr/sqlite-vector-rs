use std::fmt;

use bytemuck::{Pod, cast_slice};
use half::f16;

/// Errors from vector type operations.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorTypeError {
    UnknownType(String),
    DimensionMismatch { expected: usize, got: usize },
    NonFiniteValue,
}

impl fmt::Display for VectorTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownType(name) => write!(f, "unknown vector type: {name}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "expected {expected} dimensions, got {got}")
            }
            Self::NonFiniteValue => write!(f, "vector contains NaN or Inf"),
        }
    }
}

impl std::error::Error for VectorTypeError {}

/// Supported vector element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorType {
    Float2,
    Float4,
    Float8,
    Int1,
    Int2,
    Int4,
}

impl VectorType {
    /// Parse a SQL type name into a VectorType.
    pub fn from_name(name: &str) -> Result<Self, VectorTypeError> {
        match name {
            "float2" => Ok(Self::Float2),
            "float4" => Ok(Self::Float4),
            "float8" => Ok(Self::Float8),
            "int1" => Ok(Self::Int1),
            "int2" => Ok(Self::Int2),
            "int4" => Ok(Self::Int4),
            other => Err(VectorTypeError::UnknownType(other.to_string())),
        }
    }

    /// Size in bytes of one element.
    pub fn element_size(&self) -> usize {
        match self {
            Self::Float2 => 2,
            Self::Float4 => 4,
            Self::Float8 => 8,
            Self::Int1 => 1,
            Self::Int2 => 2,
            Self::Int4 => 4,
        }
    }

    /// Expected blob size for a given dimension.
    pub fn blob_size(&self, dim: usize) -> usize {
        dim * self.element_size()
    }

    /// Validate that a blob has the correct size for the given dimension.
    pub fn validate_blob(&self, blob: &[u8], dim: usize) -> Result<(), VectorTypeError> {
        let expected = self.blob_size(dim);
        if blob.len() != expected {
            return Err(VectorTypeError::DimensionMismatch {
                expected: dim,
                got: blob.len() / self.element_size(),
            });
        }
        Ok(())
    }

    /// Check that all float values are finite (not NaN or Inf).
    /// No-op for integer types. The `dim` parameter is used to verify
    /// the blob length before casting.
    pub fn validate_finite(&self, blob: &[u8], dim: usize) -> Result<(), VectorTypeError> {
        self.validate_blob(blob, dim)?;
        match self {
            Self::Float2 => {
                let values: &[f16] = cast_slice(blob);
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(VectorTypeError::NonFiniteValue);
                }
            }
            Self::Float4 => {
                let values: &[f32] = cast_slice(blob);
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(VectorTypeError::NonFiniteValue);
                }
            }
            Self::Float8 => {
                let values: &[f64] = cast_slice(blob);
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(VectorTypeError::NonFiniteValue);
                }
            }
            Self::Int1 | Self::Int2 | Self::Int4 => {} // integers are always finite
        }
        Ok(())
    }

    /// Cast a typed slice to a byte blob. Generic helper.
    pub fn slice_to_blob<T: Pod>(&self, values: &[T]) -> Vec<u8> {
        cast_slice(values).to_vec()
    }

    /// Cast a byte blob back to a typed slice. Generic helper.
    /// Caller must ensure the blob was created with the matching type.
    pub fn blob_to_slice<'a, T: Pod>(&self, blob: &'a [u8]) -> &'a [T] {
        cast_slice(blob)
    }

    /// Returns true if this is a float type (has NaN/Inf concerns).
    pub fn is_float(&self) -> bool {
        matches!(self, Self::Float2 | Self::Float4 | Self::Float8)
    }

    /// SQL type name string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Float2 => "float2",
            Self::Float4 => "float4",
            Self::Float8 => "float8",
            Self::Int1 => "int1",
            Self::Int2 => "int2",
            Self::Int4 => "int4",
        }
    }
}
