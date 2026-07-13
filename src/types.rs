use std::borrow::Cow;
use std::fmt;

use bytemuck::{Pod, cast_slice, pod_read_unaligned};
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

/// Cast a byte blob to a typed slice, tolerating misaligned input.
///
/// SQLite blob pointers and `Vec<u8>` buffers are not guaranteed to be
/// aligned for T; fall back to an owned copy when they are not.
pub fn cast_blob<T: Pod>(blob: &[u8]) -> Cow<'_, [T]> {
    match bytemuck::try_cast_slice::<u8, T>(blob) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(
            blob.chunks_exact(std::mem::size_of::<T>())
                .map(pod_read_unaligned)
                .collect(),
        ),
    }
}

/// Encode a typed slice to a byte blob. The inverse of [`cast_blob`].
pub fn slice_to_blob<T: Pod>(values: &[T]) -> Vec<u8> {
    cast_slice(values).to_vec()
}

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
                let values = cast_blob::<f16>(blob);
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(VectorTypeError::NonFiniteValue);
                }
            }
            Self::Float4 => {
                let values = cast_blob::<f32>(blob);
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(VectorTypeError::NonFiniteValue);
                }
            }
            Self::Float8 => {
                let values = cast_blob::<f64>(blob);
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(VectorTypeError::NonFiniteValue);
                }
            }
            Self::Int1 | Self::Int2 | Self::Int4 => {} // integers are always finite
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::cast_slice;
    use half::f16;

    // ----------------------------------------------------------------
    // from_name
    // ----------------------------------------------------------------

    #[test]
    fn from_name_all_valid() {
        assert_eq!(VectorType::from_name("float2"), Ok(VectorType::Float2));
        assert_eq!(VectorType::from_name("float4"), Ok(VectorType::Float4));
        assert_eq!(VectorType::from_name("float8"), Ok(VectorType::Float8));
        assert_eq!(VectorType::from_name("int1"), Ok(VectorType::Int1));
        assert_eq!(VectorType::from_name("int2"), Ok(VectorType::Int2));
        assert_eq!(VectorType::from_name("int4"), Ok(VectorType::Int4));
    }

    #[test]
    fn from_name_unknown_returns_error() {
        let err = VectorType::from_name("float3").unwrap_err();
        assert_eq!(err, VectorTypeError::UnknownType("float3".to_string()));
    }

    #[test]
    fn from_name_case_sensitive() {
        // "Float4" is not a valid name — the matcher is lowercase-only.
        assert!(VectorType::from_name("Float4").is_err());
        assert!(VectorType::from_name("INT1").is_err());
        assert!(VectorType::from_name("").is_err());
    }

    // ----------------------------------------------------------------
    // name — round-trip with from_name
    // ----------------------------------------------------------------

    #[test]
    fn name_round_trips_with_from_name() {
        let variants = [
            VectorType::Float2,
            VectorType::Float4,
            VectorType::Float8,
            VectorType::Int1,
            VectorType::Int2,
            VectorType::Int4,
        ];
        for vt in variants {
            assert_eq!(VectorType::from_name(vt.name()), Ok(vt));
        }
    }

    // ----------------------------------------------------------------
    // element_size
    // ----------------------------------------------------------------

    #[test]
    fn element_size_correct() {
        assert_eq!(VectorType::Float2.element_size(), 2);
        assert_eq!(VectorType::Float4.element_size(), 4);
        assert_eq!(VectorType::Float8.element_size(), 8);
        assert_eq!(VectorType::Int1.element_size(), 1);
        assert_eq!(VectorType::Int2.element_size(), 2);
        assert_eq!(VectorType::Int4.element_size(), 4);
    }

    // ----------------------------------------------------------------
    // blob_size
    // ----------------------------------------------------------------

    #[test]
    fn blob_size_is_element_size_times_dim() {
        for vt in [
            VectorType::Float2,
            VectorType::Float4,
            VectorType::Float8,
            VectorType::Int1,
            VectorType::Int2,
            VectorType::Int4,
        ] {
            for dim in [0, 1, 3, 128, 1536] {
                assert_eq!(vt.blob_size(dim), vt.element_size() * dim);
            }
        }
    }

    // ----------------------------------------------------------------
    // validate_blob
    // ----------------------------------------------------------------

    #[test]
    fn validate_blob_correct_size_ok() {
        let blob = vec![0u8; VectorType::Float4.blob_size(4)]; // 16 bytes
        assert!(VectorType::Float4.validate_blob(&blob, 4).is_ok());
    }

    #[test]
    fn validate_blob_too_short_returns_error() {
        let blob = vec![0u8; 12]; // 12 bytes for a 4-dim float4 should be 16
        let err = VectorType::Float4.validate_blob(&blob, 4).unwrap_err();
        assert_eq!(
            err,
            VectorTypeError::DimensionMismatch {
                expected: 4,
                got: 3
            }
        );
    }

    #[test]
    fn validate_blob_too_long_returns_error() {
        let blob = vec![0u8; 20]; // 20 bytes for a 4-dim float4 should be 16
        let err = VectorType::Float4.validate_blob(&blob, 4).unwrap_err();
        assert_eq!(
            err,
            VectorTypeError::DimensionMismatch {
                expected: 4,
                got: 5
            }
        );
    }

    #[test]
    fn validate_blob_int_types() {
        let blob = vec![0u8; 6]; // 6 × i16 == 3 dims
        assert!(VectorType::Int2.validate_blob(&blob, 3).is_ok());

        let err = VectorType::Int2.validate_blob(&blob, 4).unwrap_err();
        assert_eq!(
            err,
            VectorTypeError::DimensionMismatch {
                expected: 4,
                got: 3
            }
        );
    }

    // ----------------------------------------------------------------
    // validate_finite
    // ----------------------------------------------------------------

    #[test]
    fn validate_finite_all_finite_f32_ok() {
        let values: Vec<f32> = vec![1.0, -2.5, 0.0, f32::MAX];
        let blob = slice_to_blob(&values);
        assert!(VectorType::Float4.validate_finite(&blob, 4).is_ok());
    }

    #[test]
    fn validate_finite_nan_f32_errors() {
        let values: Vec<f32> = vec![1.0, f32::NAN, 3.0];
        let blob = slice_to_blob(&values);
        assert_eq!(
            VectorType::Float4.validate_finite(&blob, 3).unwrap_err(),
            VectorTypeError::NonFiniteValue
        );
    }

    #[test]
    fn validate_finite_inf_f32_errors() {
        let values: Vec<f32> = vec![1.0, f32::INFINITY];
        let blob = slice_to_blob(&values);
        assert_eq!(
            VectorType::Float4.validate_finite(&blob, 2).unwrap_err(),
            VectorTypeError::NonFiniteValue
        );
    }

    #[test]
    fn validate_finite_neg_inf_f64_errors() {
        let values: Vec<f64> = vec![0.0, f64::NEG_INFINITY];
        let blob = slice_to_blob(&values);
        assert_eq!(
            VectorType::Float8.validate_finite(&blob, 2).unwrap_err(),
            VectorTypeError::NonFiniteValue
        );
    }

    #[test]
    fn validate_finite_all_finite_f64_ok() {
        let values: Vec<f64> = vec![1.0, -2.5, 0.0, f64::MAX];
        let blob = slice_to_blob(&values);
        assert!(VectorType::Float8.validate_finite(&blob, 4).is_ok());
    }

    #[test]
    fn validate_finite_nan_f16_errors() {
        let values: Vec<f16> = vec![f16::from_f32(1.0), f16::NAN];
        let blob = slice_to_blob(&values);
        assert_eq!(
            VectorType::Float2.validate_finite(&blob, 2).unwrap_err(),
            VectorTypeError::NonFiniteValue
        );
    }

    #[test]
    fn validate_finite_inf_f16_errors() {
        let values: Vec<f16> = vec![f16::INFINITY];
        let blob = slice_to_blob(&values);
        assert_eq!(
            VectorType::Float2.validate_finite(&blob, 1).unwrap_err(),
            VectorTypeError::NonFiniteValue
        );
    }

    #[test]
    fn validate_finite_all_finite_f16_ok() {
        let values: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(-0.5), f16::from_f32(0.0)];
        let blob = slice_to_blob(&values);
        assert!(VectorType::Float2.validate_finite(&blob, 3).is_ok());
    }

    #[test]
    fn validate_finite_integer_types_always_ok() {
        // Integer types never contain NaN/Inf; validate_finite should be a no-op.
        let i8_blob = slice_to_blob::<i8>(&[i8::MIN, 0, i8::MAX]);
        let i16_blob = slice_to_blob::<i16>(&[i16::MIN, 0, i16::MAX]);
        let i32_blob = slice_to_blob::<i32>(&[i32::MIN, 0, i32::MAX]);

        assert!(VectorType::Int1.validate_finite(&i8_blob, 3).is_ok());
        assert!(VectorType::Int2.validate_finite(&i16_blob, 3).is_ok());
        assert!(VectorType::Int4.validate_finite(&i32_blob, 3).is_ok());
    }

    // ----------------------------------------------------------------
    // is_float
    // ----------------------------------------------------------------

    #[test]
    fn is_float_true_for_float_variants() {
        assert!(VectorType::Float2.is_float());
        assert!(VectorType::Float4.is_float());
        assert!(VectorType::Float8.is_float());
    }

    #[test]
    fn is_float_false_for_int_variants() {
        assert!(!VectorType::Int1.is_float());
        assert!(!VectorType::Int2.is_float());
        assert!(!VectorType::Int4.is_float());
    }

    // ----------------------------------------------------------------
    // slice_to_blob / blob_to_slice — round-trip
    // ----------------------------------------------------------------

    #[test]
    fn round_trip_f32() {
        let original: Vec<f32> = vec![1.0, -2.5, 3.125, 0.0];
        let blob = slice_to_blob(&original);
        assert_eq!(blob.len(), original.len() * 4);
        let recovered = cast_blob::<f32>(&blob);
        assert_eq!(recovered.as_ref(), original.as_slice());
    }

    #[test]
    fn round_trip_f64() {
        let original: Vec<f64> = vec![1.0, -2.5, f64::MAX, f64::MIN_POSITIVE];
        let blob = slice_to_blob(&original);
        assert_eq!(blob.len(), original.len() * 8);
        let recovered = cast_blob::<f64>(&blob);
        assert_eq!(recovered.as_ref(), original.as_slice());
    }

    #[test]
    fn round_trip_f16() {
        let original: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(-0.5), f16::from_f32(0.0)];
        let blob = slice_to_blob(&original);
        assert_eq!(blob.len(), original.len() * 2);
        let recovered = cast_blob::<f16>(&blob);
        assert_eq!(recovered.as_ref(), original.as_slice());
    }

    #[test]
    fn round_trip_i8() {
        let original: Vec<i8> = vec![i8::MIN, -1, 0, 1, i8::MAX];
        let blob = slice_to_blob(&original);
        assert_eq!(blob.len(), original.len());
        let recovered = cast_blob::<i8>(&blob);
        assert_eq!(recovered.as_ref(), original.as_slice());
    }

    #[test]
    fn round_trip_i16() {
        let original: Vec<i16> = vec![i16::MIN, -1, 0, 1, i16::MAX];
        let blob = slice_to_blob(&original);
        assert_eq!(blob.len(), original.len() * 2);
        let recovered = cast_blob::<i16>(&blob);
        assert_eq!(recovered.as_ref(), original.as_slice());
    }

    #[test]
    fn round_trip_i32() {
        let original: Vec<i32> = vec![i32::MIN, -1, 0, 1, i32::MAX];
        let blob = slice_to_blob(&original);
        assert_eq!(blob.len(), original.len() * 4);
        let recovered = cast_blob::<i32>(&blob);
        assert_eq!(recovered.as_ref(), original.as_slice());
    }

    // Verify that slice_to_blob produces the same bytes as bytemuck::cast_slice
    // directly, ensuring no extra copies or byte-swaps are introduced.
    #[test]
    fn slice_to_blob_matches_bytemuck_cast_slice() {
        let values: Vec<f32> = vec![1.0_f32, 2.0, 3.0];
        let expected: &[u8] = cast_slice(&values);
        let got = slice_to_blob(&values);
        assert_eq!(got.as_slice(), expected);
    }

    #[test]
    fn cast_blob_handles_misaligned_input() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut padded = vec![0u8];
        padded.extend_from_slice(cast_slice(&values));
        let misaligned = &padded[1..]; // guaranteed misaligned for f32 (alloc + 1)
        let out: std::borrow::Cow<'_, [f32]> = cast_blob(misaligned);
        assert_eq!(out.as_ref(), values.as_slice());
    }
}
