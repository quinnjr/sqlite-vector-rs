use std::fmt;

use half::f16;
use serde_json::Value;

use crate::types::{VectorType, VectorTypeError, cast_blob, slice_to_blob};

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
            Ok(slice_to_blob(&values))
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
            Ok(slice_to_blob(&values))
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
            Ok(slice_to_blob(&values))
        }
        VectorType::Int1 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_i64().ok_or(JsonError::NonNumericElement(i))? as i8;
                values.push(n);
            }
            Ok(slice_to_blob(&values))
        }
        VectorType::Int2 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_i64().ok_or(JsonError::NonNumericElement(i))? as i16;
                values.push(n);
            }
            Ok(slice_to_blob(&values))
        }
        VectorType::Int4 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_i64().ok_or(JsonError::NonNumericElement(i))? as i32;
                values.push(n);
            }
            Ok(slice_to_blob(&values))
        }
    }
}

/// Convert a vector blob back to a JSON array string.
pub fn blob_to_json(blob: &[u8], vtype: VectorType) -> Result<String, JsonError> {
    let values: Vec<Value> = match vtype {
        VectorType::Float2 => {
            let s = cast_blob::<f16>(blob);
            s.iter().map(|v| Value::from(v.to_f64())).collect()
        }
        VectorType::Float4 => {
            let s = cast_blob::<f32>(blob);
            s.iter().map(|v| Value::from(*v)).collect()
        }
        VectorType::Float8 => {
            let s = cast_blob::<f64>(blob);
            s.iter().map(|v| Value::from(*v)).collect()
        }
        VectorType::Int1 => {
            let s = cast_blob::<i8>(blob);
            s.iter().map(|v| Value::from(*v as i64)).collect()
        }
        VectorType::Int2 => {
            let s = cast_blob::<i16>(blob);
            s.iter().map(|v| Value::from(*v as i64)).collect()
        }
        VectorType::Int4 => {
            let s = cast_blob::<i32>(blob);
            s.iter().map(|v| Value::from(*v as i64)).collect()
        }
    };
    serde_json::to_string(&values).map_err(JsonError::Parse)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------ //
    // Helpers
    // ------------------------------------------------------------------ //

    /// Parse a JSON string produced by blob_to_json into a Vec<f64> for
    /// comparison, avoiding floating-point Display formatting issues.
    fn parse_json_floats(s: &str) -> Vec<f64> {
        let v: Vec<serde_json::Value> = serde_json::from_str(s).unwrap();
        v.iter().map(|x| x.as_f64().unwrap()).collect()
    }

    fn parse_json_ints(s: &str) -> Vec<i64> {
        let v: Vec<serde_json::Value> = serde_json::from_str(s).unwrap();
        v.iter().map(|x| x.as_i64().unwrap()).collect()
    }

    // ------------------------------------------------------------------ //
    // Round-trip tests for all 6 VectorType variants
    // ------------------------------------------------------------------ //

    #[test]
    fn round_trip_float2() {
        // f16 has ~3 significant decimal digits; use values exactly
        // representable in half precision.
        let json = "[1.0, -0.5, 0.25]";
        let blob = json_to_blob(json, VectorType::Float2).unwrap();
        // 3 elements * 2 bytes each
        assert_eq!(blob.len(), 6);
        let out = blob_to_json(&blob, VectorType::Float2).unwrap();
        let vals = parse_json_floats(&out);
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-3);
        assert!((vals[1] - (-0.5)).abs() < 1e-3);
        assert!((vals[2] - 0.25).abs() < 1e-3);
    }

    #[test]
    fn round_trip_float4() {
        let json = "[1.5, -2.25, 0.0, 100.0]";
        let blob = json_to_blob(json, VectorType::Float4).unwrap();
        assert_eq!(blob.len(), 16); // 4 elements * 4 bytes
        let out = blob_to_json(&blob, VectorType::Float4).unwrap();
        let vals = parse_json_floats(&out);
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 1.5).abs() < 1e-6);
        assert!((vals[1] - (-2.25)).abs() < 1e-6);
        assert!((vals[2] - 0.0).abs() < 1e-6);
        assert!((vals[3] - 100.0).abs() < 1e-3);
    }

    #[test]
    fn round_trip_float8() {
        let json = "[3.141592653589793, -2.718281828459045, 0.0]";
        let blob = json_to_blob(json, VectorType::Float8).unwrap();
        assert_eq!(blob.len(), 24); // 3 elements * 8 bytes
        let out = blob_to_json(&blob, VectorType::Float8).unwrap();
        let vals = parse_json_floats(&out);
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - std::f64::consts::PI).abs() < 1e-15);
        assert!((vals[1] - (-std::f64::consts::E)).abs() < 1e-15);
        assert!((vals[2] - 0.0).abs() < 1e-15);
    }

    #[test]
    fn round_trip_int1() {
        let json = "[0, 127, -128, -1, 42]";
        let blob = json_to_blob(json, VectorType::Int1).unwrap();
        assert_eq!(blob.len(), 5); // 5 elements * 1 byte
        let out = blob_to_json(&blob, VectorType::Int1).unwrap();
        assert_eq!(parse_json_ints(&out), vec![0, 127, -128, -1, 42]);
    }

    #[test]
    fn round_trip_int2() {
        let json = "[0, 32767, -32768, -1, 1000]";
        let blob = json_to_blob(json, VectorType::Int2).unwrap();
        assert_eq!(blob.len(), 10); // 5 elements * 2 bytes
        let out = blob_to_json(&blob, VectorType::Int2).unwrap();
        assert_eq!(parse_json_ints(&out), vec![0, 32767, -32768, -1, 1000]);
    }

    #[test]
    fn round_trip_int4() {
        let json = "[0, 2147483647, -2147483648, -1, 99999]";
        let blob = json_to_blob(json, VectorType::Int4).unwrap();
        assert_eq!(blob.len(), 20); // 5 elements * 4 bytes
        let out = blob_to_json(&blob, VectorType::Int4).unwrap();
        assert_eq!(
            parse_json_ints(&out),
            vec![0, 2147483647, -2147483648, -1, 99999]
        );
    }

    // ------------------------------------------------------------------ //
    // json_to_blob error cases
    // ------------------------------------------------------------------ //

    #[test]
    fn json_to_blob_rejects_object() {
        let err = json_to_blob("{\"x\": 1}", VectorType::Float4).unwrap_err();
        assert!(
            matches!(err, JsonError::NotAnArray),
            "expected NotAnArray, got {err}"
        );
    }

    #[test]
    fn json_to_blob_rejects_bare_number() {
        let err = json_to_blob("42", VectorType::Int4).unwrap_err();
        assert!(matches!(err, JsonError::NotAnArray));
    }

    #[test]
    fn json_to_blob_rejects_bare_string() {
        let err = json_to_blob("\"hello\"", VectorType::Float8).unwrap_err();
        assert!(matches!(err, JsonError::NotAnArray));
    }

    #[test]
    fn json_to_blob_rejects_malformed_json() {
        let err = json_to_blob("[1, 2,", VectorType::Float4).unwrap_err();
        assert!(matches!(err, JsonError::Parse(_)));
    }

    #[test]
    fn json_to_blob_rejects_string_element_float4() {
        let err = json_to_blob("[1.0, \"two\", 3.0]", VectorType::Float4).unwrap_err();
        assert!(matches!(err, JsonError::NonNumericElement(1)));
    }

    #[test]
    fn json_to_blob_rejects_string_element_int2() {
        let err = json_to_blob("[\"bad\", 2]", VectorType::Int2).unwrap_err();
        assert!(matches!(err, JsonError::NonNumericElement(0)));
    }

    #[test]
    fn json_to_blob_rejects_null_element_float2() {
        // null is not a number; expect NonNumericElement
        let err = json_to_blob("[1.0, null]", VectorType::Float2).unwrap_err();
        assert!(matches!(err, JsonError::NonNumericElement(1)));
    }

    // ------------------------------------------------------------------ //
    // Empty array / empty blob
    // ------------------------------------------------------------------ //

    #[test]
    fn json_to_blob_empty_array_all_types() {
        for vtype in [
            VectorType::Float2,
            VectorType::Float4,
            VectorType::Float8,
            VectorType::Int1,
            VectorType::Int2,
            VectorType::Int4,
        ] {
            let blob = json_to_blob("[]", vtype)
                .unwrap_or_else(|e| panic!("empty array failed for {vtype:?}: {e}"));
            assert!(
                blob.is_empty(),
                "expected empty blob for {vtype:?}, got {} bytes",
                blob.len()
            );
        }
    }

    #[test]
    fn blob_to_json_empty_blob_all_types() {
        // bytemuck::cast_slice requires the input slice to be aligned for the
        // target type even when it has zero length.  Build each empty blob via
        // the canonical helper so the pointer is correctly aligned.
        let empty_f16: &[f16] = &[];
        let empty_f32: &[f32] = &[];
        let empty_f64: &[f64] = &[];
        let empty_i8: &[i8] = &[];
        let empty_i16: &[i16] = &[];
        let empty_i32: &[i32] = &[];

        let cases: &[(&[u8], VectorType)] = &[
            (bytemuck::cast_slice(empty_f16), VectorType::Float2),
            (bytemuck::cast_slice(empty_f32), VectorType::Float4),
            (bytemuck::cast_slice(empty_f64), VectorType::Float8),
            (bytemuck::cast_slice(empty_i8), VectorType::Int1),
            (bytemuck::cast_slice(empty_i16), VectorType::Int2),
            (bytemuck::cast_slice(empty_i32), VectorType::Int4),
        ];

        for (blob, vtype) in cases {
            let out = blob_to_json(blob, *vtype)
                .unwrap_or_else(|e| panic!("empty blob failed for {vtype:?}: {e}"));
            assert_eq!(out, "[]", "expected '[]' for {vtype:?}, got {out:?}");
        }
    }

    // ------------------------------------------------------------------ //
    // Float precision
    // ------------------------------------------------------------------ //

    #[test]
    fn float4_precision_survives_round_trip() {
        // These values are exactly representable in f32.
        let inputs: Vec<f32> = vec![0.1, 0.2, 0.3, -0.1, 1.0 / 3.0];
        let blob = slice_to_blob(&inputs);
        let out = blob_to_json(&blob, VectorType::Float4).unwrap();
        let vals = parse_json_floats(&out);
        for (expected, actual) in inputs.iter().zip(vals.iter()) {
            // f32 -> f64 -> string -> f64 should recover the f32 value
            // within f32 epsilon.
            assert!(
                (actual - *expected as f64).abs() < f32::EPSILON as f64,
                "f32 precision lost: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn float8_full_precision_survives_round_trip() {
        // Use values whose magnitude keeps relative error within f64::EPSILON.
        // serde_json uses ryu, which emits the shortest decimal that round-
        // trips back to the same f64 bits, so the recovered value must equal
        // the input exactly (zero absolute error for normal magnitudes).
        let inputs: Vec<f64> = vec![
            1.0 / 7.0,
            std::f64::consts::PI,
            -std::f64::consts::SQRT_2,
            1.234_567_890_123_456_8e10,
        ];
        let blob = slice_to_blob(&inputs);
        let out = blob_to_json(&blob, VectorType::Float8).unwrap();
        let vals = parse_json_floats(&out);
        for (expected, actual) in inputs.iter().zip(vals.iter()) {
            // Ryu guarantees the shortest round-trip representation, so the
            // parsed value should be bit-identical to the original.
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "f64 bit pattern changed: expected {expected}, got {actual}"
            );
        }
    }

    // ------------------------------------------------------------------ //
    // Integer edge cases: negatives and zero
    // ------------------------------------------------------------------ //

    #[test]
    fn int1_negative_and_zero() {
        let json = "[-128, -1, 0, 1, 127]";
        let blob = json_to_blob(json, VectorType::Int1).unwrap();
        let slice = cast_blob::<i8>(&blob);
        assert_eq!(slice.as_ref(), &[-128_i8, -1, 0, 1, 127]);
    }

    #[test]
    fn int2_negative_and_zero() {
        let json = "[-32768, -100, 0, 100, 32767]";
        let blob = json_to_blob(json, VectorType::Int2).unwrap();
        let slice = cast_blob::<i16>(&blob);
        assert_eq!(slice.as_ref(), &[-32768_i16, -100, 0, 100, 32767]);
    }

    #[test]
    fn int4_negative_and_zero() {
        let json = "[-2147483648, -1, 0, 1, 2147483647]";
        let blob = json_to_blob(json, VectorType::Int4).unwrap();
        let slice = cast_blob::<i32>(&blob);
        assert_eq!(slice.as_ref(), &[-2147483648_i32, -1, 0, 1, 2147483647]);
    }

    // ------------------------------------------------------------------ //
    // Blob size invariants
    // ------------------------------------------------------------------ //

    #[test]
    fn blob_size_matches_element_size_times_count() {
        let cases: &[(&str, VectorType, usize, usize)] = &[
            ("[1.0]", VectorType::Float2, 1, 2),
            ("[1.0, 2.0]", VectorType::Float4, 2, 4),
            ("[1.0, 2.0, 3.0]", VectorType::Float8, 3, 8),
            ("[1]", VectorType::Int1, 1, 1),
            ("[1, 2]", VectorType::Int2, 2, 2),
            ("[1, 2, 3, 4]", VectorType::Int4, 4, 4),
        ];
        for (json, vtype, count, elem_bytes) in cases {
            let blob = json_to_blob(json, *vtype).unwrap();
            assert_eq!(
                blob.len(),
                count * elem_bytes,
                "{vtype:?}: expected {} bytes, got {}",
                count * elem_bytes,
                blob.len()
            );
        }
    }
}
