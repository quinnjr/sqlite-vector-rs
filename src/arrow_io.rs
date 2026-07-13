use std::io::Cursor;
use std::sync::Arc;

use arrow_array::*;
use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};

use crate::types::{VectorType, cast_blob, slice_to_blob};

#[derive(Debug)]
pub struct ArrowError(pub String);

impl std::fmt::Display for ArrowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "arrow error: {}", self.0)
    }
}

impl std::error::Error for ArrowError {}

impl From<arrow_schema::ArrowError> for ArrowError {
    fn from(e: arrow_schema::ArrowError) -> Self {
        Self(e.to_string())
    }
}

/// Convert a list of raw vector blobs into an Arrow IPC byte buffer.
pub fn vectors_to_arrow_ipc(
    blobs: &[Vec<u8>],
    vtype: VectorType,
    dim: usize,
) -> Result<Vec<u8>, ArrowError> {
    let (inner_dt, values_array) = build_values_array(blobs, vtype, dim)?;

    let field = Arc::new(Field::new("item", inner_dt, true));
    let list_array = FixedSizeListArray::new(field, dim as i32, values_array, None);

    let schema = Schema::new(vec![Field::new(
        "vector",
        list_array.data_type().clone(),
        false,
    )]);
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(list_array)])
        .map_err(|e| ArrowError(e.to_string()))?;

    let mut buf = Vec::new();
    let mut writer =
        StreamWriter::try_new(&mut buf, &schema).map_err(|e| ArrowError(e.to_string()))?;
    writer
        .write(&batch)
        .map_err(|e| ArrowError(e.to_string()))?;
    writer.finish().map_err(|e| ArrowError(e.to_string()))?;
    drop(writer);

    Ok(buf)
}

/// Parse an Arrow IPC byte buffer back into raw vector blobs.
///
/// If `dim` is 0 the dimension is inferred from the FixedSizeListArray's
/// `value_length()`, which is encoded in the Arrow schema.
pub fn arrow_ipc_to_vectors(
    ipc_bytes: &[u8],
    vtype: VectorType,
    dim: usize,
) -> Result<Vec<Vec<u8>>, ArrowError> {
    let reader = StreamReader::try_new(Cursor::new(ipc_bytes), None)
        .map_err(|e| ArrowError(e.to_string()))?;

    let mut all_blobs = Vec::new();
    for batch_result in reader {
        let batch = batch_result.map_err(|e| ArrowError(e.to_string()))?;
        let list_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| ArrowError("expected FixedSizeListArray".into()))?;

        // Auto-detect dimension from the Arrow schema when caller passes 0.
        let effective_dim = if dim == 0 {
            list_col.value_length() as usize
        } else {
            dim
        };

        for i in 0..list_col.len() {
            let sub = list_col.value(i);
            let blob = extract_blob_from_array(&sub, vtype, effective_dim)?;
            all_blobs.push(blob);
        }
    }

    Ok(all_blobs)
}

fn build_values_array(
    blobs: &[Vec<u8>],
    vtype: VectorType,
    dim: usize,
) -> Result<(DataType, ArrayRef), ArrowError> {
    let total_elements = blobs.len() * dim;
    match vtype {
        VectorType::Float4 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v = cast_blob::<f32>(blob);
                flat.extend_from_slice(&v[..]);
            }
            Ok((DataType::Float32, Arc::new(Float32Array::from(flat))))
        }
        VectorType::Float8 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v = cast_blob::<f64>(blob);
                flat.extend_from_slice(&v[..]);
            }
            Ok((DataType::Float64, Arc::new(Float64Array::from(flat))))
        }
        VectorType::Float2 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v = cast_blob::<half::f16>(blob);
                flat.extend(v.iter().copied());
            }
            Ok((DataType::Float16, Arc::new(Float16Array::from(flat))))
        }
        VectorType::Int1 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v = cast_blob::<i8>(blob);
                flat.extend_from_slice(&v[..]);
            }
            Ok((DataType::Int8, Arc::new(Int8Array::from(flat))))
        }
        VectorType::Int2 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v = cast_blob::<i16>(blob);
                flat.extend_from_slice(&v[..]);
            }
            Ok((DataType::Int16, Arc::new(Int16Array::from(flat))))
        }
        VectorType::Int4 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v = cast_blob::<i32>(blob);
                flat.extend_from_slice(&v[..]);
            }
            Ok((DataType::Int32, Arc::new(Int32Array::from(flat))))
        }
    }
}

fn extract_blob_from_array(
    array: &ArrayRef,
    vtype: VectorType,
    dim: usize,
) -> Result<Vec<u8>, ArrowError> {
    match vtype {
        VectorType::Float4 => {
            let a = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| ArrowError("expected Float32Array".into()))?;
            let values: Vec<f32> = (0..dim).map(|i| a.value(i)).collect();
            Ok(slice_to_blob(&values))
        }
        VectorType::Float8 => {
            let a = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| ArrowError("expected Float64Array".into()))?;
            let values: Vec<f64> = (0..dim).map(|i| a.value(i)).collect();
            Ok(slice_to_blob(&values))
        }
        VectorType::Float2 => {
            let a = array
                .as_any()
                .downcast_ref::<Float16Array>()
                .ok_or_else(|| ArrowError("expected Float16Array".into()))?;
            let values: Vec<half::f16> = (0..dim).map(|i| a.value(i)).collect();
            Ok(slice_to_blob(&values))
        }
        VectorType::Int1 => {
            let a = array
                .as_any()
                .downcast_ref::<Int8Array>()
                .ok_or_else(|| ArrowError("expected Int8Array".into()))?;
            let values: Vec<i8> = (0..dim).map(|i| a.value(i)).collect();
            Ok(slice_to_blob(&values))
        }
        VectorType::Int2 => {
            let a = array
                .as_any()
                .downcast_ref::<Int16Array>()
                .ok_or_else(|| ArrowError("expected Int16Array".into()))?;
            let values: Vec<i16> = (0..dim).map(|i| a.value(i)).collect();
            Ok(slice_to_blob(&values))
        }
        VectorType::Int4 => {
            let a = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| ArrowError("expected Int32Array".into()))?;
            let values: Vec<i32> = (0..dim).map(|i| a.value(i)).collect();
            Ok(slice_to_blob(&values))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VectorType;
    use half::f16;

    // ----------------------------------------------------------------
    // helpers
    // ----------------------------------------------------------------

    /// Build a Float4 blob from a slice of f32 values.
    fn f32_blob(values: &[f32]) -> Vec<u8> {
        slice_to_blob(values)
    }

    /// Build a Float8 blob from a slice of f64 values.
    fn f64_blob(values: &[f64]) -> Vec<u8> {
        slice_to_blob(values)
    }

    /// Build an Int1 blob from a slice of i8 values.
    fn i8_blob(values: &[i8]) -> Vec<u8> {
        slice_to_blob(values)
    }

    /// Build an Int2 blob from a slice of i16 values.
    fn i16_blob(values: &[i16]) -> Vec<u8> {
        slice_to_blob(values)
    }

    /// Build an Int4 blob from a slice of i32 values.
    fn i32_blob(values: &[i32]) -> Vec<u8> {
        slice_to_blob(values)
    }

    /// Build a Float2 blob from a slice of f16 values.
    fn f16_blob(values: &[f16]) -> Vec<u8> {
        slice_to_blob(values)
    }

    // ----------------------------------------------------------------
    // 1. Round-trip Float4
    // ----------------------------------------------------------------

    #[test]
    fn round_trip_float4() {
        let blobs = vec![f32_blob(&[1.0_f32, 2.0, 3.0])];
        let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Float4, 3).unwrap();
        let result = arrow_ipc_to_vectors(&ipc, VectorType::Float4, 3).unwrap();
        assert_eq!(result, blobs);
    }

    // ----------------------------------------------------------------
    // 2. Round-trip Float8
    // ----------------------------------------------------------------

    #[test]
    fn round_trip_float8() {
        let blobs = vec![f64_blob(&[1.0_f64, -2.5, 3.125])];
        let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Float8, 3).unwrap();
        let result = arrow_ipc_to_vectors(&ipc, VectorType::Float8, 3).unwrap();
        assert_eq!(result, blobs);
    }

    // ----------------------------------------------------------------
    // 3. Round-trip Int1, Int2, Int4
    // ----------------------------------------------------------------

    #[test]
    fn round_trip_int1() {
        let blobs = vec![i8_blob(&[i8::MIN, 0, i8::MAX])];
        let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Int1, 3).unwrap();
        let result = arrow_ipc_to_vectors(&ipc, VectorType::Int1, 3).unwrap();
        assert_eq!(result, blobs);
    }

    #[test]
    fn round_trip_int2() {
        let blobs = vec![i16_blob(&[i16::MIN, 0, i16::MAX])];
        let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Int2, 3).unwrap();
        let result = arrow_ipc_to_vectors(&ipc, VectorType::Int2, 3).unwrap();
        assert_eq!(result, blobs);
    }

    #[test]
    fn round_trip_int4() {
        let blobs = vec![i32_blob(&[i32::MIN, 0, i32::MAX])];
        let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Int4, 3).unwrap();
        let result = arrow_ipc_to_vectors(&ipc, VectorType::Int4, 3).unwrap();
        assert_eq!(result, blobs);
    }

    // ----------------------------------------------------------------
    // 4. Round-trip Float2 (half precision)
    // ----------------------------------------------------------------

    #[test]
    fn round_trip_float2() {
        let values = vec![f16::from_f32(1.0), f16::from_f32(-0.5), f16::from_f32(0.25)];
        let blobs = vec![f16_blob(&values)];
        let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Float2, 3).unwrap();
        let result = arrow_ipc_to_vectors(&ipc, VectorType::Float2, 3).unwrap();
        assert_eq!(result, blobs);
    }

    // ----------------------------------------------------------------
    // 5. Empty blobs list → empty IPC → empty result
    // ----------------------------------------------------------------

    #[test]
    fn empty_blobs_round_trip() {
        // vectors_to_arrow_ipc with an empty slice should produce valid IPC
        // that decodes back to an empty vector list.
        //
        // Note: dim must be non-zero for Arrow schema encoding even when
        // there are no rows; we use dim=4 as a representative dimension.
        let blobs: Vec<Vec<u8>> = vec![];
        let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Float4, 4).unwrap();
        assert!(
            !ipc.is_empty(),
            "IPC buffer must contain at least the schema header"
        );
        let result = arrow_ipc_to_vectors(&ipc, VectorType::Float4, 4).unwrap();
        assert!(result.is_empty());
    }

    // ----------------------------------------------------------------
    // 6. Dim auto-detection: encode with dim=3, decode with dim=0
    // ----------------------------------------------------------------

    #[test]
    fn dim_auto_detection_float4() {
        let blobs = vec![f32_blob(&[10.0_f32, 20.0, 30.0])];
        let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Float4, 3).unwrap();
        // Pass dim=0 to trigger auto-detection from the Arrow schema.
        let result = arrow_ipc_to_vectors(&ipc, VectorType::Float4, 0).unwrap();
        assert_eq!(result, blobs);
    }

    #[test]
    fn dim_auto_detection_int2() {
        let blobs = vec![i16_blob(&[1_i16, 2, 3])];
        let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Int2, 3).unwrap();
        let result = arrow_ipc_to_vectors(&ipc, VectorType::Int2, 0).unwrap();
        assert_eq!(result, blobs);
    }

    // ----------------------------------------------------------------
    // 7. Multiple vectors round-trip (5 vectors)
    // ----------------------------------------------------------------

    #[test]
    fn multiple_vectors_float4() {
        let blobs: Vec<Vec<u8>> = (0..5_u32)
            .map(|i| {
                let base = i as f32;
                f32_blob(&[base, base + 1.0, base + 2.0, base + 3.0])
            })
            .collect();
        let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Float4, 4).unwrap();
        let result = arrow_ipc_to_vectors(&ipc, VectorType::Float4, 4).unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result, blobs);
    }

    #[test]
    fn multiple_vectors_int4() {
        let blobs: Vec<Vec<u8>> = (0..5_i32)
            .map(|i| i32_blob(&[i * 10, i * 10 + 1, i * 10 + 2]))
            .collect();
        let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Int4, 3).unwrap();
        let result = arrow_ipc_to_vectors(&ipc, VectorType::Int4, 3).unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result, blobs);
    }

    // ----------------------------------------------------------------
    // 8. Single vector round-trip
    // ----------------------------------------------------------------

    #[test]
    fn single_vector_float8() {
        let blobs = vec![f64_blob(&[std::f64::consts::PI, std::f64::consts::E])];
        let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Float8, 2).unwrap();
        let result = arrow_ipc_to_vectors(&ipc, VectorType::Float8, 2).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result, blobs);
    }

    #[test]
    fn single_vector_int1() {
        let blobs = vec![i8_blob(&[-1_i8, 0, 127])];
        let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Int1, 3).unwrap();
        let result = arrow_ipc_to_vectors(&ipc, VectorType::Int1, 3).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result, blobs);
    }
}
