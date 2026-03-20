use std::io::Cursor;
use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch};
use arrow_array::*;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};

use crate::types::VectorType;

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
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(list_array)],
    ).map_err(|e| ArrowError(e.to_string()))?;

    let mut buf = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buf, &schema)
        .map_err(|e| ArrowError(e.to_string()))?;
    writer.write(&batch).map_err(|e| ArrowError(e.to_string()))?;
    writer.finish().map_err(|e| ArrowError(e.to_string()))?;
    drop(writer);

    Ok(buf)
}

/// Parse an Arrow IPC byte buffer back into raw vector blobs.
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
        let list_col = batch.column(0)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| ArrowError("expected FixedSizeListArray".into()))?;

        for i in 0..list_col.len() {
            let sub = list_col.value(i);
            let blob = extract_blob_from_array(&sub, vtype, dim)?;
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
                let v: &[f32] = vtype.blob_to_slice(blob);
                flat.extend_from_slice(v);
            }
            Ok((DataType::Float32, Arc::new(Float32Array::from(flat))))
        }
        VectorType::Float8 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v: &[f64] = vtype.blob_to_slice(blob);
                flat.extend_from_slice(v);
            }
            Ok((DataType::Float64, Arc::new(Float64Array::from(flat))))
        }
        VectorType::Float2 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v: &[half::f16] = vtype.blob_to_slice(blob);
                flat.extend(v.iter().copied());
            }
            Ok((DataType::Float16, Arc::new(Float16Array::from(flat))))
        }
        VectorType::Int1 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v: &[i8] = vtype.blob_to_slice(blob);
                flat.extend_from_slice(v);
            }
            Ok((DataType::Int8, Arc::new(Int8Array::from(flat))))
        }
        VectorType::Int2 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v: &[i16] = vtype.blob_to_slice(blob);
                flat.extend_from_slice(v);
            }
            Ok((DataType::Int16, Arc::new(Int16Array::from(flat))))
        }
        VectorType::Int4 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v: &[i32] = vtype.blob_to_slice(blob);
                flat.extend_from_slice(v);
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
            let a = array.as_any().downcast_ref::<Float32Array>()
                .ok_or_else(|| ArrowError("expected Float32Array".into()))?;
            let values: Vec<f32> = (0..dim).map(|i| a.value(i)).collect();
            Ok(vtype.slice_to_blob(&values))
        }
        VectorType::Float8 => {
            let a = array.as_any().downcast_ref::<Float64Array>()
                .ok_or_else(|| ArrowError("expected Float64Array".into()))?;
            let values: Vec<f64> = (0..dim).map(|i| a.value(i)).collect();
            Ok(vtype.slice_to_blob(&values))
        }
        VectorType::Float2 => {
            let a = array.as_any().downcast_ref::<Float16Array>()
                .ok_or_else(|| ArrowError("expected Float16Array".into()))?;
            let values: Vec<half::f16> = (0..dim).map(|i| a.value(i)).collect();
            Ok(vtype.slice_to_blob(&values))
        }
        VectorType::Int1 => {
            let a = array.as_any().downcast_ref::<Int8Array>()
                .ok_or_else(|| ArrowError("expected Int8Array".into()))?;
            let values: Vec<i8> = (0..dim).map(|i| a.value(i)).collect();
            Ok(vtype.slice_to_blob(&values))
        }
        VectorType::Int2 => {
            let a = array.as_any().downcast_ref::<Int16Array>()
                .ok_or_else(|| ArrowError("expected Int16Array".into()))?;
            let values: Vec<i16> = (0..dim).map(|i| a.value(i)).collect();
            Ok(vtype.slice_to_blob(&values))
        }
        VectorType::Int4 => {
            let a = array.as_any().downcast_ref::<Int32Array>()
                .ok_or_else(|| ArrowError("expected Int32Array".into()))?;
            let values: Vec<i32> = (0..dim).map(|i| a.value(i)).collect();
            Ok(vtype.slice_to_blob(&values))
        }
    }
}
