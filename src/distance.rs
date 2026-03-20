use std::fmt;

use bytemuck::cast_slice;
use half::f16;

use crate::types::VectorType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    L2,
    Cosine,
    InnerProduct,
}

impl DistanceMetric {
    pub fn from_name(name: &str) -> Result<Self, DistanceError> {
        match name {
            "l2" => Ok(Self::L2),
            "cosine" => Ok(Self::Cosine),
            "ip" => Ok(Self::InnerProduct),
            other => Err(DistanceError::UnknownMetric(other.to_string())),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Cosine => "cosine",
            Self::InnerProduct => "ip",
        }
    }

    /// Convert to usearch MetricKind.
    pub fn to_usearch(&self) -> usearch::MetricKind {
        match self {
            Self::L2 => usearch::MetricKind::L2sq,
            Self::Cosine => usearch::MetricKind::Cos,
            Self::InnerProduct => usearch::MetricKind::IP,
        }
    }
}

#[derive(Debug)]
pub enum DistanceError {
    UnknownMetric(String),
    DimensionMismatch,
    Usearch(String),
}

impl fmt::Display for DistanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownMetric(name) => write!(f, "unknown metric: {name}"),
            Self::DimensionMismatch => write!(f, "vector dimensions do not match"),
            Self::Usearch(e) => write!(f, "usearch error: {e}"),
        }
    }
}

impl std::error::Error for DistanceError {}

/// Compute distance between two vector blobs.
///
/// Both blobs must be the same type and dimension. For int2/int4 types,
/// values are cast to f32 before computation since usearch only supports
/// f32, f64, f16, and i8 natively.
pub fn compute_distance(
    a: &[u8],
    b: &[u8],
    vtype: VectorType,
    metric: DistanceMetric,
    dim: usize,
) -> Result<f64, DistanceError> {
    let expected_size = vtype.blob_size(dim);
    if a.len() != expected_size || b.len() != expected_size {
        return Err(DistanceError::DimensionMismatch);
    }

    match vtype {
        VectorType::Float4 => {
            let va: &[f32] = cast_slice(a);
            let vb: &[f32] = cast_slice(b);
            Ok(scalar_distance(va, vb, metric))
        }
        VectorType::Float8 => {
            let va: &[f64] = cast_slice(a);
            let vb: &[f64] = cast_slice(b);
            Ok(scalar_distance_f64(va, vb, metric))
        }
        VectorType::Float2 => {
            let va: &[f16] = cast_slice(a);
            let vb: &[f16] = cast_slice(b);
            let fa: Vec<f32> = va.iter().map(|v| v.to_f32()).collect();
            let fb: Vec<f32> = vb.iter().map(|v| v.to_f32()).collect();
            Ok(scalar_distance(&fa, &fb, metric))
        }
        VectorType::Int1 => {
            let va: &[i8] = cast_slice(a);
            let vb: &[i8] = cast_slice(b);
            let fa: Vec<f32> = va.iter().map(|v| *v as f32).collect();
            let fb: Vec<f32> = vb.iter().map(|v| *v as f32).collect();
            Ok(scalar_distance(&fa, &fb, metric))
        }
        VectorType::Int2 => {
            let va: &[i16] = cast_slice(a);
            let vb: &[i16] = cast_slice(b);
            let fa: Vec<f32> = va.iter().map(|v| *v as f32).collect();
            let fb: Vec<f32> = vb.iter().map(|v| *v as f32).collect();
            Ok(scalar_distance(&fa, &fb, metric))
        }
        VectorType::Int4 => {
            let va: &[i32] = cast_slice(a);
            let vb: &[i32] = cast_slice(b);
            let fa: Vec<f32> = va.iter().map(|v| *v as f32).collect();
            let fb: Vec<f32> = vb.iter().map(|v| *v as f32).collect();
            Ok(scalar_distance(&fa, &fb, metric))
        }
    }
}

fn scalar_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::L2 => {
            a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() as f64
        }
        DistanceMetric::Cosine => {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let denom = norm_a * norm_b;
            if denom == 0.0 { 1.0 } else { 1.0 - (dot / denom) as f64 }
        }
        DistanceMetric::InnerProduct => {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            -(dot as f64)
        }
    }
}

fn scalar_distance_f64(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::L2 => {
            a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
        }
        DistanceMetric::Cosine => {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            let denom = norm_a * norm_b;
            if denom == 0.0 { 1.0 } else { 1.0 - (dot / denom) }
        }
        DistanceMetric::InnerProduct => {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            -dot
        }
    }
}

/// Map VectorType to usearch ScalarKind for index creation.
pub fn vtype_to_scalar_kind(vtype: VectorType) -> usearch::ScalarKind {
    match vtype {
        VectorType::Float2 => usearch::ScalarKind::F16,
        VectorType::Float4 => usearch::ScalarKind::F32,
        VectorType::Float8 => usearch::ScalarKind::F64,
        VectorType::Int1 => usearch::ScalarKind::I8,
        // i16/i32 not natively supported by usearch, quantize to f32
        VectorType::Int2 | VectorType::Int4 => usearch::ScalarKind::F32,
    }
}
