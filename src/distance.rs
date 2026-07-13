use std::fmt;

use half::f16;

use crate::types::{VectorType, cast_blob};

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
            let va = cast_blob::<f32>(a);
            let vb = cast_blob::<f32>(b);
            Ok(scalar_distance_pairs(
                va.iter().zip(vb.iter()).map(|(x, y)| (*x, *y)),
                metric,
            ))
        }
        VectorType::Float8 => {
            let va = cast_blob::<f64>(a);
            let vb = cast_blob::<f64>(b);
            Ok(scalar_distance_f64(&va[..], &vb[..], metric))
        }
        VectorType::Float2 => {
            let va = cast_blob::<f16>(a);
            let vb = cast_blob::<f16>(b);
            Ok(scalar_distance_pairs(
                va.iter()
                    .zip(vb.iter())
                    .map(|(x, y)| (x.to_f32(), y.to_f32())),
                metric,
            ))
        }
        VectorType::Int1 => {
            let va = cast_blob::<i8>(a);
            let vb = cast_blob::<i8>(b);
            Ok(scalar_distance_pairs(
                va.iter()
                    .zip(vb.iter())
                    .map(|(x, y)| (*x as f32, *y as f32)),
                metric,
            ))
        }
        VectorType::Int2 => {
            let va = cast_blob::<i16>(a);
            let vb = cast_blob::<i16>(b);
            Ok(scalar_distance_pairs(
                va.iter()
                    .zip(vb.iter())
                    .map(|(x, y)| (*x as f32, *y as f32)),
                metric,
            ))
        }
        VectorType::Int4 => {
            let va = cast_blob::<i32>(a);
            let vb = cast_blob::<i32>(b);
            Ok(scalar_distance_pairs(
                va.iter()
                    .zip(vb.iter())
                    .map(|(x, y)| (*x as f32, *y as f32)),
                metric,
            ))
        }
    }
}

fn scalar_distance_pairs(pairs: impl Iterator<Item = (f32, f32)>, metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::L2 => pairs
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>() as f64,
        DistanceMetric::InnerProduct => -(pairs.map(|(x, y)| x * y).sum::<f32>() as f64),
        DistanceMetric::Cosine => {
            let (dot, na, nb) = pairs.fold((0f32, 0f32, 0f32), |(d, a, b), (x, y)| {
                (d + x * y, a + x * x, b + y * y)
            });
            let denom = na.sqrt() * nb.sqrt();
            if denom == 0.0 {
                1.0
            } else {
                1.0 - (dot / denom) as f64
            }
        }
    }
}

fn scalar_distance_f64(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::L2 => a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum(),
        DistanceMetric::Cosine => {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            let denom = norm_a * norm_b;
            if denom == 0.0 {
                1.0
            } else {
                1.0 - (dot / denom)
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::cast_slice;

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    fn f32_blob(values: &[f32]) -> Vec<u8> {
        cast_slice(values).to_vec()
    }

    fn f64_blob(values: &[f64]) -> Vec<u8> {
        cast_slice(values).to_vec()
    }

    fn i32_blob(values: &[i32]) -> Vec<u8> {
        cast_slice(values).to_vec()
    }

    fn i8_blob(values: &[i8]) -> Vec<u8> {
        cast_slice(values).to_vec()
    }

    fn f16_blob(values: &[half::f16]) -> Vec<u8> {
        cast_slice(values).to_vec()
    }

    /// Assert two f64 values are within `eps` of each other.
    fn assert_approx(actual: f64, expected: f64, eps: f64) {
        assert!(
            (actual - expected).abs() < eps,
            "expected {expected} ± {eps}, got {actual}"
        );
    }

    // ----------------------------------------------------------------
    // DistanceMetric::from_name
    // ----------------------------------------------------------------

    #[test]
    fn from_name_valid_l2() {
        assert_eq!(DistanceMetric::from_name("l2").unwrap(), DistanceMetric::L2);
    }

    #[test]
    fn from_name_valid_cosine() {
        assert_eq!(
            DistanceMetric::from_name("cosine").unwrap(),
            DistanceMetric::Cosine
        );
    }

    #[test]
    fn from_name_valid_ip() {
        assert_eq!(
            DistanceMetric::from_name("ip").unwrap(),
            DistanceMetric::InnerProduct
        );
    }

    #[test]
    fn from_name_unknown_returns_error() {
        let err = DistanceMetric::from_name("manhattan").unwrap_err();
        assert!(
            matches!(err, DistanceError::UnknownMetric(ref s) if s == "manhattan"),
            "unexpected error variant: {err}"
        );
    }

    #[test]
    fn from_name_empty_string_returns_error() {
        assert!(DistanceMetric::from_name("").is_err());
    }

    #[test]
    fn from_name_case_sensitive() {
        // The matcher is lowercase-only; uppercase variants must be rejected.
        assert!(DistanceMetric::from_name("L2").is_err());
        assert!(DistanceMetric::from_name("Cosine").is_err());
        assert!(DistanceMetric::from_name("IP").is_err());
    }

    // ----------------------------------------------------------------
    // DistanceMetric::name — round-trip with from_name
    // ----------------------------------------------------------------

    #[test]
    fn name_round_trips_with_from_name() {
        let variants = [
            DistanceMetric::L2,
            DistanceMetric::Cosine,
            DistanceMetric::InnerProduct,
        ];
        for metric in variants {
            assert_eq!(
                DistanceMetric::from_name(metric.name()).unwrap(),
                metric,
                "round-trip failed for {:?}",
                metric
            );
        }
    }

    // ----------------------------------------------------------------
    // DistanceMetric::to_usearch — MetricKind mapping
    // ----------------------------------------------------------------

    #[test]
    fn to_usearch_l2_maps_to_l2sq() {
        assert_eq!(DistanceMetric::L2.to_usearch(), usearch::MetricKind::L2sq);
    }

    #[test]
    fn to_usearch_cosine_maps_to_cos() {
        assert_eq!(
            DistanceMetric::Cosine.to_usearch(),
            usearch::MetricKind::Cos
        );
    }

    #[test]
    fn to_usearch_ip_maps_to_ip() {
        assert_eq!(
            DistanceMetric::InnerProduct.to_usearch(),
            usearch::MetricKind::IP
        );
    }

    // ----------------------------------------------------------------
    // vtype_to_scalar_kind — all six types
    // ----------------------------------------------------------------

    #[test]
    fn vtype_to_scalar_kind_float2_is_f16() {
        assert_eq!(
            vtype_to_scalar_kind(VectorType::Float2),
            usearch::ScalarKind::F16
        );
    }

    #[test]
    fn vtype_to_scalar_kind_float4_is_f32() {
        assert_eq!(
            vtype_to_scalar_kind(VectorType::Float4),
            usearch::ScalarKind::F32
        );
    }

    #[test]
    fn vtype_to_scalar_kind_float8_is_f64() {
        assert_eq!(
            vtype_to_scalar_kind(VectorType::Float8),
            usearch::ScalarKind::F64
        );
    }

    #[test]
    fn vtype_to_scalar_kind_int1_is_i8() {
        assert_eq!(
            vtype_to_scalar_kind(VectorType::Int1),
            usearch::ScalarKind::I8
        );
    }

    #[test]
    fn vtype_to_scalar_kind_int2_quantizes_to_f32() {
        // i16 is not natively supported by usearch; it is quantized to f32.
        assert_eq!(
            vtype_to_scalar_kind(VectorType::Int2),
            usearch::ScalarKind::F32
        );
    }

    #[test]
    fn vtype_to_scalar_kind_int4_quantizes_to_f32() {
        // i32 is not natively supported by usearch; it is quantized to f32.
        assert_eq!(
            vtype_to_scalar_kind(VectorType::Int4),
            usearch::ScalarKind::F32
        );
    }

    // ----------------------------------------------------------------
    // compute_distance — dimension mismatch guard
    // ----------------------------------------------------------------

    #[test]
    fn compute_distance_dimension_mismatch_returns_error() {
        let a = f32_blob(&[1.0, 0.0, 0.0]);
        let b = f32_blob(&[1.0, 0.0]); // wrong length
        let err = compute_distance(&a, &b, VectorType::Float4, DistanceMetric::L2, 3).unwrap_err();
        assert!(
            matches!(err, DistanceError::DimensionMismatch),
            "expected DimensionMismatch, got {err}"
        );
    }

    // ----------------------------------------------------------------
    // compute_distance — Float4 / L2
    // ----------------------------------------------------------------

    #[test]
    fn float4_l2_identical_vectors_is_zero() {
        let v = f32_blob(&[1.0, 2.0, 3.0]);
        let d = compute_distance(&v, &v, VectorType::Float4, DistanceMetric::L2, 3).unwrap();
        assert_approx(d, 0.0, 1e-10);
    }

    #[test]
    fn float4_l2_orthogonal_unit_vectors_is_two() {
        // [1,0,0] vs [0,1,0]: squared L2 = (1-0)² + (0-1)² + (0-0)² = 2.0
        let a = f32_blob(&[1.0, 0.0, 0.0]);
        let b = f32_blob(&[0.0, 1.0, 0.0]);
        let d = compute_distance(&a, &b, VectorType::Float4, DistanceMetric::L2, 3).unwrap();
        assert_approx(d, 2.0, 1e-6);
    }

    #[test]
    fn float4_l2_known_distance() {
        // [3, 4] vs [0, 0]: squared L2 = 9 + 16 = 25
        let a = f32_blob(&[3.0, 4.0]);
        let b = f32_blob(&[0.0, 0.0]);
        let d = compute_distance(&a, &b, VectorType::Float4, DistanceMetric::L2, 2).unwrap();
        assert_approx(d, 25.0, 1e-5);
    }

    // ----------------------------------------------------------------
    // compute_distance — Float4 / Cosine
    // ----------------------------------------------------------------

    #[test]
    fn float4_cosine_identical_vectors_is_zero() {
        let v = f32_blob(&[1.0, 2.0, 3.0]);
        let d = compute_distance(&v, &v, VectorType::Float4, DistanceMetric::Cosine, 3).unwrap();
        assert_approx(d, 0.0, 1e-6);
    }

    #[test]
    fn float4_cosine_orthogonal_vectors_is_one() {
        // [1,0] and [0,1] have zero dot product → cosine distance = 1.
        let a = f32_blob(&[1.0, 0.0]);
        let b = f32_blob(&[0.0, 1.0]);
        let d = compute_distance(&a, &b, VectorType::Float4, DistanceMetric::Cosine, 2).unwrap();
        assert_approx(d, 1.0, 1e-6);
    }

    #[test]
    fn float4_cosine_antiparallel_vectors_is_two() {
        // [1,0] and [-1,0] are antiparallel → cosine distance = 2.
        let a = f32_blob(&[1.0, 0.0]);
        let b = f32_blob(&[-1.0, 0.0]);
        let d = compute_distance(&a, &b, VectorType::Float4, DistanceMetric::Cosine, 2).unwrap();
        assert_approx(d, 2.0, 1e-6);
    }

    #[test]
    fn float4_cosine_zero_vector_returns_one() {
        // When the denominator is zero the implementation returns 1.0.
        let a = f32_blob(&[0.0, 0.0, 0.0]);
        let b = f32_blob(&[0.0, 0.0, 0.0]);
        let d = compute_distance(&a, &b, VectorType::Float4, DistanceMetric::Cosine, 3).unwrap();
        assert_approx(d, 1.0, 1e-10);
    }

    // ----------------------------------------------------------------
    // compute_distance — Float4 / InnerProduct
    // ----------------------------------------------------------------

    #[test]
    fn float4_ip_unit_vectors_dot_product() {
        // [1,0,0]·[0,0,1] = 0  →  -(0) = 0.0
        let a = f32_blob(&[1.0, 0.0, 0.0]);
        let b = f32_blob(&[0.0, 0.0, 1.0]);
        let d =
            compute_distance(&a, &b, VectorType::Float4, DistanceMetric::InnerProduct, 3).unwrap();
        assert_approx(d, 0.0, 1e-6);
    }

    #[test]
    fn float4_ip_parallel_unit_vectors() {
        // [1,0]·[1,0] = 1  →  -(1) = -1.0  (higher similarity = more negative)
        let a = f32_blob(&[1.0, 0.0]);
        let b = f32_blob(&[1.0, 0.0]);
        let d =
            compute_distance(&a, &b, VectorType::Float4, DistanceMetric::InnerProduct, 2).unwrap();
        assert_approx(d, -1.0, 1e-6);
    }

    #[test]
    fn float4_ip_known_value() {
        // [1,2]·[3,4] = 11  →  -(11) = -11.0
        let a = f32_blob(&[1.0, 2.0]);
        let b = f32_blob(&[3.0, 4.0]);
        let d =
            compute_distance(&a, &b, VectorType::Float4, DistanceMetric::InnerProduct, 2).unwrap();
        assert_approx(d, -11.0, 1e-5);
    }

    // ----------------------------------------------------------------
    // compute_distance — Float8
    // ----------------------------------------------------------------

    #[test]
    fn float8_l2_identical_vectors_is_zero() {
        let v = f64_blob(&[1.0, 2.0, 3.0]);
        let d = compute_distance(&v, &v, VectorType::Float8, DistanceMetric::L2, 3).unwrap();
        assert_approx(d, 0.0, 1e-15);
    }

    #[test]
    fn float8_l2_known_distance() {
        // [1, 1] vs [4, 5]: (1-4)² + (1-5)² = 9 + 16 = 25
        let a = f64_blob(&[1.0, 1.0]);
        let b = f64_blob(&[4.0, 5.0]);
        let d = compute_distance(&a, &b, VectorType::Float8, DistanceMetric::L2, 2).unwrap();
        assert_approx(d, 25.0, 1e-12);
    }

    #[test]
    fn float8_cosine_orthogonal_is_one() {
        let a = f64_blob(&[1.0, 0.0]);
        let b = f64_blob(&[0.0, 1.0]);
        let d = compute_distance(&a, &b, VectorType::Float8, DistanceMetric::Cosine, 2).unwrap();
        assert_approx(d, 1.0, 1e-14);
    }

    #[test]
    fn float8_cosine_zero_vector_returns_one() {
        let a = f64_blob(&[0.0, 0.0]);
        let b = f64_blob(&[0.0, 0.0]);
        let d = compute_distance(&a, &b, VectorType::Float8, DistanceMetric::Cosine, 2).unwrap();
        assert_approx(d, 1.0, 1e-15);
    }

    #[test]
    fn float8_ip_known_value() {
        // [2,3]·[4,5] = 23  →  -(23) = -23.0
        let a = f64_blob(&[2.0, 3.0]);
        let b = f64_blob(&[4.0, 5.0]);
        let d =
            compute_distance(&a, &b, VectorType::Float8, DistanceMetric::InnerProduct, 2).unwrap();
        assert_approx(d, -23.0, 1e-12);
    }

    // ----------------------------------------------------------------
    // compute_distance — Int4
    // ----------------------------------------------------------------

    #[test]
    fn int4_l2_identical_vectors_is_zero() {
        let v = i32_blob(&[10, -5, 3]);
        let d = compute_distance(&v, &v, VectorType::Int4, DistanceMetric::L2, 3).unwrap();
        assert_approx(d, 0.0, 1e-10);
    }

    #[test]
    fn int4_l2_known_distance() {
        // [0, 0] vs [3, 4]: (3-0)² + (4-0)² = 9 + 16 = 25, cast to f32 first
        let a = i32_blob(&[0, 0]);
        let b = i32_blob(&[3, 4]);
        let d = compute_distance(&a, &b, VectorType::Int4, DistanceMetric::L2, 2).unwrap();
        assert_approx(d, 25.0, 1e-5);
    }

    #[test]
    fn int4_cosine_orthogonal_is_one() {
        let a = i32_blob(&[1, 0]);
        let b = i32_blob(&[0, 1]);
        let d = compute_distance(&a, &b, VectorType::Int4, DistanceMetric::Cosine, 2).unwrap();
        assert_approx(d, 1.0, 1e-6);
    }

    #[test]
    fn int4_ip_known_value() {
        // [1, 2]·[3, 4] = 11  →  -(11) = -11.0  (int4 is cast to f32)
        let a = i32_blob(&[1, 2]);
        let b = i32_blob(&[3, 4]);
        let d =
            compute_distance(&a, &b, VectorType::Int4, DistanceMetric::InnerProduct, 2).unwrap();
        assert_approx(d, -11.0, 1e-5);
    }

    // ----------------------------------------------------------------
    // compute_distance — Int1 (i8)
    // ----------------------------------------------------------------

    #[test]
    fn int1_l2_known_distance() {
        // [3, 4] vs [0, 0] cast to f32: 9 + 16 = 25
        let a = i8_blob(&[3, 4]);
        let b = i8_blob(&[0, 0]);
        let d = compute_distance(&a, &b, VectorType::Int1, DistanceMetric::L2, 2).unwrap();
        assert_approx(d, 25.0, 1e-5);
    }

    // ----------------------------------------------------------------
    // compute_distance — Float2 (f16)
    // ----------------------------------------------------------------

    #[test]
    fn float2_cosine_orthogonal_is_one() {
        let a = f16_blob(&[half::f16::from_f32(1.0), half::f16::from_f32(0.0)]);
        let b = f16_blob(&[half::f16::from_f32(0.0), half::f16::from_f32(1.0)]);
        let d = compute_distance(&a, &b, VectorType::Float2, DistanceMetric::Cosine, 2).unwrap();
        // f16 has limited precision; use a looser tolerance.
        assert_approx(d, 1.0, 1e-3);
    }

    #[test]
    fn float2_l2_identical_vectors_is_zero() {
        let v = f16_blob(&[
            half::f16::from_f32(1.0),
            half::f16::from_f32(-2.0),
            half::f16::from_f32(0.5),
        ]);
        let d = compute_distance(&v, &v, VectorType::Float2, DistanceMetric::L2, 3).unwrap();
        assert_approx(d, 0.0, 1e-6);
    }

    #[test]
    fn float2_matches_f32_reference() {
        let a: Vec<half::f16> = [0.5f32, -1.0, 2.0]
            .iter()
            .map(|v| half::f16::from_f32(*v))
            .collect();
        let b: Vec<half::f16> = [1.5f32, 0.25, -2.0]
            .iter()
            .map(|v| half::f16::from_f32(*v))
            .collect();
        let blob_a = cast_slice(&a).to_vec();
        let blob_b = cast_slice(&b).to_vec();
        for metric in [
            DistanceMetric::L2,
            DistanceMetric::Cosine,
            DistanceMetric::InnerProduct,
        ] {
            let d16 = compute_distance(&blob_a, &blob_b, VectorType::Float2, metric, 3).unwrap();
            let fa: Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
            let fb: Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
            let dref = compute_distance(
                cast_slice(&fa),
                cast_slice(&fb),
                VectorType::Float4,
                metric,
                3,
            )
            .unwrap();
            assert!((d16 - dref).abs() < 1e-3, "{metric:?}: {d16} vs {dref}");
        }
    }
}
