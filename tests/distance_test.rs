use sqlite_vector_rs::distance::{DistanceMetric, compute_distance};
use sqlite_vector_rs::types::{VectorType, slice_to_blob};

#[test]
fn l2_identical_vectors_float4() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let blob_a = slice_to_blob(&a);
    let dist =
        compute_distance(&blob_a, &blob_a, VectorType::Float4, DistanceMetric::L2, 3).unwrap();
    assert!((dist - 0.0).abs() < 1e-6);
}

#[test]
fn l2_known_distance_float4() {
    let a: Vec<f32> = vec![1.0, 0.0, 0.0];
    let b: Vec<f32> = vec![0.0, 1.0, 0.0];
    let blob_a = slice_to_blob(&a);
    let blob_b = slice_to_blob(&b);
    let dist =
        compute_distance(&blob_a, &blob_b, VectorType::Float4, DistanceMetric::L2, 3).unwrap();
    // Squared L2: (1-0)^2 + (0-1)^2 + (0-0)^2 = 2.0
    assert!((dist - 2.0).abs() < 1e-6);
}

#[test]
fn cosine_identical_float4() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let blob_a = slice_to_blob(&a);
    let dist = compute_distance(
        &blob_a,
        &blob_a,
        VectorType::Float4,
        DistanceMetric::Cosine,
        3,
    )
    .unwrap();
    assert!(dist.abs() < 1e-5);
}

#[test]
fn cosine_orthogonal_float4() {
    let a: Vec<f32> = vec![1.0, 0.0];
    let b: Vec<f32> = vec![0.0, 1.0];
    let blob_a = slice_to_blob(&a);
    let blob_b = slice_to_blob(&b);
    let dist = compute_distance(
        &blob_a,
        &blob_b,
        VectorType::Float4,
        DistanceMetric::Cosine,
        2,
    )
    .unwrap();
    // cosine distance = 1 - cos(90°) = 1.0
    assert!((dist - 1.0).abs() < 1e-5);
}

#[test]
fn inner_product_known_float4() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0, 6.0];
    let blob_a = slice_to_blob(&a);
    let blob_b = slice_to_blob(&b);
    let dist = compute_distance(
        &blob_a,
        &blob_b,
        VectorType::Float4,
        DistanceMetric::InnerProduct,
        3,
    )
    .unwrap();
    // -dot(a,b) = -(1*4 + 2*5 + 3*6) = -32.0
    assert!((dist - (-32.0)).abs() < 1e-6);
}

#[test]
fn parse_metric_names() {
    assert_eq!(DistanceMetric::from_name("l2").unwrap(), DistanceMetric::L2);
    assert_eq!(
        DistanceMetric::from_name("cosine").unwrap(),
        DistanceMetric::Cosine
    );
    assert_eq!(
        DistanceMetric::from_name("ip").unwrap(),
        DistanceMetric::InnerProduct
    );
    assert!(DistanceMetric::from_name("hamming").is_err());
}

#[test]
fn distance_dimension_mismatch() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![1.0, 2.0];
    let blob_a = slice_to_blob(&a);
    let blob_b = slice_to_blob(&b);
    assert!(compute_distance(&blob_a, &blob_b, VectorType::Float4, DistanceMetric::L2, 3).is_err());
}
