use sqlite_vector_rs::distance::DistanceMetric;
use sqlite_vector_rs::index::HnswIndex;
use sqlite_vector_rs::types::{VectorType, slice_to_blob};

#[test]
fn create_empty_index() {
    let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
    assert_eq!(idx.len(), 0);
    assert!(idx.is_empty());
}

#[test]
fn add_and_search_float4() {
    let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
    let v1: Vec<f32> = vec![1.0, 0.0, 0.0];
    let v2: Vec<f32> = vec![0.0, 1.0, 0.0];
    let v3: Vec<f32> = vec![0.0, 0.0, 1.0];
    idx.add(1, &slice_to_blob(&v1)).unwrap();
    idx.add(2, &slice_to_blob(&v2)).unwrap();
    idx.add(3, &slice_to_blob(&v3)).unwrap();

    let query: Vec<f32> = vec![1.0, 0.1, 0.0];
    let results = idx.search(&slice_to_blob(&query), 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1); // closest should be v1
}

#[test]
fn remove_vector() {
    let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
    let v1: Vec<f32> = vec![1.0, 0.0, 0.0];
    let v2: Vec<f32> = vec![0.0, 1.0, 0.0];
    idx.add(1, &slice_to_blob(&v1)).unwrap();
    idx.add(2, &slice_to_blob(&v2)).unwrap();
    assert_eq!(idx.len(), 2);

    idx.remove(1).unwrap();
    let query: Vec<f32> = vec![1.0, 0.0, 0.0];
    let results = idx.search(&slice_to_blob(&query), 2).unwrap();
    assert!(!results.iter().any(|(k, _)| *k == 1));
}

#[test]
fn serialize_round_trip() {
    let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
    let v1: Vec<f32> = vec![1.0, 0.0, 0.0];
    let v2: Vec<f32> = vec![0.0, 1.0, 0.0];
    idx.add(1, &slice_to_blob(&v1)).unwrap();
    idx.add(2, &slice_to_blob(&v2)).unwrap();

    let buf = idx.save_to_buffer().unwrap();
    assert!(!buf.is_empty());

    let idx2 = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
    idx2.load_from_buffer(&buf).unwrap();

    let query: Vec<f32> = vec![1.0, 0.0, 0.0];
    let results = idx2.search(&slice_to_blob(&query), 1).unwrap();
    assert_eq!(results[0].0, 1);
}

#[test]
fn search_empty_index() {
    let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
    let query: Vec<f32> = vec![1.0, 0.0, 0.0];
    let results = idx.search(&slice_to_blob(&query), 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn custom_hnsw_params() {
    use sqlite_vector_rs::index::HnswParams;
    let params = HnswParams {
        m: 32,
        ef_construction: 400,
        ef_search: 128,
    };
    let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::Cosine, Some(params)).unwrap();
    let v1: Vec<f32> = vec![1.0, 0.0, 0.0];
    idx.add(1, &slice_to_blob(&v1)).unwrap();
    assert_eq!(idx.len(), 1);
}
