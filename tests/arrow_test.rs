use sqlite_vector_rs::arrow_io::{arrow_ipc_to_vectors, vectors_to_arrow_ipc};
use sqlite_vector_rs::types::{VectorType, slice_to_blob};

#[test]
fn round_trip_float4_vectors() {
    let vectors: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let blobs: Vec<Vec<u8>> = vectors.iter().map(|v| slice_to_blob(v)).collect();

    let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Float4, 3).unwrap();
    assert!(!ipc.is_empty());

    let restored = arrow_ipc_to_vectors(&ipc, VectorType::Float4, 3).unwrap();
    assert_eq!(restored.len(), 3);
    assert_eq!(restored[0], blobs[0]);
    assert_eq!(restored[1], blobs[1]);
    assert_eq!(restored[2], blobs[2]);
}

#[test]
fn round_trip_int1_vectors() {
    let vectors: Vec<Vec<i8>> = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
    let blobs: Vec<Vec<u8>> = vectors.iter().map(|v| slice_to_blob(v)).collect();

    let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Int1, 4).unwrap();
    let restored = arrow_ipc_to_vectors(&ipc, VectorType::Int1, 4).unwrap();
    assert_eq!(restored, blobs);
}

#[test]
fn empty_vectors_round_trip() {
    let blobs: Vec<Vec<u8>> = vec![];
    let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Float4, 3).unwrap();
    let restored = arrow_ipc_to_vectors(&ipc, VectorType::Float4, 3).unwrap();
    assert!(restored.is_empty());
}
