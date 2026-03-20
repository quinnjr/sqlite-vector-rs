mod common;

use common::open_with_extension;
use sqlite_vector_rs::types::VectorType;

#[test]
fn vector_from_json_and_back() {
    let conn = open_with_extension();
    let json: String = conn.query_row(
        "SELECT vector_to_json(vector_from_json('[1.0, 2.0, 3.0]', 'float4'), 'float4')",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(json, "[1.0,2.0,3.0]");
}

#[test]
fn vector_distance_l2() {
    let conn = open_with_extension();
    let a = VectorType::Float4.slice_to_blob(&[1.0f32, 0.0, 0.0]);
    let b = VectorType::Float4.slice_to_blob(&[0.0f32, 1.0, 0.0]);
    let dist: f64 = conn.query_row(
        "SELECT vector_distance(?, ?, 'l2', 'float4')",
        [a.as_slice(), b.as_slice()],
        |row| row.get(0),
    ).unwrap();
    assert!((dist - 2.0).abs() < 1e-6);
}

#[test]
fn vector_dims() {
    let conn = open_with_extension();
    let v = VectorType::Float4.slice_to_blob(&[1.0f32, 2.0, 3.0, 4.0]);
    let dims: i64 = conn.query_row(
        "SELECT vector_dims(?, 'float4')",
        [v.as_slice()],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(dims, 4);
}
