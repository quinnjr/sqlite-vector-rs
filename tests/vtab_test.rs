mod common;

use common::open_with_extension;
use sqlite_vector_rs::types::VectorType;

#[test]
fn create_virtual_table() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE test_emb USING vector(dim=3, type=float4, metric=l2)"
    ).unwrap();
}

#[test]
fn insert_and_full_scan() {
    // Basic insert + full-table scan. KNN via knn_match() requires
    // FindFunctionVTab which is deferred to Task 14.
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)"
    ).unwrap();

    let v1 = VectorType::Float4.slice_to_blob(&[1.0f32, 0.0, 0.0]);
    let v2 = VectorType::Float4.slice_to_blob(&[0.0f32, 1.0, 0.0]);
    let v3 = VectorType::Float4.slice_to_blob(&[0.0f32, 0.0, 1.0]);

    conn.execute("INSERT INTO emb(vector) VALUES(?)", [v1.as_slice()]).unwrap();
    conn.execute("INSERT INTO emb(vector) VALUES(?)", [v2.as_slice()]).unwrap();
    conn.execute("INSERT INTO emb(vector) VALUES(?)", [v3.as_slice()]).unwrap();

    // Full scan should return all 3 rows
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM emb", [], |row| row.get(0)
    ).unwrap();
    assert_eq!(count, 3);
}

#[test]
fn reject_wrong_dimension() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)"
    ).unwrap();

    let wrong = VectorType::Float4.slice_to_blob(&[1.0f32, 0.0]); // 2-dim, expected 3
    let result = conn.execute("INSERT INTO emb(vector) VALUES(?)", [wrong.as_slice()]);
    assert!(result.is_err());
}

#[test]
fn reject_nan() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)"
    ).unwrap();

    let with_nan = VectorType::Float4.slice_to_blob(&[1.0f32, f32::NAN, 3.0]);
    let result = conn.execute("INSERT INTO emb(vector) VALUES(?)", [with_nan.as_slice()]);
    assert!(result.is_err());
}

#[test]
fn empty_table_scan() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)"
    ).unwrap();

    // Full table scan on empty table
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM emb", [], |row| row.get(0)
    ).unwrap();
    assert_eq!(count, 0);
}
