mod common;

use common::open_with_extension;
use rusqlite::params;
use sqlite_vector_rs::types::slice_to_blob;

#[test]
fn create_virtual_table() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE test_emb USING vector(dim=3, type=float4, metric=l2)")
        .unwrap();
}

#[test]
fn insert_and_full_scan() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)")
        .unwrap();

    let v1 = slice_to_blob(&[1.0f32, 0.0, 0.0]);
    let v2 = slice_to_blob(&[0.0f32, 1.0, 0.0]);
    let v3 = slice_to_blob(&[0.0f32, 0.0, 1.0]);

    conn.execute("INSERT INTO emb(vector) VALUES(?)", [v1.as_slice()])
        .unwrap();
    conn.execute("INSERT INTO emb(vector) VALUES(?)", [v2.as_slice()])
        .unwrap();
    conn.execute("INSERT INTO emb(vector) VALUES(?)", [v3.as_slice()])
        .unwrap();

    // Full scan should return all 3 rows
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM emb", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 3);
}

#[test]
fn reject_wrong_dimension() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)")
        .unwrap();

    let wrong = slice_to_blob(&[1.0f32, 0.0]); // 2-dim, expected 3
    let result = conn.execute("INSERT INTO emb(vector) VALUES(?)", [wrong.as_slice()]);
    assert!(result.is_err());
}

#[test]
fn reject_nan() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)")
        .unwrap();

    let with_nan = slice_to_blob(&[1.0f32, f32::NAN, 3.0]);
    let result = conn.execute("INSERT INTO emb(vector) VALUES(?)", [with_nan.as_slice()]);
    assert!(result.is_err());
}

#[test]
fn knn_search() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)")
        .unwrap();

    // Insert 3 orthogonal unit vectors
    let v1 = slice_to_blob(&[1.0f32, 0.0, 0.0]);
    let v2 = slice_to_blob(&[0.0f32, 1.0, 0.0]);
    let v3 = slice_to_blob(&[0.0f32, 0.0, 1.0]);

    conn.execute("INSERT INTO emb(vector) VALUES(?)", [v1.as_slice()])
        .unwrap();
    conn.execute("INSERT INTO emb(vector) VALUES(?)", [v2.as_slice()])
        .unwrap();
    conn.execute("INSERT INTO emb(vector) VALUES(?)", [v3.as_slice()])
        .unwrap();

    // Query for nearest neighbor to [1, 0, 0] — should return v1 first
    let query = slice_to_blob(&[0.9f32, 0.1, 0.0]);
    let mut stmt = conn
        .prepare("SELECT id, distance FROM emb WHERE knn_match(distance, ?) LIMIT 2")
        .unwrap();

    let rows: Vec<(i64, f64)> = stmt
        .query_map(params![query.as_slice()], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(rows.len(), 2);
    // First result should be closest (v1 at [1,0,0])
    assert_eq!(rows[0].0, 1); // rowid 1
    assert!(
        rows[0].1 < rows[1].1,
        "results should be ordered by distance"
    );
}

#[test]
fn empty_table_scan() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)")
        .unwrap();

    // Full table scan on empty table
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM emb", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0);
}
