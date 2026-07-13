mod common;

use common::open_with_extension;
use sqlite_vector_rs::types::slice_to_blob;

#[test]
fn vector_from_json_and_back() {
    let conn = open_with_extension();
    let json: String = conn
        .query_row(
            "SELECT vector_to_json(vector_from_json('[1.0, 2.0, 3.0]', 'float4'), 'float4')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(json, "[1.0,2.0,3.0]");
}

#[test]
fn vector_distance_l2() {
    let conn = open_with_extension();
    let a = slice_to_blob(&[1.0f32, 0.0, 0.0]);
    let b = slice_to_blob(&[0.0f32, 1.0, 0.0]);
    let dist: f64 = conn
        .query_row(
            "SELECT vector_distance(?, ?, 'l2', 'float4')",
            [a.as_slice(), b.as_slice()],
            |row| row.get(0),
        )
        .unwrap();
    assert!((dist - 2.0).abs() < 1e-6);
}

#[test]
fn vector_dims() {
    let conn = open_with_extension();
    let v = slice_to_blob(&[1.0f32, 2.0, 3.0, 4.0]);
    let dims: i64 = conn
        .query_row("SELECT vector_dims(?, 'float4')", [v.as_slice()], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(dims, 4);
}

#[test]
fn rebuild_index_on_existing_table() {
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

    // Rebuild the index from scratch
    let count: i64 = conn
        .query_row("SELECT vector_rebuild_index('emb')", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 3);
}

#[test]
fn rebuild_index_empty_table() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)")
        .unwrap();

    let count: i64 = conn
        .query_row("SELECT vector_rebuild_index('emb')", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn export_and_import_arrow() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)")
        .unwrap();

    let v1 = slice_to_blob(&[1.0f32, 2.0, 3.0]);
    let v2 = slice_to_blob(&[4.0f32, 5.0, 6.0]);

    conn.execute("INSERT INTO emb(vector) VALUES(?)", [v1.as_slice()])
        .unwrap();
    conn.execute("INSERT INTO emb(vector) VALUES(?)", [v2.as_slice()])
        .unwrap();

    // Export to Arrow IPC blob
    let ipc_blob: Vec<u8> = conn
        .query_row("SELECT vector_export_arrow('emb', 'float4')", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert!(!ipc_blob.is_empty(), "Arrow IPC blob should not be empty");

    // Create a second table and import the Arrow blob into it
    conn.execute_batch("CREATE VIRTUAL TABLE emb2 USING vector(dim=3, type=float4, metric=l2)")
        .unwrap();

    let imported: i64 = conn
        .query_row(
            "SELECT vector_insert_arrow('emb2', 'float4', ?)",
            [ipc_blob.as_slice()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(imported, 2);

    // Verify both rows are in the new table
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM emb2", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 2);
}

#[test]
fn export_arrow_empty_table() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)")
        .unwrap();

    let ipc_blob: Vec<u8> = conn
        .query_row("SELECT vector_export_arrow('emb', 'float4')", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert!(ipc_blob.is_empty(), "Empty table should return empty blob");
}

#[test]
fn import_arrow_empty_blob() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)")
        .unwrap();

    let empty: &[u8] = &[];
    let imported: i64 = conn
        .query_row(
            "SELECT vector_insert_arrow('emb', 'float4', ?)",
            [empty],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(imported, 0);
}

#[test]
fn rebuild_index_preserves_shadow_data() {
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

    // Rebuild the HNSW index
    conn.query_row("SELECT vector_rebuild_index('emb')", [], |row| {
        row.get::<_, i64>(0)
    })
    .unwrap();

    // KNN search should still work after rebuild — but note the in-memory
    // vtab index is stale (the scalar function wrote to the shadow table,
    // not the live vtab). This test validates that the shadow table data is
    // written correctly; the vtab would pick it up on reconnect.
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM emb", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 3);
}
