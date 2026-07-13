mod common;

use common::open_with_extension;
use rusqlite::params;

/// All 6 element types supported by the vector extension.
const TYPES: &[&str] = &["float2", "float4", "float8", "int1", "int2", "int4"];

/// All 3 distance metrics supported by the vector extension.
const METRICS: &[&str] = &["l2", "cosine", "ip"];

/// JSON array used for insert and query.
///
/// For cosine metric, zero vectors produce undefined distance, so we use
/// non-zero values throughout. For integer types the values are cast at
/// insert time (e.g. i8 for int1), so we keep them small and in-range.
const TEST_VECTOR_JSON: &str = "[1,2,3,4]";
const QUERY_VECTOR_JSON: &str = "[1,2,3,5]";

/// Both index modes supported by the vector extension.
const MODES: &[&str] = &["hnsw", "exact"];

/// Run a single (type, metric, mode) combination:
///   1. Create a virtual table with dim=4, the given type, metric, and mode.
///   2. Insert one vector via `vector_from_json`.
///   3. Perform a KNN query and verify at least one result is returned.
///
/// Exact and HNSW must agree on these small fixtures, so the assertion
/// bodies are identical regardless of mode.
fn run_combination(vec_type: &str, metric: &str, mode: &str) {
    let conn = open_with_extension();

    let create_sql = format!(
        "CREATE VIRTUAL TABLE emb USING vector(dim=4, type={vec_type}, metric={metric}, mode={mode})"
    );
    conn.execute_batch(&create_sql)
        .unwrap_or_else(|e| panic!("CREATE failed for ({vec_type}, {metric}, {mode}): {e}"));

    // Insert a vector using vector_from_json so the blob is correctly typed.
    let insert_sql = format!(
        "INSERT INTO emb(vector) VALUES(vector_from_json('{TEST_VECTOR_JSON}', '{vec_type}'))"
    );
    conn.execute_batch(&insert_sql)
        .unwrap_or_else(|e| panic!("INSERT failed for ({vec_type}, {metric}, {mode}): {e}"));

    // Build the query blob via vector_from_json.
    let query_blob: Vec<u8> = conn
        .query_row(
            &format!("SELECT vector_from_json('{QUERY_VECTOR_JSON}', '{vec_type}')"),
            [],
            |row| row.get(0),
        )
        .unwrap_or_else(|e| {
            panic!("vector_from_json failed for ({vec_type}, {metric}, {mode}): {e}")
        });

    // KNN search: expect exactly 1 result (we only inserted 1 vector).
    let mut stmt = conn
        .prepare("SELECT id, distance FROM emb WHERE knn_match(distance, ?) LIMIT 5")
        .unwrap_or_else(|e| panic!("PREPARE failed for ({vec_type}, {metric}, {mode}): {e}"));

    let rows: Vec<(i64, f64)> = stmt
        .query_map(params![query_blob.as_slice()], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap_or_else(|e| panic!("query_map failed for ({vec_type}, {metric}, {mode}): {e}"))
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|e| panic!("row collect failed for ({vec_type}, {metric}, {mode}): {e}"));

    assert!(
        !rows.is_empty(),
        "KNN returned no results for ({vec_type}, {metric}, {mode})"
    );
    assert_eq!(
        rows[0].0, 1,
        "Expected rowid 1 for ({vec_type}, {metric}, {mode}), got {}",
        rows[0].0
    );
}

// ── float2 ────────────────────────────────────────────────────────────────────

#[test]
fn type_float2_metric_l2() {
    run_combination("float2", "l2", "hnsw");
}

#[test]
fn type_float2_metric_l2_exact() {
    run_combination("float2", "l2", "exact");
}

#[test]
fn type_float2_metric_cosine() {
    run_combination("float2", "cosine", "hnsw");
}

#[test]
fn type_float2_metric_cosine_exact() {
    run_combination("float2", "cosine", "exact");
}

#[test]
fn type_float2_metric_ip() {
    run_combination("float2", "ip", "hnsw");
}

#[test]
fn type_float2_metric_ip_exact() {
    run_combination("float2", "ip", "exact");
}

// ── float4 ────────────────────────────────────────────────────────────────────

#[test]
fn type_float4_metric_l2() {
    run_combination("float4", "l2", "hnsw");
}

#[test]
fn type_float4_metric_l2_exact() {
    run_combination("float4", "l2", "exact");
}

#[test]
fn type_float4_metric_cosine() {
    run_combination("float4", "cosine", "hnsw");
}

#[test]
fn type_float4_metric_cosine_exact() {
    run_combination("float4", "cosine", "exact");
}

#[test]
fn type_float4_metric_ip() {
    run_combination("float4", "ip", "hnsw");
}

#[test]
fn type_float4_metric_ip_exact() {
    run_combination("float4", "ip", "exact");
}

// ── float8 ────────────────────────────────────────────────────────────────────

#[test]
fn type_float8_metric_l2() {
    run_combination("float8", "l2", "hnsw");
}

#[test]
fn type_float8_metric_l2_exact() {
    run_combination("float8", "l2", "exact");
}

#[test]
fn type_float8_metric_cosine() {
    run_combination("float8", "cosine", "hnsw");
}

#[test]
fn type_float8_metric_cosine_exact() {
    run_combination("float8", "cosine", "exact");
}

#[test]
fn type_float8_metric_ip() {
    run_combination("float8", "ip", "hnsw");
}

#[test]
fn type_float8_metric_ip_exact() {
    run_combination("float8", "ip", "exact");
}

// ── int1 ────────────────────────────────────────────────────────────────────

#[test]
fn type_int1_metric_l2() {
    run_combination("int1", "l2", "hnsw");
}

#[test]
fn type_int1_metric_l2_exact() {
    run_combination("int1", "l2", "exact");
}

#[test]
fn type_int1_metric_cosine() {
    run_combination("int1", "cosine", "hnsw");
}

#[test]
fn type_int1_metric_cosine_exact() {
    run_combination("int1", "cosine", "exact");
}

#[test]
fn type_int1_metric_ip() {
    run_combination("int1", "ip", "hnsw");
}

#[test]
fn type_int1_metric_ip_exact() {
    run_combination("int1", "ip", "exact");
}

// ── int2 ────────────────────────────────────────────────────────────────────

#[test]
fn type_int2_metric_l2() {
    run_combination("int2", "l2", "hnsw");
}

#[test]
fn type_int2_metric_l2_exact() {
    run_combination("int2", "l2", "exact");
}

#[test]
fn type_int2_metric_cosine() {
    run_combination("int2", "cosine", "hnsw");
}

#[test]
fn type_int2_metric_cosine_exact() {
    run_combination("int2", "cosine", "exact");
}

#[test]
fn type_int2_metric_ip() {
    run_combination("int2", "ip", "hnsw");
}

#[test]
fn type_int2_metric_ip_exact() {
    run_combination("int2", "ip", "exact");
}

// ── int4 ────────────────────────────────────────────────────────────────────

#[test]
fn type_int4_metric_l2() {
    run_combination("int4", "l2", "hnsw");
}

#[test]
fn type_int4_metric_l2_exact() {
    run_combination("int4", "l2", "exact");
}

#[test]
fn type_int4_metric_cosine() {
    run_combination("int4", "cosine", "hnsw");
}

#[test]
fn type_int4_metric_cosine_exact() {
    run_combination("int4", "cosine", "exact");
}

#[test]
fn type_int4_metric_ip() {
    run_combination("int4", "ip", "hnsw");
}

#[test]
fn type_int4_metric_ip_exact() {
    run_combination("int4", "ip", "exact");
}

/// Smoke test: all 36 combinations (6 types x 3 metrics x 2 modes) are
/// covered by the individual tests above. This test documents the expected
/// matrix shape without re-running them.
#[test]
fn type_matrix_coverage() {
    assert_eq!(TYPES.len(), 6, "expected 6 element types");
    assert_eq!(METRICS.len(), 3, "expected 3 distance metrics");
    assert_eq!(MODES.len(), 2, "expected 2 index modes");
    assert_eq!(
        TYPES.len() * METRICS.len() * MODES.len(),
        36,
        "expected 36 combinations"
    );
}
