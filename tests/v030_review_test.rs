mod common;

use common::open_with_extension;

/// Regression test for a code-review finding: a huge `LIMIT` fed straight into
/// `k` as `usize` caused `BinaryHeap::with_capacity`/`Vec::with_capacity` to
/// attempt a multi-hundred-GB allocation, which panics across SQLite's C FFI
/// boundary and aborts the host process (verified via the `sqlite3` CLI on
/// pre-fix HEAD: `memory allocation of 320000000000 bytes failed` + SIGABRT,
/// for both hnsw and mode=exact tables). A giant LIMIT is legal SQL meaning
/// "no limit" in practice here, so the fixed behavior is to return all
/// matching rows rather than erroring.
#[test]
fn knn_huge_limit_does_not_abort_hnsw() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2);")
        .unwrap();
    for i in 0..5 {
        conn.execute(
            "INSERT INTO t(vector) VALUES (vector_from_json(?1, 'float4'))",
            [format!("[{i}.0, 0.0]")],
        )
        .unwrap();
    }
    let ids: Vec<i64> = conn
        .prepare(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 5000000000",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids.len(), 5, "huge LIMIT should return all rows, not crash");
}

#[test]
fn knn_huge_limit_does_not_abort_exact() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2, mode=exact);",
    )
    .unwrap();
    for i in 0..5 {
        conn.execute(
            "INSERT INTO t(vector) VALUES (vector_from_json(?1, 'float4'))",
            [format!("[{i}.0, 0.0]")],
        )
        .unwrap();
    }
    let ids: Vec<i64> = conn
        .prepare(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 5000000000",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids.len(), 5, "huge LIMIT should return all rows, not crash");
}

/// A negative SQL LIMIT means "no limit" in SQLite; make sure knn_match
/// interprets it the same way instead of underflowing `k` to `usize::MAX`.
#[test]
fn knn_negative_limit_returns_rows() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2);")
        .unwrap();
    for i in 0..5 {
        conn.execute(
            "INSERT INTO t(vector) VALUES (vector_from_json(?1, 'float4'))",
            [format!("[{i}.0, 0.0]")],
        )
        .unwrap();
    }
    let ids: Vec<i64> = conn
        .prepare(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT -1",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids.len(), 5, "negative LIMIT should return all rows");
}

/// Sibling to `knn_negative_limit_returns_rows`, for a `mode=exact` table:
/// negative LIMIT must mean "no limit" there too, not underflow `k`.
#[test]
fn knn_negative_limit_returns_rows_exact() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE e USING vector(dim=2, type=float4, metric=l2, mode=exact);",
    )
    .unwrap();
    for i in 0..6 {
        conn.execute(
            "INSERT INTO e(vector) VALUES (vector_from_json(?1, 'float4'))",
            [format!("[{i}.0, 0.0]")],
        )
        .unwrap();
    }
    let ids: Vec<i64> = conn
        .prepare(
            "SELECT id FROM e WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT -1",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids.len(), 6, "negative LIMIT should return all rows (mode=exact)");
}
