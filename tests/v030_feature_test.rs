mod common;

use common::{open_file_with_extension, open_with_extension};

#[test]
fn unpersisted_commits_survive_reconnect_via_reconcile() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.db");
    {
        let conn = open_file_with_extension(&path);
        conn.execute_batch(
            "CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2, sync_every=1000000);",
        )
        .unwrap();
        for i in 0..10 {
            conn.execute(
                "INSERT INTO t(vector) VALUES (vector_from_json(?1, 'float4'))",
                [format!("[{i}.0, 0.0]")],
            )
            .unwrap();
        }
    } // dropped without ever reaching the persist threshold
    let conn = open_file_with_extension(&path);
    let n: i64 = conn
        .query_row(
            "SELECT count(*) FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 20",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 10, "reconcile must recover rows committed after the last persist");
}

#[test]
fn stale_deletes_trigger_rebuild_on_connect() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.db");
    {
        let conn = open_file_with_extension(&path);
        conn.execute_batch(
            "CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2, sync_every=1000000);",
        )
        .unwrap();
        for i in 0..5 {
            conn.execute(
                "INSERT INTO t(vector) VALUES (vector_from_json(?1, 'float4'))",
                [format!("[{i}.0, 0.0]")],
            )
            .unwrap();
        }
        conn.query_row("SELECT vector_sync_index('t')", [], |r| r.get::<_, i64>(0))
            .unwrap(); // graph persisted with 5 keys
        conn.execute("DELETE FROM t WHERE id <= 2", []).unwrap(); // not persisted
    }
    let conn = open_file_with_extension(&path);
    let ids: Vec<i64> = conn
        .prepare(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 10",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids.len(), 3, "stale graph keys must be dropped by rebuild");
    assert!(ids.iter().all(|id| *id >= 3));
}

#[test]
fn vector_sync_index_persists_graph_and_state() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2, sync_every=1000000);
         INSERT INTO t(vector) VALUES (vector_from_json('[1.0, 0.0]', 'float4'));",
    )
    .unwrap();
    conn.query_row("SELECT vector_sync_index('t')", [], |r| r.get::<_, i64>(0))
        .unwrap();
    let has_graph: i64 = conn
        .query_row("SELECT count(*) FROM t_index WHERE key = 'hnsw_graph'", [], |r| r.get(0))
        .unwrap();
    let state: String = conn
        .query_row("SELECT value FROM t_index WHERE key = 'graph_state'", [], |r| r.get(0))
        .unwrap();
    assert_eq!(has_graph, 1);
    let v: serde_json::Value = serde_json::from_str(&state).unwrap();
    assert_eq!(v["row_count"], 1);
}

#[test]
fn full_scan_streams_all_rows_in_order() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE s USING vector(dim=2, type=float4, metric=l2);")
        .unwrap();
    for i in 0..200 {
        conn.execute(
            "INSERT INTO s(vector) VALUES (vector_from_json(?1, 'float4'))",
            [format!("[{i}.0, 0.0]")],
        )
        .unwrap();
    }
    let ids: Vec<i64> = conn
        .prepare("SELECT id FROM s")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids.len(), 200);
    assert_eq!(ids.first(), Some(&1));
    assert_eq!(ids.last(), Some(&200));
}
