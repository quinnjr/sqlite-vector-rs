mod common;

use common::open_with_extension;
use rusqlite::Connection;

fn create_2d(conn: &Connection) {
    conn.execute_batch("CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2);")
        .unwrap();
}

fn insert_json(conn: &Connection, json: &str) {
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json(?1, 'float4'))",
        [json],
    )
    .unwrap();
}

#[test]
fn update_preserves_rowid_and_index() {
    let conn = open_with_extension();
    create_2d(&conn);
    insert_json(&conn, "[1.0, 0.0]");
    insert_json(&conn, "[0.0, 1.0]");
    conn.execute_batch(
        "UPDATE t SET vector = vector_from_json('[9.0, 9.0]', 'float4') WHERE id = 1;",
    )
    .unwrap();

    let ids: Vec<i64> = conn
        .prepare("SELECT id FROM t ORDER BY id")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids, vec![1, 2], "UPDATE must not renumber rows");

    let nearest: i64 = conn
        .query_row(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[9.0, 9.0]', 'float4')) LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(nearest, 1, "KNN must find the updated vector under its original id");
}

#[test]
fn update_can_change_rowid() {
    let conn = open_with_extension();
    create_2d(&conn);
    insert_json(&conn, "[1.0, 0.0]");
    conn.execute_batch("UPDATE t SET id = 7 WHERE id = 1;").unwrap();
    let id: i64 = conn.query_row("SELECT id FROM t", [], |r| r.get(0)).unwrap();
    assert_eq!(id, 7);
    let nearest: i64 = conn
        .query_row(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[1.0, 0.0]', 'float4')) LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(nearest, 7, "index must be re-keyed when the rowid changes");
}

fn create_2d_with_metadata(conn: &Connection) {
    conn.execute_batch(
        "CREATE VIRTUAL TABLE mu USING vector(dim=2, type=float4, metric=l2, metadata=\"label TEXT\");",
    )
    .unwrap();
}

fn insert_json_with_label(conn: &Connection, json: &str, label: &str) {
    conn.execute(
        "INSERT INTO mu(vector, label) VALUES (vector_from_json(?1, 'float4'), ?2)",
        rusqlite::params![json, label],
    )
    .unwrap();
}

#[test]
fn update_vector_only_retains_metadata() {
    let conn = open_with_extension();
    create_2d_with_metadata(&conn);
    insert_json_with_label(&conn, "[1.0, 0.0]", "a");

    conn.execute_batch(
        "UPDATE mu SET vector = vector_from_json('[9.0, 9.0]', 'float4') WHERE id = 1;",
    )
    .unwrap();

    let label: String = conn
        .query_row("SELECT CAST(label AS TEXT) FROM mu WHERE id = 1", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(label, "a", "metadata must survive a vector-only UPDATE");
}

#[test]
fn rollback_on_fresh_connection_discards_index_entries() {
    let conn = open_with_extension();
    create_2d(&conn);
    conn.execute_batch(
        "BEGIN;
         INSERT INTO t(vector) VALUES (vector_from_json('[0.0, 0.0]', 'float4'));
         INSERT INTO t(vector) VALUES (vector_from_json('[0.1, 0.1]', 'float4'));
         ROLLBACK;",
    )
    .unwrap();
    // Rowids 1..2 are reused after rollback; inserts must not hit duplicate keys.
    insert_json(&conn, "[100.0, 100.0]");
    insert_json(&conn, "[101.0, 101.0]");
    let n: i64 = conn
        .query_row(
            "SELECT count(*) FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 10",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 2, "index must contain exactly the committed rows");
}

#[test]
fn rollback_to_savepoint_restores_index_to_savepoint() {
    let conn = open_with_extension();
    create_2d(&conn);
    insert_json(&conn, "[1.0, 1.0]");
    conn.execute_batch(
        "BEGIN;
         INSERT INTO t(vector) VALUES (vector_from_json('[2.0, 2.0]', 'float4'));
         SAVEPOINT sp1;
         INSERT INTO t(vector) VALUES (vector_from_json('[3.0, 3.0]', 'float4'));
         ROLLBACK TO sp1;
         COMMIT;",
    )
    .unwrap();
    let n: i64 = conn
        .query_row(
            "SELECT count(*) FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 10",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 2, "index must reflect rows 1 and 2 only");
}

#[test]
fn update_metadata_only_retains_vector() {
    let conn = open_with_extension();
    create_2d_with_metadata(&conn);
    insert_json_with_label(&conn, "[1.0, 0.0]", "a");

    conn.execute_batch("UPDATE mu SET label = 'b' WHERE id = 1;")
        .unwrap();

    let nearest: i64 = conn
        .query_row(
            "SELECT id FROM mu WHERE knn_match(distance, vector_from_json('[1.0, 0.0]', 'float4')) LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        nearest, 1,
        "vector must survive a metadata-only UPDATE (KNN still finds original vector)"
    );

    let label: String = conn
        .query_row("SELECT CAST(label AS TEXT) FROM mu WHERE id = 1", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(label, "b");
}

#[test]
fn update_metadata_to_null_sets_null() {
    let conn = open_with_extension();
    create_2d_with_metadata(&conn);
    insert_json_with_label(&conn, "[1.0, 0.0]", "a");

    conn.execute_batch("UPDATE mu SET label = NULL WHERE id = 1;")
        .unwrap();

    let is_null: bool = conn
        .query_row("SELECT label IS NULL FROM mu WHERE id = 1", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert!(is_null, "explicit SET label = NULL must actually null the column");
}

#[test]
fn update_id_only_retains_vector_and_metadata() {
    let conn = open_with_extension();
    create_2d_with_metadata(&conn);
    insert_json_with_label(&conn, "[1.0, 0.0]", "a");

    conn.execute_batch("UPDATE mu SET id = 7 WHERE id = 1;")
        .unwrap();

    let label: String = conn
        .query_row("SELECT CAST(label AS TEXT) FROM mu WHERE id = 7", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(label, "a", "metadata must survive an id-only UPDATE");

    let nearest: i64 = conn
        .query_row(
            "SELECT id FROM mu WHERE knn_match(distance, vector_from_json('[1.0, 0.0]', 'float4')) LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        nearest, 7,
        "vector must survive an id-only UPDATE, re-keyed under the new id"
    );
}

#[test]
fn insert_with_explicit_rowid() {
    let conn = open_with_extension();
    create_2d(&conn);
    conn.execute_batch(
        "INSERT INTO t(id, vector) VALUES (42, vector_from_json('[2.0, 2.0]', 'float4'));",
    )
    .unwrap();
    let id: i64 = conn.query_row("SELECT id FROM t", [], |r| r.get(0)).unwrap();
    assert_eq!(id, 42);
    let nearest: i64 = conn
        .query_row(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[2.0, 2.0]', 'float4')) LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(nearest, 42);
}

#[test]
fn insert_duplicate_rowid_errors() {
    let conn = open_with_extension();
    create_2d(&conn);
    conn.execute_batch(
        "INSERT INTO t(id, vector) VALUES (5, vector_from_json('[1.0, 0.0]', 'float4'));",
    )
    .unwrap();
    let err = conn.execute_batch(
        "INSERT INTO t(id, vector) VALUES (5, vector_from_json('[0.0, 1.0]', 'float4'));",
    );
    assert!(err.is_err(), "duplicate explicit rowid must be a constraint error");
}

#[test]
fn metadata_columns_keep_declared_types() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE m USING vector(dim=2, type=float4, metric=l2, metadata=\"label TEXT, score REAL\");
         INSERT INTO m(vector, label, score) VALUES (vector_from_json('[0.0, 0.0]', 'float4'), 'a', 1.5);",
    )
    .unwrap();
    let (t_label, t_score): (String, String) = conn
        .query_row("SELECT typeof(label), typeof(score) FROM m", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!((t_label.as_str(), t_score.as_str()), ("text", "real"));

    let n: i64 = conn
        .query_row("SELECT count(*) FROM m WHERE label = 'a' AND score > 1.0", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 1, "metadata filters must match typed values");

    // KNN mode must preserve types too.
    let t_knn: String = conn
        .query_row(
            "SELECT typeof(label) FROM m WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(t_knn, "text");
}
