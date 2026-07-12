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
