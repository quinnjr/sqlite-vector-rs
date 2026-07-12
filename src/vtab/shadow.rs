use crate::vtab::config::VectorTableConfig;

/// SQL statements for shadow table management.
pub struct ShadowOps;

impl ShadowOps {
    pub fn create_data_table_sql(config: &VectorTableConfig) -> String {
        let mut cols = vec![
            "id INTEGER PRIMARY KEY AUTOINCREMENT".to_string(),
            "vector BLOB NOT NULL".to_string(),
        ];
        for (name, sql_type) in &config.metadata_columns {
            cols.push(format!("{name} {sql_type}"));
        }
        format!(
            "CREATE TABLE IF NOT EXISTS \"{}_data\"({})",
            config.table_name,
            cols.join(", ")
        )
    }

    pub fn create_index_table_sql(config: &VectorTableConfig) -> String {
        format!(
            "CREATE TABLE IF NOT EXISTS \"{}_index\"(key TEXT PRIMARY KEY, value BLOB)",
            config.table_name
        )
    }

    pub fn drop_shadow_tables_sql(table_name: &str) -> Vec<String> {
        vec![
            format!("DROP TABLE IF EXISTS \"{table_name}_data\""),
            format!("DROP TABLE IF EXISTS \"{table_name}_index\""),
        ]
    }

    pub fn insert_data_sql(config: &VectorTableConfig) -> String {
        let mut col_names = vec!["vector".to_string()];
        let mut placeholders = vec!["?".to_string()];
        for (name, _) in &config.metadata_columns {
            col_names.push(name.clone());
            placeholders.push("?".to_string());
        }
        format!(
            "INSERT INTO \"{}_data\"({}) VALUES({})",
            config.table_name,
            col_names.join(", "),
            placeholders.join(", ")
        )
    }

    pub fn insert_vector_only_sql(table_name: &str) -> String {
        format!("INSERT INTO \"{table_name}_data\"(vector) VALUES(?)")
    }

    pub fn update_data_sql(config: &VectorTableConfig) -> String {
        let mut sets = vec!["id = ?".to_string(), "vector = ?".to_string()];
        for (name, _) in &config.metadata_columns {
            sets.push(format!("{name} = ?"));
        }
        format!(
            "UPDATE \"{}_data\" SET {} WHERE id = ?",
            config.table_name,
            sets.join(", ")
        )
    }

    pub fn delete_data_sql(table_name: &str) -> String {
        format!("DELETE FROM \"{table_name}_data\" WHERE id = ?")
    }

    pub fn select_data_sql(table_name: &str) -> String {
        format!("SELECT * FROM \"{table_name}_data\" WHERE id = ?")
    }

    pub fn select_all_data_sql(table_name: &str) -> String {
        format!("SELECT * FROM \"{table_name}_data\"")
    }

    pub fn upsert_index_sql(table_name: &str) -> String {
        format!("INSERT OR REPLACE INTO \"{table_name}_index\"(key, value) VALUES(?, ?)")
    }

    pub fn select_index_sql(table_name: &str) -> String {
        format!("SELECT value FROM \"{table_name}_index\" WHERE key = ?")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vtab::config::VectorTableConfig;

    fn base_config() -> VectorTableConfig {
        VectorTableConfig::parse(&["vector", "main", "emb", "dim=3"]).unwrap()
    }

    fn config_with_metadata() -> VectorTableConfig {
        VectorTableConfig::parse(&[
            "vector",
            "main",
            "emb",
            "dim=3",
            "metadata=label TEXT,score REAL",
        ])
        .unwrap()
    }

    // --- create_data_table_sql ---

    #[test]
    fn create_data_table_sql_no_metadata_contains_table_name() {
        let sql = ShadowOps::create_data_table_sql(&base_config());
        assert!(sql.contains("emb_data"), "expected 'emb_data' in: {sql}");
    }

    #[test]
    fn create_data_table_sql_no_metadata_has_id_column() {
        let sql = ShadowOps::create_data_table_sql(&base_config());
        assert!(
            sql.contains("id INTEGER PRIMARY KEY AUTOINCREMENT"),
            "expected id column in: {sql}"
        );
    }

    #[test]
    fn create_data_table_sql_no_metadata_has_vector_column() {
        let sql = ShadowOps::create_data_table_sql(&base_config());
        assert!(
            sql.contains("vector BLOB NOT NULL"),
            "expected vector column in: {sql}"
        );
    }

    #[test]
    fn create_data_table_sql_no_metadata_exact() {
        let sql = ShadowOps::create_data_table_sql(&base_config());
        assert_eq!(
            sql,
            "CREATE TABLE IF NOT EXISTS \"emb_data\"(id INTEGER PRIMARY KEY AUTOINCREMENT, vector BLOB NOT NULL)"
        );
    }

    #[test]
    fn create_data_table_sql_with_metadata_contains_label_column() {
        let sql = ShadowOps::create_data_table_sql(&config_with_metadata());
        assert!(
            sql.contains("label TEXT"),
            "expected 'label TEXT' in: {sql}"
        );
    }

    #[test]
    fn create_data_table_sql_with_metadata_contains_score_column() {
        let sql = ShadowOps::create_data_table_sql(&config_with_metadata());
        assert!(
            sql.contains("score REAL"),
            "expected 'score REAL' in: {sql}"
        );
    }

    #[test]
    fn create_data_table_sql_with_metadata_exact() {
        let sql = ShadowOps::create_data_table_sql(&config_with_metadata());
        assert_eq!(
            sql,
            "CREATE TABLE IF NOT EXISTS \"emb_data\"(id INTEGER PRIMARY KEY AUTOINCREMENT, vector BLOB NOT NULL, label TEXT, score REAL)"
        );
    }

    // --- create_index_table_sql ---

    #[test]
    fn create_index_table_sql_contains_table_name() {
        let sql = ShadowOps::create_index_table_sql(&base_config());
        assert!(sql.contains("emb_index"), "expected 'emb_index' in: {sql}");
    }

    #[test]
    fn create_index_table_sql_has_key_column() {
        let sql = ShadowOps::create_index_table_sql(&base_config());
        assert!(
            sql.contains("key TEXT PRIMARY KEY"),
            "expected 'key TEXT PRIMARY KEY' in: {sql}"
        );
    }

    #[test]
    fn create_index_table_sql_has_value_column() {
        let sql = ShadowOps::create_index_table_sql(&base_config());
        assert!(
            sql.contains("value BLOB"),
            "expected 'value BLOB' in: {sql}"
        );
    }

    #[test]
    fn create_index_table_sql_exact() {
        let sql = ShadowOps::create_index_table_sql(&base_config());
        assert_eq!(
            sql,
            "CREATE TABLE IF NOT EXISTS \"emb_index\"(key TEXT PRIMARY KEY, value BLOB)"
        );
    }

    // --- drop_shadow_tables_sql ---

    #[test]
    fn drop_shadow_tables_sql_returns_two_statements() {
        let stmts = ShadowOps::drop_shadow_tables_sql("emb");
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn drop_shadow_tables_sql_drops_data_table() {
        let stmts = ShadowOps::drop_shadow_tables_sql("emb");
        assert!(
            stmts[0].contains("emb_data"),
            "expected 'emb_data' in: {}",
            stmts[0]
        );
        assert_eq!(stmts[0], "DROP TABLE IF EXISTS \"emb_data\"");
    }

    #[test]
    fn drop_shadow_tables_sql_drops_index_table() {
        let stmts = ShadowOps::drop_shadow_tables_sql("emb");
        assert!(
            stmts[1].contains("emb_index"),
            "expected 'emb_index' in: {}",
            stmts[1]
        );
        assert_eq!(stmts[1], "DROP TABLE IF EXISTS \"emb_index\"");
    }

    // --- insert_data_sql ---

    #[test]
    fn insert_data_sql_no_metadata_exact() {
        let sql = ShadowOps::insert_data_sql(&base_config());
        assert_eq!(sql, "INSERT INTO \"emb_data\"(vector) VALUES(?)");
    }

    #[test]
    fn insert_data_sql_with_metadata_contains_label() {
        let sql = ShadowOps::insert_data_sql(&config_with_metadata());
        assert!(sql.contains("label"), "expected 'label' in: {sql}");
    }

    #[test]
    fn insert_data_sql_with_metadata_contains_score() {
        let sql = ShadowOps::insert_data_sql(&config_with_metadata());
        assert!(sql.contains("score"), "expected 'score' in: {sql}");
    }

    #[test]
    fn insert_data_sql_with_metadata_has_correct_placeholder_count() {
        let sql = ShadowOps::insert_data_sql(&config_with_metadata());
        // vector + label + score = 3 placeholders
        let placeholder_count = sql.matches('?').count();
        assert_eq!(placeholder_count, 3, "expected 3 placeholders in: {sql}");
    }

    #[test]
    fn insert_data_sql_with_metadata_exact() {
        let sql = ShadowOps::insert_data_sql(&config_with_metadata());
        assert_eq!(
            sql,
            "INSERT INTO \"emb_data\"(vector, label, score) VALUES(?, ?, ?)"
        );
    }

    // --- insert_vector_only_sql ---

    #[test]
    fn insert_vector_only_sql_exact() {
        let sql = ShadowOps::insert_vector_only_sql("emb");
        assert_eq!(sql, "INSERT INTO \"emb_data\"(vector) VALUES(?)");
    }

    // --- delete_data_sql ---

    #[test]
    fn delete_data_sql_exact() {
        let sql = ShadowOps::delete_data_sql("emb");
        assert_eq!(sql, "DELETE FROM \"emb_data\" WHERE id = ?");
    }

    // --- select_data_sql ---

    #[test]
    fn select_data_sql_exact() {
        let sql = ShadowOps::select_data_sql("emb");
        assert_eq!(sql, "SELECT * FROM \"emb_data\" WHERE id = ?");
    }

    // --- select_all_data_sql ---

    #[test]
    fn select_all_data_sql_exact() {
        let sql = ShadowOps::select_all_data_sql("emb");
        assert_eq!(sql, "SELECT * FROM \"emb_data\"");
    }

    #[test]
    fn select_all_data_sql_no_where_clause() {
        let sql = ShadowOps::select_all_data_sql("emb");
        assert!(!sql.contains("WHERE"), "unexpected WHERE clause in: {sql}");
    }

    // --- upsert_index_sql ---

    #[test]
    fn upsert_index_sql_exact() {
        let sql = ShadowOps::upsert_index_sql("emb");
        assert_eq!(
            sql,
            "INSERT OR REPLACE INTO \"emb_index\"(key, value) VALUES(?, ?)"
        );
    }

    #[test]
    fn upsert_index_sql_contains_insert_or_replace() {
        let sql = ShadowOps::upsert_index_sql("emb");
        assert!(
            sql.contains("INSERT OR REPLACE INTO \"emb_index\""),
            "expected INSERT OR REPLACE into emb_index in: {sql}"
        );
    }

    // --- select_index_sql ---

    #[test]
    fn select_index_sql_exact() {
        let sql = ShadowOps::select_index_sql("emb");
        assert_eq!(sql, "SELECT value FROM \"emb_index\" WHERE key = ?");
    }

    // --- special characters in table name ---

    #[test]
    fn special_table_name_data_table() {
        let sql = ShadowOps::create_data_table_sql(
            &VectorTableConfig::parse(&["vector", "main", "my_table", "dim=3"]).unwrap(),
        );
        assert!(
            sql.contains("my_table_data"),
            "expected 'my_table_data' in: {sql}"
        );
    }

    #[test]
    fn special_table_name_index_table() {
        let sql = ShadowOps::create_index_table_sql(
            &VectorTableConfig::parse(&["vector", "main", "my_table", "dim=3"]).unwrap(),
        );
        assert!(
            sql.contains("my_table_index"),
            "expected 'my_table_index' in: {sql}"
        );
    }

    #[test]
    fn special_table_name_drop_shadow_tables() {
        let stmts = ShadowOps::drop_shadow_tables_sql("my_table");
        assert_eq!(stmts[0], "DROP TABLE IF EXISTS \"my_table_data\"");
        assert_eq!(stmts[1], "DROP TABLE IF EXISTS \"my_table_index\"");
    }

    #[test]
    fn special_table_name_insert_vector_only() {
        let sql = ShadowOps::insert_vector_only_sql("my_table");
        assert_eq!(sql, "INSERT INTO \"my_table_data\"(vector) VALUES(?)");
    }

    #[test]
    fn special_table_name_delete_data() {
        let sql = ShadowOps::delete_data_sql("my_table");
        assert_eq!(sql, "DELETE FROM \"my_table_data\" WHERE id = ?");
    }

    #[test]
    fn special_table_name_select_data() {
        let sql = ShadowOps::select_data_sql("my_table");
        assert_eq!(sql, "SELECT * FROM \"my_table_data\" WHERE id = ?");
    }

    #[test]
    fn special_table_name_select_all_data() {
        let sql = ShadowOps::select_all_data_sql("my_table");
        assert_eq!(sql, "SELECT * FROM \"my_table_data\"");
    }

    #[test]
    fn special_table_name_upsert_index() {
        let sql = ShadowOps::upsert_index_sql("my_table");
        assert_eq!(
            sql,
            "INSERT OR REPLACE INTO \"my_table_index\"(key, value) VALUES(?, ?)"
        );
    }

    #[test]
    fn special_table_name_select_index() {
        let sql = ShadowOps::select_index_sql("my_table");
        assert_eq!(sql, "SELECT value FROM \"my_table_index\" WHERE key = ?");
    }
}
